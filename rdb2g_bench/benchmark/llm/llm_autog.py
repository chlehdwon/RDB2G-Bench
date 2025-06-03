import json
import os

from pathlib import Path
from typing import Dict
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from common.text_embedder import GloveTextEmbedding
from common.search_space.gnn_search_space import GNNNodeSearchSpace, IDGNNLinkSearchSpace
from common.search_space.search_space import TotalSearchSpace

from benchmark.dataset import PerformancePredictionDataset
from benchmark.llm.llm_micro_action import LLMMicroActionSet
from benchmark.llm.llm_utils import get_edge_info, get_budget, get_available_edges, conduct_multiple_actions, check_none_action, update_edge_set, update_score, update_action, remove_invalid_history_actions, get_history_actions

from openai import OpenAI
import json
import argparse
from prompts.prompt import type_infer_prompt
from prompts.prompt import augmentation_prompt
import os
import ast

os.environ["ANTHROPIC_API_KEY"] ="YOUR_API_KEY"

import anthropic
client = anthropic.Anthropic()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-top3")
    parser.add_argument("--budget_percentage", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default='final')
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/relbench_examples"),
    )
    parser.add_argument("--result_dir", type=str, default="../results")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-latest")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--result_file", type=int, default=1)
    return parser.parse_args()


args = parse_args()

#### ARGS ####
initial_budget = get_budget(args.dataset, args.task, args.budget_percentage)
print(f"Dataset: {args.dataset} Task: {args.task} Budget: {initial_budget}")

json_file = f"./outputs-{args.result_file}/{args.dataset}_{args.task}_{args.budget_percentage}.json"
if not os.path.exists(json_file):
    print(f"File {json_file} does not exist. Creating directory.")
    os.makedirs(f"./outputs-{args.result_file}", exist_ok=True)
else :
    print(f"File {json_file} already exists. Exiting.")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)
task_type = task.task_type

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

if task_type == TaskType.BINARY_CLASSIFICATION:
    higher_is_better = True
    gnn_space_class = GNNNodeSearchSpace
    src_table = task.entity_table
    dst_table = None
elif task_type == TaskType.REGRESSION:
    higher_is_better = False
    gnn_space_class = GNNNodeSearchSpace
    src_table = task.entity_table
    dst_table = None
elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
    higher_is_better = True
    gnn_space_class = GNNNodeSearchSpace
    src_table = task.entity_table
    dst_table = None
elif task_type == TaskType.LINK_PREDICTION:
    higher_is_better = True
    gnn_space_class = IDGNNLinkSearchSpace
    # Assuming task object has source_table and destination_table for link prediction
    src_table = task.src_entity_table
    dst_table = task.dst_entity_table
    if src_table is None or dst_table is None:
            raise ValueError("Link prediction task missing source_table or destination_table attribute.")
else:
    raise ValueError(f"Task type {task_type} is unsupported for determining GNNSpaceClass and tables.")

print("Initializing LLMMicroActionSet...")
llm_micro_action_set = LLMMicroActionSet(
    dataset=args.dataset,
    task=args.task,
    hetero_data=data, # Get HeteroData from the dataset instance
    GNNSpaceClass=gnn_space_class,
    num_layers=2,
    src_entity_table=src_table,
    dst_entity_table=dst_table
)
print("LLMMicroActionSet initialized.")

print("Initializing dataset...")
perf_pred_dataset = PerformancePredictionDataset(
    dataset_name=args.dataset,
    task_name=args.task,
    tag=args.tag,
    cache_dir=args.cache_dir,
    result_dir=args.result_dir,
    seed=args.seed, # Use run-specific seed
    device=str(device),
)
print("Dataset initialized.")
  


###### INITIALIZATION ########

# 1) action result
action_result = []
valid_action_result = []
best_valid_action_result = []
history_actions = ""
error_msg = "" 
last_action_num = 0

# 2) edge set 
all_graphs = llm_micro_action_set.search_space.generate_all_graphs()
full_edges = llm_micro_action_set.search_space.full_edges
r2e_edges, f2p_edges = get_available_edges(full_edges)
current_graph_idx = llm_micro_action_set.search_space.get_full_graph_idx(all_graphs)
initial_edge_set = tuple([int(i) for i in perf_pred_dataset.get(current_graph_idx).graph_bin_str])
current_edge_set = initial_edge_set
graph_idx = current_graph_idx
best_edge_set = current_edge_set

# 3) score 
initial_score = perf_pred_dataset.get(current_graph_idx).y.item()
current_score = initial_score
best_score = initial_score
past_score = initial_score
score_result = []

# 4) budget
initial_budget = initial_budget
budget = initial_budget
max_initial_trials = max(int(initial_budget * 0.1), 1)
args.max_history_actions = max(int(initial_budget * 0.1), 15)

print('===============================================')
print(f"initial_score: {initial_score}")
print(f"initial_budget: {initial_budget}")
print(f"max_initial_trials: {max_initial_trials}")
print(f"max_history_actions: {args.max_history_actions}")
print('===============================================')

### INITIAL ATTEMPT 
initial_trials = 0
while budget > 0 :
    action_result, valid_action_result, best_valid_action_result, last_action_num = [], [], [], 0 # FIX TO INITIAL ACTION RESULT
    initial_trials += 1
    budget -= 1
    edge_info = get_edge_info(full_edges, initial_edge_set, llm_micro_action_set)
    aug_prompt = augmentation_prompt(dataset_name = args.dataset, 
                                    task_name = args.task, 
                                    edge_info=edge_info,
                                    error_msg=error_msg)
    if error_msg != "":
        print(f"===== Error Feedback ======\n{aug_prompt.split('Warning:')[-1].split('Now, you need to')[0].strip()}")
    message = client.messages.create(
                model=args.model,
                max_tokens=2048,
                temperature=args.temperature ,
                system="Imagine you are an expert graph data scientist",  
                messages=[{"role": "user", 
                           "content": [ {"type": "text",
                                         "text": aug_prompt}]}           
                ])
    response = message.content[0].text
    response_text = response.split('<selection>')[-1].split('</selection>')[0]
    print(f"====== Response text ====== \n{response_text}")

    try:
        parsed_all_actions = response_text.replace('null', 'None') if 'null' in response_text else response_text
        parsed_all_actions = ast.literal_eval(parsed_all_actions)
    except:
        # Case1: Redo the Initial Attempt if the response gets Parsing ERROR  
        print(f"ERROR parsing response: {response_text}")
        continue 
    
    # Case2: Redo the Initial Attempt if the selected action is None
    if check_none_action(parsed_all_actions):
        print(f"retrying due to None action")
        continue


    # Conduct multiple actions
    valid_actions, invalid_actions, new_edge_set, graph_idx, error_msg = conduct_multiple_actions(
        actions=parsed_all_actions,
        llm_micro_action_set=llm_micro_action_set,
        current_edge_set=initial_edge_set # FIX TO INITIAL EDGE SET
    )

    new_score = perf_pred_dataset.get(graph_idx).y.item() if graph_idx != -1 else current_score # FIX TO INITIAL SCORE

    if (new_score > initial_score and higher_is_better) or (new_score < initial_score and not higher_is_better):
        update_best = True
    else:
        update_best = False

    # Update the score
    past_score, current_score, best_score, score_result = update_score(current_score,  new_score, best_score,score_result,update_best)
    # Update the edge set
    current_edge_set, best_edge_set = update_edge_set( current_edge_set, new_edge_set, best_edge_set, update_best)
    # Update the action
    action_result, valid_action_result, best_valid_action_result, last_action_num = update_action(parsed_all_actions,  valid_actions, action_result, valid_action_result, best_valid_action_result, last_action_num, update_best)
    
    print(f"===============================================")
    print(f"Current budget: {budget}/{initial_budget}")
    print(f"update_best: {update_best}")
    print(f"Best score: {best_score:.4f}")
    print(f"Current score: {current_score:.4f}")
    print(f"Score result: {[round(r, 4) for r in score_result]}")
    print(f"Last action num: {last_action_num}")
    print(f"# of Actions: {len(action_result)}")
    print(f"# of Valid actions: {len(valid_action_result)}")
    print(f"# of Best Valid actions: {len(best_valid_action_result)}")
    print(f"===============================================")
    if update_best:
        break
    
    if initial_trials >= max_initial_trials:
        print(f"No more Initial trials. Exiting with remaining budget {budget}/{initial_budget}")
        break


while budget > 0:
    budget -= 1
    edge_info = get_edge_info(full_edges, current_edge_set, llm_micro_action_set)
    history_actions = get_history_actions(valid_action_result, args.max_history_actions)
    
    aug_prompt = augmentation_prompt(dataset_name = args.dataset, 
                                     task_name = args.task, 
                                     edge_info=edge_info,
                                     history_actions=history_actions, 
                                     error_msg=error_msg, 
                                     past_score=past_score, 
                                     current_score=current_score, 
                                     initial_score=initial_score,
                                     higher_is_better=higher_is_better,
                                     budget=budget,
                                     initial_attempt=False,
                                     last_action_num=last_action_num) 

    if error_msg != "":
        print(f"====== Warning Prompt ======\n{aug_prompt.split('Warning:')[-1].split('<selection>')[0].strip()}")
    print(f"====== Prompt ======\n{aug_prompt.split('</input>')[-1].strip()}")
    message = client.messages.create(
                model=args.model,
                max_tokens=1000,
                temperature=args.temperature,
                system="Imagine you are an expert graph data scientist",  
                messages=[{"role": "user",
                           "content": [ {"type": "text",
                                         "text": aug_prompt}]}           
                ])
    response = message.content[0].text
    response_text = response.split('<selection>')[-1].split('</selection>')[0]
    print(f"====== Response text ======\n{response_text}")
    
    try:
        parsed_all_actions = response_text.replace('null', 'None') if 'null' in response_text else response_text
        parsed_all_actions = ast.literal_eval(parsed_all_actions)
    except:
        print(f"ERROR parsing response: {response_text}")
        parsed_all_actions = []
        continue 
    
    if check_none_action(parsed_all_actions):
        print(f"No more action. Exiting with remaining budget {budget}/{initial_budget}")
        break

    parsed_all_actions = [parsed_all_actions] if type(parsed_all_actions) == dict else parsed_all_actions

    # update valid actions
    valid_actions, invalid_actions, new_edge_set, graph_idx, error_msg = conduct_multiple_actions(
        actions=parsed_all_actions,
        llm_micro_action_set=llm_micro_action_set,
        current_edge_set=current_edge_set
    )

    new_score = perf_pred_dataset.get(graph_idx).y.item() if graph_idx != -1 else current_score

    if (new_score > best_score and higher_is_better) or (new_score < best_score and not higher_is_better):
        update_best = True
    else:
        update_best = False

    if graph_idx != -1:
        # Update the edge set
        current_edge_set, best_edge_set = update_edge_set( current_edge_set, new_edge_set, best_edge_set, update_best)

    # Update the score
    past_score, current_score, best_score, score_result = update_score(current_score, new_score, best_score, score_result, update_best)

    # update the action
    action_result, valid_action_result, best_valid_action_result, last_action_num = update_action(parsed_all_actions,  valid_actions, action_result, valid_action_result, best_valid_action_result, last_action_num, update_best)
    
    print(f"===============================================")
    print(f"Current budget: {budget}/{initial_budget}")
    print(f"Best score: {best_score:.4f}")
    print(f"Current score: {current_score:.4f}")
    print(f"Score result: {[round(r, 4) for r in score_result]}")
    print(f"Last action num: {last_action_num}")
    print(f"# of Actions: {len(action_result)}")
    print(f"# of Valid actions: {len(valid_action_result)}")
    print(f"# of Best Valid actions: {len(best_valid_action_result)}")
    print(f"===============================================")

    
    if budget == 0 :
        print(f"No more budget. Exiting with remaining budget {budget}/{initial_budget}")
        break


return_result = {
    "best_score": best_score,
    "best_edge_set": best_edge_set,
    "best_valid_action_result": best_valid_action_result,
    "initial_score": initial_score,
    "initial_edge_set": current_edge_set,
    "budget_percentage": args.budget_percentage,
    "initial_budget": initial_budget,
    "remaining_budget": budget,
    "history_actions": history_actions,
    "action_result": action_result,
    "score_result": score_result,
}

with open(json_file, "w") as f:
    json.dump(return_result, f, indent=2)
print(f"Results saved to {json_file}")

print('===============================================')
print(f"Dataset: {args.dataset} Task: {args.task}")
print(f"Initial score: {initial_score:.4f}")
print(f"Final score: {best_score:.4f}")
print(f"Score result: {score_result}")
print('===============================================')

