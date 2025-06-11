import os
import json
import ast
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Union
from pathlib import Path
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
import anthropic

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from ...common.text_embedder import GloveTextEmbedding
from ...common.search_space.gnn_search_space import GNNNodeSearchSpace, IDGNNLinkSearchSpace
from ..dataset import PerformancePredictionDataset
from .llm_micro_action import LLMMicroActionSet
from .llm_utils import (
    get_edge_info, get_budget, get_available_edges, conduct_multiple_actions,
    check_none_action, update_edge_set, update_score, update_action,
    get_history_actions
)
from .prompts.prompt import augmentation_prompt


def run_llm_baseline(
    dataset: str = "rel-f1",
    task: str = "driver-top3",
    budget_percentage: float = 0.05,
    seed: int = 42,
    model: str = "claude-3-5-sonnet-latest",
    temperature: float = 0.8,
    tag: str = "final",
    cache_dir: str = "~/.cache/relbench_examples",
    result_dir: str = "./results",
    **kwargs
) -> Dict:
    """
    Run Large Language Model baseline for neural architecture search.
    
    This function provides an LLM-based baseline that leverages the reasoning
    capabilities of large language models to guide graph neural network architecture
    search. The approach uses natural language prompts to describe the search space
    and current state, then interprets LLM responses to execute micro actions.
    
    Args:
        dataset (str): Name of the RelBench dataset to use. Defaults to "rel-f1".
        task (str): Name of the RelBench task to evaluate. Defaults to "driver-top3".
        budget_percentage (float): Budget as fraction of total search space (0.0-1.0).
            Defaults to 0.05.
        seed (int): Random seed for reproducibility. Defaults to 42.
        model (str): Anthropic model name to use. Defaults to "claude-3-5-sonnet-latest".
        temperature (float): Sampling temperature for LLM responses (0.0-1.0).
            Defaults to 0.8.
        tag (str): Experiment tag for result organization. Defaults to "final".
        cache_dir (str): Directory for caching datasets and models.
            Defaults to "~/.cache/relbench_examples".
        result_dir (str): Directory for saving results.
            Defaults to "./results".
        **kwargs: Additional configuration parameters passed to underlying components.
        
    Returns:
        Dict[str, Any]: Dictionary containing LLM baseline results.
        
        - best_score (float): Best performance achieved during search
        - initial_score (float): Initial performance of full graph
        - best_edge_set (tuple): Best graph configuration found
        - action_history (List): Sequence of actions taken
        - budget_used (int): Number of evaluations performed
        - score_trajectory (List[float]): Performance over time
        - error_count (int): Number of invalid actions attempted
        - success_rate (float): Fraction of valid actions
        - total_time (float): Total execution time
        - convergence_step (int): Step when best score was found
        - final_improvement (float): Best score minus initial score
        
    Example:
        >>> # Set API key first
        >>> import os
        >>> os.environ["ANTHROPIC_API_KEY"] = "your_api_key_here"
        >>> 
        >>> # Run LLM baseline
        >>> results = run_llm_baseline(
        ...     dataset="rel-f1",
        ...     task="driver-top3",
        ...     budget_percentage=0.05,
        ...     model="claude-3-5-sonnet-latest",
        ...     temperature=0.8,
        ...     seed=42
        ... )
        >>> print(f"Best performance: {results['best_score']:.4f}")
        >>> print(f"Initial performance: {results['initial_score']:.4f}")
        >>> print(f"Improvement: {results['best_score'] - results['initial_score']:.4f}")
        
    Note:
        This function requires an active internet connection and valid Anthropic API credentials.
        The ANTHROPIC_API_KEY environment variable must be set before calling this function.
        Results are automatically saved to JSON files in the specified output directory.
    """
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable must be set. "
            "Please set it using: export ANTHROPIC_API_KEY='your_api_key'"
        )
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(
        dataset=dataset,
        task=task,
        budget_percentage=budget_percentage,
        seed=seed,
        tag=tag,
        cache_dir=cache_dir,
        result_dir=result_dir,
        model=model,
        temperature=temperature
    )
    
    client = anthropic.Anthropic()
    
    initial_budget = get_budget(args.dataset, args.task, args.budget_percentage)
    print(f"Dataset: {args.dataset} Task: {args.task} Budget: {initial_budget}")
    
    json_file = f"./outputs/{args.tag}/{args.dataset}_{args.task}_{args.budget_percentage}.json"
    if not os.path.exists(os.path.dirname(json_file)):
        print(f"Creating directory: {os.path.dirname(json_file)}")
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    seed_everything(args.seed)
    
    dataset_obj: Dataset = get_dataset(args.dataset, download=True)
    task_obj: EntityTask = get_task(args.dataset, args.task, download=True)
    task_type = task_obj.task_type
    
    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset_obj.get_db())
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)
    
    data, col_stats_dict = make_pkey_fkey_graph(
        dataset_obj.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
    )
    
    if task_type == TaskType.BINARY_CLASSIFICATION:
        higher_is_better = True
        gnn_space_class = GNNNodeSearchSpace
        src_table = task_obj.entity_table
        dst_table = None
    elif task_type == TaskType.REGRESSION:
        higher_is_better = False
        gnn_space_class = GNNNodeSearchSpace
        src_table = task_obj.entity_table
        dst_table = None
    elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
        higher_is_better = True
        gnn_space_class = GNNNodeSearchSpace
        src_table = task_obj.entity_table
        dst_table = None
    elif task_type == TaskType.LINK_PREDICTION:
        higher_is_better = True
        gnn_space_class = IDGNNLinkSearchSpace
        src_table = task_obj.src_entity_table
        dst_table = task_obj.dst_entity_table
        if src_table is None or dst_table is None:
            raise ValueError("Link prediction task missing source_table or destination_table attribute.")
    else:
        raise ValueError(f"Task type {task_type} is unsupported for determining GNNSpaceClass and tables.")
    
    print("Initializing MicroActionSet...")
    llm_micro_action_set = LLMMicroActionSet(
        dataset=args.dataset,
        task=args.task,
        hetero_data=data,
        GNNSpaceClass=gnn_space_class,
        num_layers=2,
        src_entity_table=src_table,
        dst_entity_table=dst_table
    )
    print("MicroActionSet initialized.")
    
    print("Initializing dataset...")
    perf_pred_dataset = PerformancePredictionDataset(
        dataset_name=args.dataset,
        task_name=args.task,
        tag=args.tag,
        cache_dir=args.cache_dir,
        result_dir=args.result_dir,
        seed=args.seed,
        device=str(device),
    )
    print("Dataset initialized.")
    
    action_result = []
    valid_action_result = []
    best_valid_action_result = []
    history_actions = ""
    error_msg = ""
    last_action_num = 0
    
    all_graphs = llm_micro_action_set.search_space.generate_all_graphs()
    full_edges = llm_micro_action_set.search_space.full_edges
    r2e_edges, f2p_edges = get_available_edges(full_edges)
    current_graph_idx = llm_micro_action_set.search_space.get_full_graph_idx(all_graphs)
    initial_edge_set = tuple([int(i) for i in perf_pred_dataset.get(current_graph_idx).graph_bin_str])
    current_edge_set = initial_edge_set
    best_edge_set = current_edge_set
    
    initial_score = perf_pred_dataset.get(current_graph_idx).y.item()
    current_score = initial_score
    best_score = initial_score
    past_score = initial_score
    score_result = []
    
    budget = initial_budget
    max_initial_trials = max(int(initial_budget * 0.1), 1)
    max_history_actions = max(int(initial_budget * 0.1), 15)
    
    print('===============================================')
    print(f"initial_score: {initial_score}")
    print(f"initial_budget: {initial_budget}")
    print(f"max_initial_trials: {max_initial_trials}")
    print(f"max_history_actions: {max_history_actions}")
    print('===============================================')
    
    initial_trials = 0
    while budget > 0:
        action_result, valid_action_result, best_valid_action_result, last_action_num = [], [], [], 0
        initial_trials += 1
        budget -= 1
        
        edge_info = get_edge_info(full_edges, initial_edge_set, llm_micro_action_set)
        aug_prompt = augmentation_prompt(
            dataset_name=args.dataset,
            task_name=args.task,
            edge_info=edge_info,
            error_msg=error_msg
        )
        
        if error_msg != "":
            print(f"===== Error Feedback ======\n{aug_prompt.split('Warning:')[-1].split('Now, you need to')[0].strip()}")
        
        message = client.messages.create(
            model=args.model,
            max_tokens=2048,
            temperature=args.temperature,
            system="Imagine you are an expert graph data scientist",
            messages=[{"role": "user",
                      "content": [{"type": "text", "text": aug_prompt}]}]
        )
        
        response = message.content[0].text
        response_text = response.split('<selection>')[-1].split('</selection>')[0]
        print(f"====== Response text ====== \n{response_text}")
        
        try:
            parsed_all_actions = response_text.replace('null', 'None') if 'null' in response_text else response_text
            parsed_all_actions = ast.literal_eval(parsed_all_actions)
        except:
            print(f"ERROR parsing response: {response_text}")
            continue
        
        if check_none_action(parsed_all_actions):
            print(f"retrying due to None action")
            continue
        
        valid_actions, invalid_actions, new_edge_set, graph_idx, error_msg = conduct_multiple_actions(
            actions=parsed_all_actions,
            llm_micro_action_set=llm_micro_action_set,
            current_edge_set=initial_edge_set
        )
        
        new_score = perf_pred_dataset.get(graph_idx).y.item() if graph_idx != -1 else current_score
        
        if (new_score > initial_score and higher_is_better) or (new_score < initial_score and not higher_is_better):
            update_best = True
        else:
            update_best = False
        
        past_score, current_score, best_score, score_result = update_score(
            current_score, new_score, best_score, score_result, update_best
        )
        current_edge_set, best_edge_set = update_edge_set(
            current_edge_set, new_edge_set, best_edge_set, update_best
        )
        action_result, valid_action_result, best_valid_action_result, last_action_num = update_action(
            parsed_all_actions, valid_actions, action_result, valid_action_result,
            best_valid_action_result, last_action_num, update_best
        )
        
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
        history_actions = get_history_actions(valid_action_result, max_history_actions)
        
        aug_prompt = augmentation_prompt(
            dataset_name=args.dataset,
            task_name=args.task,
            edge_info=edge_info,
            history_actions=history_actions,
            error_msg=error_msg,
            past_score=past_score,
            current_score=current_score,
            initial_score=initial_score,
            higher_is_better=higher_is_better,
            budget=budget,
            initial_attempt=False,
            last_action_num=last_action_num
        )
        
        if error_msg != "":
            print(f"====== Warning Prompt ======\n{aug_prompt.split('Warning:')[-1].split('<selection>')[0].strip()}")
        print(f"====== Prompt ======\n{aug_prompt.split('</input>')[-1].strip()}")
        
        message = client.messages.create(
            model=args.model,
            max_tokens=1000,
            temperature=args.temperature,
            system="Imagine you are an expert graph data scientist",
            messages=[{"role": "user",
                      "content": [{"type": "text", "text": aug_prompt}]}]
        )
        
        response = message.content[0].text
        response_text = response.split('<selection>')[-1].split('</selection>')[0]
        print(f"====== Response text ======\n{response_text}")
        
        try:
            parsed_all_actions = response_text.replace('null', 'None') if 'null' in response_text else response_text
            parsed_all_actions = ast.literal_eval(parsed_all_actions)
        except:
            print(f"ERROR parsing response: {response_text}")
            continue
        
        if check_none_action(parsed_all_actions):
            print(f"No more action. Exiting with remaining budget {budget}/{initial_budget}")
            break
        
        parsed_all_actions = [parsed_all_actions] if type(parsed_all_actions) == dict else parsed_all_actions
        
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
            current_edge_set, best_edge_set = update_edge_set(
                current_edge_set, new_edge_set, best_edge_set, update_best
            )
        
        past_score, current_score, best_score, score_result = update_score(
            current_score, new_score, best_score, score_result, update_best
        )
        action_result, valid_action_result, best_valid_action_result, last_action_num = update_action(
            parsed_all_actions, valid_actions, action_result, valid_action_result,
            best_valid_action_result, last_action_num, update_best
        )
        
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
        
        if budget == 0:
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
    
    return return_result 