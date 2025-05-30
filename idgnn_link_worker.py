# Reference: https://github.com/snap-stanford/relbench/blob/main/examples/idgnn_link.py

import argparse
import copy
import gc
import json
import os
import time
import math

# os.environ['XDG_CACHE_HOME'] = '/data/cache'
import warnings
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from tqdm import tqdm

from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from common.text_embedder import GloveTextEmbedding
from common.search_space.gnn_search_space import IDGNNLinkSearchSpace
from common.search_space.search_space import TotalSearchSpace

from dataset.models.model import Model
from dataset.utils import integrate_edge_tf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-avito")
parser.add_argument("--task", type=str, default="user-ad-visit")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--gnn", type=str, default="GraphSAGE")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument(
    "--result_dir",
    type=str,
    default=os.path.expanduser("./results"),
)
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
parser.add_argument("--debug_idx", type=int, default=-1)

# worker args
parser.add_argument("--idx", type=int)
parser.add_argument("--workers", type=int)
parser.add_argument("--target_indices", type=str, default=None, help="Comma-separated list of specific graph indices to run.")
args = parser.parse_args()

if args.gnn == "GPS": args.num_neighbors = 32

if not args.debug and args.target_indices is None and (args.idx is None or args.workers is None):
    raise ValueError("idx and workers must be specified when not in debug mode and target_indices is not provided")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

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

def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in loader_dict["train"]:
        batch = integrate_edge_tf(batch, edge_tf_dict)
        batch = batch.to(device)
        out = model.forward_dst_readout(
            batch,
            task.src_entity_table, task.dst_entity_table
        ).flatten()

        batch_size = batch[task.src_entity_table].batch_size

        # Get ground-truth
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Get target label
        target = torch.isin(
            batch[task.dst_entity_table].batch
            + batch_size * batch[task.dst_entity_table].n_id,
            src_batch + batch_size * dst_index,
        ).float()

        # Optimization
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        optimizer.step()

        loss_accum += float(loss) * out.numel()
        count_accum += out.numel()

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    if count_accum == 0:
        warnings.warn(
            f"Did not sample a single '{task.dst_entity_table}' "
            f"node in any mini-batch. Try to increase the number "
            f"of layers/hops and re-try. If you run into memory "
            f"issues with deeper nets, decrease the batch size."
        )

    return loss_accum / count_accum if count_accum > 0 else float("nan")


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list: list[Tensor] = []
    for batch in loader:
        batch = integrate_edge_tf(batch, edge_tf_dict)
        batch = batch.to(device)
        out = (
            model.forward_dst_readout(
                batch,
                task.src_entity_table, task.dst_entity_table
            )
            .detach()
            .flatten()
        )
        batch_size = batch[task.src_entity_table].batch_size
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
        scores[
            batch[task.dst_entity_table].batch, batch[task.dst_entity_table].n_id
        ] = torch.sigmoid(out)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred

search_space = TotalSearchSpace(
    dataset=args.dataset,
    task=args.task,
    hetero_data=data,
    GNNSearchSpace=IDGNNLinkSearchSpace,
    src_entity_table=task.src_entity_table,
    dst_entity_table=task.dst_entity_table,
    num_layers=args.num_layers,
)
all_graphs = search_space.generate_all_graphs()
search_space_size = len(all_graphs)
full_graph_idx = search_space.get_full_graph_idx(all_graphs)
if args.debug:
    if args.debug_idx == -1:
        args.debug_idx = full_graph_idx
    args.idx = args.debug_idx
    args.workers = search_space_size

columns = ["idx", "graph"] + ["train_tune_metric"] + ["val_tune_metric"] + ["test_tune_metric"] + ["params", "train_time", "valid_time", "test_time"] + ["dataset", "task", "seed"]
csv_dir = f"{args.result_dir}/tables/{args.dataset}/{args.task}/{args.tag}"
os.makedirs(csv_dir, exist_ok=True)
csv_path = f"{csv_dir}/{args.seed}.csv"
if not os.path.exists(csv_path):
    print(f"Creating CSV file: {csv_path}")
    df_init = pd.DataFrame(columns=columns) 
    df_init.to_csv(csv_path, header=True, index=False)

if args.target_indices:
    try:
        target_indices_list = [int(x.strip()) for x in args.target_indices.split(',')]
        graphs_to_run = [(idx, all_graphs[idx]) for idx in target_indices_list if 0 <= idx < search_space_size]
        if len(graphs_to_run) != len(target_indices_list):
             print(f"Warning: Some indices in --target_indices were invalid or out of range (0-{search_space_size-1}).")
        print(f"Running only for specified valid indices: {[idx for idx, _ in graphs_to_run]}")
    except ValueError:
        print("Error: Invalid format for --target_indices. Please provide comma-separated integers.")
        graphs_to_run = []
else:
    if args.debug:
        if args.debug_idx == -1:
            args.debug_idx = full_graph_idx
        if 0 <= args.debug_idx < search_space_size:
            indices_to_run = [args.debug_idx]
            print(f"Running in debug mode for index: {args.debug_idx}")
        else:
            print(f"Error: --debug_idx {args.debug_idx} is out of range (0-{search_space_size-1}).")
            indices_to_run = []
    elif args.idx is not None and args.workers is not None:
        indices_to_run = list(range(args.idx, search_space_size, args.workers))
    else:
        print("Warning: No specific indices, worker info, or debug index provided. Check arguments.")
        indices_to_run = []

    graphs_to_run = [(idx, all_graphs[idx]) for idx in indices_to_run]

for idx, graph_config in graphs_to_run:
    iter_start = time.time()
    graph = torch.Tensor(graph_config).int()
    search_data = search_space.get_data(graph)
    print()
    print("===================================")
    print(f"Running graph index: {idx} / {search_space_size - 1} total")

    if args.debug:
        print(graph.tolist())
        for edge_type, is_used in zip(search_space.get_full_edges(), graph):
            if is_used:
                print(edge_type)

    edge_tf_dict = {}
    for edge_type in search_data.edge_types:
        src, rel, dst = edge_type
        if rel.startswith('r2e_'):
            table_name = rel[4:]
            edge_tf_dict[table_name] = search_data[table_name].tf

    is_time_exist = False
    for store in search_data.node_stores:
        if "time" in store:
            is_time_exist = True
    
    if len(edge_tf_dict.keys()) > 0:
        num_neighbors = [math.ceil(math.sqrt(int(args.num_neighbors / 2**i))) for i in range(args.num_layers)]
    else:
        num_neighbors = [int(args.num_neighbors / 2**i) for i in range(args.num_layers)]

    loader_dict: Dict[str, NeighborLoader] = {}
    dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_link_train_table_input(table, task)
        dst_nodes_dict[split] = table_input.dst_nodes
        loader_dict[split] = NeighborLoader(
            search_data,
            num_neighbors=num_neighbors,
            time_attr="time" if is_time_exist else None,
            input_nodes=table_input.src_nodes,
            input_time=table_input.src_time if is_time_exist else None,
            subgraph_type="bidirectional",
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=split == "train",
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            disjoint=True,
        )

    model = Model(
        data=search_data,
        col_stats_dict=col_stats_dict,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=1,
        aggr=args.aggr,
        norm="layer_norm",
        id_awareness=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {params}")

    state_dict = None
    best_val_metric = 0
    patience = 0
    train_start = time.time()   
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        if epoch % args.eval_epochs_interval == 0:
            val_pred = test(loader_dict["val"])
            val_metrics = task.evaluate(val_pred, task.get_table("val"))
            if val_metrics[tune_metric] > best_val_metric:
                best_val_metric = val_metrics[tune_metric]
                state_dict = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"Reach patience at {epoch}th epoch")
                    break
    train_time = (time.time() - train_start) / args.epochs

    model.load_state_dict(state_dict)

    train_pred = test(loader_dict["train"])
    train_metrics = task.evaluate(train_pred, task.get_table("train"))
    print(f"Best Train metrics: {train_metrics}")

    valid_start = time.time()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    valid_time = time.time() - valid_start
    print(f"Best Val metrics: {val_metrics}")

    test_start = time.time()
    test_pred = test(loader_dict["test"])
    test_metrics = task.evaluate(test_pred)
    test_time = time.time() - test_start
    print(f"Best test metrics: {test_metrics}")

    # clear memory
    del search_data
    del loader_dict['train']
    del loader_dict['val']
    del loader_dict['test']
    del loader_dict
    del model
    del optimizer

    if 'train_pred' in locals():
        del train_pred
    if 'val_pred' in locals():
        del val_pred
    if 'test_pred' in locals():
        del test_pred
    if 'state_dict' in locals():
        del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    if not args.debug:
        graph_str = ''.join(map(str, graph.int().tolist()))
        graph_str = f"graph_{graph_str}"
        one_data = [idx, graph_str] + [train_metrics[tune_metric], val_metrics[tune_metric], test_metrics[tune_metric]] + [params, train_time, valid_time, test_time] + [args.dataset, args.task, args.seed]
        df_one = pd.DataFrame([one_data], columns=columns)
        df_one.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        print(f"Total time: {time.time() - iter_start}")