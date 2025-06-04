# Reference: https://github.com/snap-stanford/relbench/blob/main/examples/idgnn_link.py

import copy
import gc
import json
import os
import time
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

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

from ..common.text_embedder import GloveTextEmbedding
from ..common.search_space.gnn_search_space import IDGNNLinkSearchSpace
from ..common.search_space.search_space import TotalSearchSpace
from .models.model import Model
from .utils import integrate_edge_tf


def run_idgnn_link_worker(
    dataset_name: str = "rel-avito",
    task_name: str = "user-ad-visit",
    lr: float = 0.001,
    epochs: int = 20,
    weight_decay: float = 0,
    eval_epochs_interval: int = 1,
    batch_size: int = 512,
    channels: int = 128,
    aggr: str = "sum",
    gnn: str = "GraphSAGE",
    num_layers: int = 2,
    num_neighbors: int = 128,
    temporal_strategy: str = "last",
    max_steps_per_epoch: int = 2000,
    num_workers: int = 0,
    seed: int = 42,
    patience: int = 20,
    cache_dir: str = os.path.expanduser("~/.cache/relbench_examples"),
    result_dir: str = os.path.expanduser("./results"),
    tag: str = "",
    debug: bool = False,
    debug_idx: int = -1,
    idx: Optional[int] = None,
    workers: Optional[int] = None,
    target_indices: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    save_csv: bool = True,
) -> Dict:
    """
    Run IDGNN link prediction worker function.
    
    Args:
        dataset_name: Name of the dataset (e.g., "rel-avito")
        task_name: Name of the task (e.g., "user-ad-visit")
        lr: Learning rate
        epochs: Number of training epochs
        weight_decay: Weight decay for optimizer
        eval_epochs_interval: Evaluation interval
        batch_size: Batch size for training
        channels: Number of hidden channels
        aggr: Aggregation method
        gnn: GNN model type ("GraphSAGE", "GIN", "GPS")
        num_layers: Number of GNN layers
        num_neighbors: Number of neighbors for sampling
        temporal_strategy: Temporal sampling strategy
        max_steps_per_epoch: Maximum steps per epoch
        num_workers: Number of workers for data loading
        seed: Random seed
        patience: Early stopping patience
        cache_dir: Cache directory path
        result_dir: Results directory path
        tag: Tag for result organization
        debug: Enable debug mode
        debug_idx: Debug graph index
        idx: Worker index for parallel processing
        workers: Total number of workers
        target_indices: Specific graph indices to run
        device: Device to use (if None, auto-detect)
        save_csv: Whether to save results to CSV file
        
    Returns:
        Dictionary containing processing status:
        - 'processed_graphs': List of graph indices that were processed
        - 'total_processed': Number of graphs processed
        - 'csv_file': Path to CSV file if save_csv=True, None otherwise
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    
    seed_everything(seed)
    
    if gnn == "GPS":
        num_neighbors = 32
    
    if not debug and target_indices is None and (idx is None or workers is None):
        raise ValueError("idx and workers must be specified when not in debug mode and target_indices is not provided")
    
    dataset: Dataset = get_dataset(dataset_name, download=True)
    task: RecommendationTask = get_task(dataset_name, task_name, download=True)
    tune_metric = "link_prediction_map"
    assert task.task_type == TaskType.LINK_PREDICTION
    
    stypes_cache_path = Path(f"{cache_dir}/{dataset_name}/stypes.json")
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
        cache_dir=f"{cache_dir}/{dataset_name}/materialized",
    )
    
    def train(model, loader_dict, optimizer, train_sparse_tensor, edge_tf_dict) -> float:
        model.train()
        loss_accum = count_accum = 0
        steps = 0
        
        for batch in loader_dict["train"]:
            batch = integrate_edge_tf(batch, edge_tf_dict)
            batch = batch.to(device)
            out = model.forward_dst_readout(
                batch,
                task.src_entity_table, task.dst_entity_table
            ).flatten()
            
            batch_size = batch[task.src_entity_table].batch_size
            
            input_id = batch[task.src_entity_table].input_id
            src_batch, dst_index = train_sparse_tensor[input_id]
            
            target = torch.isin(
                batch[task.dst_entity_table].batch
                + batch_size * batch[task.dst_entity_table].n_id,
                src_batch + batch_size * dst_index,
            ).float()
            
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(out, target)
            loss.backward()
            optimizer.step()
            
            loss_accum += float(loss) * out.numel()
            count_accum += out.numel()
            
            steps += 1
            if steps > max_steps_per_epoch:
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
    def test(model, loader: NeighborLoader, edge_tf_dict) -> np.ndarray:
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
        dataset=dataset_name,
        task=task_name,
        hetero_data=data,
        GNNSearchSpace=IDGNNLinkSearchSpace,
        src_entity_table=task.src_entity_table,
        dst_entity_table=task.dst_entity_table,
        num_layers=num_layers
    )
    
    all_graphs = search_space.generate_all_graphs()
    search_space_size = len(all_graphs)
    full_graph_idx = search_space.get_full_graph_idx(all_graphs)
    
    if target_indices is not None:
        graphs_to_run = [(idx, all_graphs[idx]) for idx in target_indices if 0 <= idx < search_space_size]
        if len(graphs_to_run) != len(target_indices):
            print(f"Warning: Some indices in target_indices were invalid or out of range (0-{search_space_size-1}).")
        print(f"Running only for specified valid indices: {[idx for idx, _ in graphs_to_run]}")
    elif debug:
        if debug_idx == -1:
            debug_idx = full_graph_idx
        if 0 <= debug_idx < search_space_size:
            indices_to_run = [debug_idx]
            print(f"Running in debug mode for index: {debug_idx}")
        else:
            print(f"Error: debug_idx {debug_idx} is out of range (0-{search_space_size-1}).")
            indices_to_run = []
        graphs_to_run = [(idx, all_graphs[idx]) for idx in indices_to_run]
    elif idx is not None and workers is not None:
        indices_to_run = list(range(idx, search_space_size, workers))
        graphs_to_run = [(idx, all_graphs[idx]) for idx in indices_to_run]
    else:
        print("Warning: No specific indices, worker info, or debug index provided. Check arguments.")
        graphs_to_run = []
    
    csv_file_path = None
    columns = None
    if save_csv:
        csv_dir = f"{result_dir}/tables/{dataset_name}/{task_name}/{tag}"
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_file_path = f"{csv_dir}/{seed}.csv"
        
        columns = ["idx", "graph", "train_tune_metric", "val_tune_metric", "test_tune_metric", "params", "train_time", "valid_time", "test_time", "dataset", "task", "seed"]
        
        if not os.path.exists(csv_file_path):
            print(f"Creating CSV file: {csv_file_path}")
            df_init = pd.DataFrame(columns=columns)
            df_init.to_csv(csv_file_path, header=True, index=False)
    
    processed_graphs = []
    
    for graph_idx, graph_config in graphs_to_run:
        iter_start = time.time()
        graph = torch.Tensor(graph_config).int()
        search_data = search_space.get_data(graph)
        
        print()
        print("=====================================")
        print(f"Running graph index: {graph_idx} / {search_space_size - 1} total")
        
        if debug:
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
            num_neighbors_list = [math.ceil(math.sqrt(int(num_neighbors / 2**i))) for i in range(num_layers)]
        else:
            num_neighbors_list = [int(num_neighbors / 2**i) for i in range(num_layers)]
        
        loader_dict: Dict[str, NeighborLoader] = {}
        train_sparse_tensor = None
        
        for split in ["train", "val", "test"]:
            table = task.get_table(split)
            table_input = get_link_train_table_input(table=table, task=task)
            
            loader_dict[split] = NeighborLoader(
                search_data,
                num_neighbors=num_neighbors_list,
                time_attr="time" if is_time_exist else None,
                input_nodes=table_input.nodes,
                input_time=table_input.time if is_time_exist else None,
                transform=table_input.transform,
                batch_size=batch_size,
                temporal_strategy=temporal_strategy,
                shuffle=split == "train",
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )
            
            if split == "train":
                train_sparse_tensor = SparseTensor.from_dgl(table_input.src_batch_to_dst_index)
                train_sparse_tensor = train_sparse_tensor.to(device)
        
        model = Model(
            data=search_data,
            col_stats_dict=col_stats_dict,
            num_layers=num_layers,
            channels=channels,
            out_channels=1,
            aggr=aggr,
            gnn=gnn,
            norm="batch_norm",
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total learnable parameters: {params}")
        
        state_dict = None
        best_val_metric = -math.inf
        patience_counter = 0
        train_start = time.time()
        
        for epoch in range(1, epochs + 1):
            train_loss = train(model, loader_dict, optimizer, train_sparse_tensor, edge_tf_dict)
            
            if epoch % eval_epochs_interval == 0 or epoch == epochs:
                val_pred = test(model, loader_dict["val"], edge_tf_dict)
                val_metrics = task.evaluate(val_pred, task.get_table("val"))
                
                if val_metrics[tune_metric] >= best_val_metric:
                    best_val_metric = val_metrics[tune_metric]
                    state_dict = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Reach patience at {epoch}th epoch")
                        break
        
        train_time = (time.time() - train_start) / epochs
        
        model.load_state_dict(state_dict)
        
        train_pred = test(model, loader_dict["train"], edge_tf_dict)
        train_metrics = task.evaluate(train_pred, task.get_table("train"))
        print(f"Best Train metrics: {train_metrics}")
        
        valid_start = time.time()
        val_pred = test(model, loader_dict["val"], edge_tf_dict)
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        valid_time = time.time() - valid_start
        print(f"Best Val metrics: {val_metrics}")
        
        test_start = time.time()
        test_pred = test(model, loader_dict["test"], edge_tf_dict)
        test_metrics = task.evaluate(test_pred)
        test_time = time.time() - test_start
        print(f"Best test metrics: {test_metrics}")
        
        graph_str = ''.join(map(str, graph.int().tolist()))
        graph_str = f"graph_{graph_str}"
        
        if save_csv and not debug:
            one_data = [graph_idx, graph_str, train_metrics[tune_metric], val_metrics[tune_metric], test_metrics[tune_metric], params, train_time, valid_time, test_time, dataset_name, task_name, seed]
            df_one = pd.DataFrame([one_data], columns=columns)
            df_one.to_csv(csv_file_path, mode='a', header=False, index=False)
        
        processed_graphs.append(graph_idx)
        
        del search_data, loader_dict, model, optimizer, train_sparse_tensor
        if 'train_pred' in locals():
            del train_pred, val_pred, test_pred
        if 'state_dict' in locals():
            del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        
        if debug:
            print(f"Total time: {time.time() - iter_start}")
    
    if save_csv and not debug:
        print(f"Results saved to: {csv_file_path}")
    
    return {
        'processed_graphs': processed_graphs,
        'total_processed': len(processed_graphs),
        'csv_file': csv_file_path if save_csv else None
    } 