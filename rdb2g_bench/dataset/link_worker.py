# Reference: https://github.com/snap-stanford/relbench/blob/main/examples/idgnn_link.py

import copy
import gc
import json
import os
import time
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

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
    cache_dir: str = "~/.cache/relbench_examples",
    result_dir: str = "./results",
    tag: str = "",
    debug: bool = False,
    debug_idx: int = -1,
    idx: Optional[int] = 0,
    workers: Optional[int] = 1,
    target_indices: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    save_csv: bool = True,
) -> Dict[str, Union[List[int], int, Optional[str]]]:
    """
    Run IDGNN link prediction worker function.
    
    This function executes link prediction experiments using ID-aware Graph Neural Networks
    on RDB2G-Bench datasets. It is specifically designed for recommendation tasks where
    the goal is to predict links between source and destination entities (e.g., user-item
    interactions). The function supports multiple graph configurations and provides
    comprehensive evaluation using ranking-based metrics.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "rel-avito", "rel-amazon").
            Defaults to "rel-avito".
        task_name (str): Name of the task (e.g., "user-ad-visit", "user-item-purchase").
            Defaults to "user-ad-visit".
        lr (float): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int): Maximum number of training epochs. Defaults to 20.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
            Defaults to 0.
        eval_epochs_interval (int): Interval for evaluation during training.
            Defaults to 1.
        batch_size (int): Batch size for training and evaluation. Defaults to 512.
        channels (int): Number of hidden channels in the GNN layers. Defaults to 128.
        aggr (str): Aggregation method for GNN layers ("sum", "mean", "max").
            Defaults to "sum".
        gnn (str): GNN architecture to use ("GraphSAGE", "GIN", "GPS").
            Defaults to "GraphSAGE".
        num_layers (int): Number of GNN layers. Defaults to 2.
        num_neighbors (int): Number of neighbors to sample per layer. Defaults to 128.
        temporal_strategy (str): Temporal sampling strategy ("last", "uniform").
            Defaults to "last".
        max_steps_per_epoch (int): Maximum number of training steps per epoch.
            Defaults to 2000.
        num_workers (int): Number of workers for data loading. Defaults to 0.
        seed (int): Random seed for reproducibility. Defaults to 42.
        patience (int): Early stopping patience (epochs without improvement).
            Defaults to 20.
        cache_dir (str): Directory for caching processed data.
            Defaults to "~/.cache/relbench_examples".
        result_dir (str): Directory for saving results. Defaults to "./results".
        tag (str): Tag for organizing results in subdirectories. Defaults to "".
        debug (bool): Enable debug mode (runs only one graph configuration).
            Defaults to False.
        debug_idx (int): Specific graph index to run in debug mode. If -1, uses
            full graph. Defaults to -1.
        idx (Optional[int]): Worker index for parallel processing. Defaults to 0.
        workers (Optional[int]): Total number of parallel workers. Defaults to 1.
        target_indices (Optional[List[int]]): Specific graph indices to process.
            If None, processes based on worker assignment. Defaults to None.
        device (Optional[torch.device]): Device to use for training. If None,
            auto-detects CUDA availability. Defaults to None.
        save_csv (bool): Whether to save results to CSV file. Defaults to True.
        
    Returns:
        Dict[str, Union[List[int], int, Optional[str]]]: Dictionary containing processing status information.
        
        - processed_graphs (List[int]): List of graph indices that were processed
        - total_processed (int): Number of graphs processed
        - csv_file (Optional[str]): Path to CSV file if save_csv=True, None otherwise

    Example:
        >>> # Run single experiment in debug mode
        >>> results = run_idgnn_link_worker(
        ...     dataset_name="rel-avito",
        ...     task_name="user-ad-visit",
        ...     debug=True,
        ...     epochs=10
        ... )
        >>> print(f"Processed {results['total_processed']} graphs")
        
        >>> # Run parallel processing
        >>> results = run_idgnn_link_worker(
        ...     dataset_name="rel-avito",
        ...     task_name="user-ad-visit",
        ...     idx=0,
        ...     workers=1,
        ...     epochs=20
        ... )
    """
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
        """
        Train the model for one epoch using binary cross-entropy loss.
        
        Args:
            model: IDGNN model to train
            loader_dict: Dictionary containing data loaders
            optimizer: PyTorch optimizer
            train_sparse_tensor: Sparse tensor for positive link supervision
            edge_tf_dict: Edge transformation dictionary
            
        Returns:
            float: Average training loss for the epoch.
            
        Note:
            The training uses binary cross-entropy loss where positive examples
            come from the sparse tensor and negative examples are sampled
            implicitly through the batch structure.
        """
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
        """
        Evaluate the model and return top-k predictions for link prediction.
        
        Args:
            model: IDGNN model to evaluate
            loader: Data loader for evaluation
            edge_tf_dict: Edge transformation dictionary
            
        Returns:
            np.ndarray: Top-k destination node predictions for each source node.
            
            Shape: (num_src_nodes, k) where k is task.eval_k
                
        Note:
            The function computes similarity scores between source and destination
            nodes, then returns the top-k destination nodes for each source node
            based on these scores.
        """
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
    csv_writer = None
    csv_file = None
    if save_csv:
        csv_dir = f"{result_dir}/tables/{dataset_name}/{task_name}/{tag}"
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_file_path = f"{csv_dir}/{seed}.csv"
        
        columns = ["idx", "graph", "train_metric", "val_metric", "test_metric", "params", "train_time", "valid_time", "test_time", "dataset", "task", "seed"]
        
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
        dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
        for split in ["train", "val", "test"]:
            table = task.get_table(split)
            table_input = get_link_train_table_input(table, task)
            dst_nodes_dict[split] = table_input.dst_nodes
            loader_dict[split] = NeighborLoader(
                search_data,
                num_neighbors=num_neighbors_list,
                time_attr="time" if is_time_exist else None,
                input_nodes=table_input.src_nodes,
                input_time=table_input.src_time if is_time_exist else None,
                subgraph_type="bidirectional",
                batch_size=batch_size,
                temporal_strategy=temporal_strategy,
                shuffle=split == "train",
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                disjoint=True,
            )
        
        model = Model(
            data=search_data,
            col_stats_dict=col_stats_dict,
            num_layers=num_layers,
            channels=channels,
            out_channels=1,
            aggr=aggr,
            norm="layer_norm",
            id_awareness=True,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)
        
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total learnable parameters: {params}")
        
        state_dict = None
        best_val_metric = 0
        patience_counter = 0
        train_start = time.time()
        
        for epoch in range(1, epochs + 1):
            train_loss = train(model, loader_dict, optimizer, train_sparse_tensor, edge_tf_dict)
            if epoch % eval_epochs_interval == 0:
                val_pred = test(model, loader_dict["val"], edge_tf_dict)
                val_metrics = task.evaluate(val_pred, task.get_table("val"))
                if val_metrics[tune_metric] > best_val_metric:
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
        
        del search_data, loader_dict, model, optimizer
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