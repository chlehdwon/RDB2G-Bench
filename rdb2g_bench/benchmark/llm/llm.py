import json
import os
# os.environ['XDG_CACHE_HOME'] = '/data/cache'

from pathlib import Path
from typing import Dict

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

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        train_ratio=args.train_ratio,
        bidirect=args.bidirect,
        valid_ratio=args.valid_ratio
    )
    print("Dataset initialized.")