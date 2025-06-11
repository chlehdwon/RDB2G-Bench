import argparse
import json
import os
# os.environ['XDG_CACHE_HOME'] = '/data/cache'
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.seed import seed_everything
from typing import Dict, Optional

from relbench.base import Dataset as RelBenchDataset, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from rdb2g_bench.common.text_embedder import GloveTextEmbedding
from rdb2g_bench.common.search_space.search_space import TotalSearchSpace
from rdb2g_bench.common.search_space.gnn_search_space import GNNNodeSearchSpace, IDGNNLinkSearchSpace

class PerformancePredictionDataset(Dataset):
    def __init__(self,
                 dataset_name: str = "rel-f1",
                 task_name: str = "driver-top3",
                 tag: Optional[str] = None,
                 cache_dir: str = "~/.cache/relbench_examples",
                 result_dir: str = "./results",
                 seed: int = 42,
                 device: str = 'cpu'):
        """
        Initialize Performance Prediction Dataset for RDB2G-Bench.

        Args:
            dataset_name (str): Name of the RelBench dataset (e.g., "rel-f1", "rel-avito").
                Defaults to "rel-f1".
            task_name (str): Name of the RelBench task (e.g., "driver-top3", "user-ad-visit").
                Task availability depends on the dataset. Defaults to "driver-top3".
            tag (Optional[str]): Identifier for the results sub-directory. If None, 
                attempts to find suitable results automatically. Used to organize
                different experimental runs.
            cache_dir (str): Directory for caching materialized graphs and schema types.
                Helps speed up repeated dataset loading. Defaults to "~/.cache/relbench_examples".
            result_dir (str): Root directory where performance results are stored.
                Expected structure: {result_dir}/tables/{dataset_name}/{task_name}/{tag}/
                Defaults to "./results".
            seed (int): Random seed for reproducibility of graph materialization and
                data processing. Defaults to 42.
            device (str): Device for GPU operations during text embedding ('cpu' or 'cuda:X').
                Defaults to 'cpu'.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.tag = tag
        self.cache_dir = cache_dir
        self.result_dir = result_dir
        self.seed = seed
        self.device = torch.device(device)

        seed_everything(self.seed)

        print(f"Loading dataset: {self.dataset_name}, task: {self.task_name}")
        self.rb_dataset: RelBenchDataset = get_dataset(self.dataset_name, download=True)
        self.task = get_task(self.dataset_name, self.task_name)

        stypes_cache_path = Path(f"{self.cache_dir}/{self.dataset_name}/stypes.json")
        try:
            with open(stypes_cache_path, "r") as f:
                col_to_stype_dict = json.load(f)
            for table, col_to_stype in col_to_stype_dict.items():
                for col, stype_str in col_to_stype.items():
                    col_to_stype[col] = stype(stype_str)
            print(f"Loaded stypes from cache: {stypes_cache_path}")
        except FileNotFoundError:
            print("Stypes cache not found, proposing stypes...")
            col_to_stype_dict = get_stype_proposal(self.rb_dataset.get_db())
            Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stypes_cache_path, "w") as f:
                serializable_stypes = {t: {c: s.value for c, s in cs.items()} for t, cs in col_to_stype_dict.items()}
                json.dump(serializable_stypes, f, indent=2)
            print(f"Saved proposed stypes to: {stypes_cache_path}")

        graph_cache_dir = f"{self.cache_dir}/{self.dataset_name}/materialized"
        print(f"Materializing graph data (cache dir: {graph_cache_dir})...")
        text_embedder=GloveTextEmbedding(device=self.device)
        data, col_stats_dict = make_pkey_fkey_graph(
            self.rb_dataset.get_db(),
            col_to_stype_dict=col_to_stype_dict,
            text_embedder_cfg=TextEmbedderConfig(
                text_embedder=text_embedder, batch_size=256
            ),
            cache_dir=graph_cache_dir,
        )
        self.hetero_data = data
        print("Graph data materialized.")

        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            self.tune_metric = "roc_auc"
            gnn_search_space_cls = GNNNodeSearchSpace
        elif self.task.task_type == TaskType.REGRESSION:
            self.tune_metric = "mae"
            gnn_search_space_cls = GNNNodeSearchSpace
        elif self.task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            self.tune_metric = "multilabel_auprc_macro"
            gnn_search_space_cls = GNNNodeSearchSpace
        elif self.task.task_type == TaskType.LINK_PREDICTION:
            self.tune_metric = "link_prediction_map"
            gnn_search_space_cls = IDGNNLinkSearchSpace
        else:
            raise ValueError(f"Unsupported task type: {self.task.task_type}")
        self.target_col = "test_metric"
        print(f"Task type: {self.task.task_type}, Tune metric: {self.tune_metric}, Target (y): {self.target_col}")

        print("Calculating full edge set using TotalSearchSpace...")
        search_space = TotalSearchSpace(
            dataset=self.dataset_name, 
            task=self.task_name,      
            hetero_data=self.hetero_data,
            GNNSearchSpace=gnn_search_space_cls,
            num_layers=2, 
            src_entity_table=self.task.entity_table if self.task.task_type != TaskType.LINK_PREDICTION else self.task.src_entity_table,
            dst_entity_table=None if self.task.task_type != TaskType.LINK_PREDICTION else self.task.dst_entity_table,
        )
        self.full_edges: list[tuple[str, str, str]] = search_space.get_full_edges()
        self.search_space = search_space
        print(f"Calculated {len(self.full_edges)} possible edge types.")

        file_dir = os.path.join(self.result_dir, "tables", self.dataset_name, self.task_name, self.tag)
        print(f"Loading results from: {file_dir}")
        if not os.path.isdir(file_dir):
            raise FileNotFoundError(f"Result directory not found: {file_dir}")
        file_names = os.listdir(file_dir)
        if not file_names:
             raise ValueError(f"No result files found in {file_dir}")

        df_result = pd.DataFrame()
        required_cols = [
            "idx", "graph",
            "test_metric",
            "params", "train_time",
            "valid_time", "test_time"
        ]
        for fn in file_names:
            if fn.endswith(".csv"):
                try:
                    df_single = pd.read_csv(os.path.join(file_dir, fn))
                    missing_cols = [col for col in required_cols if col not in df_single.columns]
                    if missing_cols:
                        print(f"Warning: File {fn} is missing columns: {missing_cols}. Skipping this file.")
                        continue
                    df_result = pd.concat([df_result, df_single[required_cols]], axis=0, ignore_index=True)
                except Exception as e:
                    print(f"Warning: Failed to read or process file {fn}: {e}")

        if df_result.empty:
            raise ValueError(f"No valid data loaded from result files in {file_dir}. Check file contents and required columns: {required_cols}")

        print(f"Loaded {len(df_result)} rows from {len([f for f in file_names if f.endswith('.csv')])} CSV files.")

        agg_dict = {
            "test_metric": ["mean", "std"],
            "params": "mean",
            "train_time": "mean",
            "valid_time": "mean",
            "test_time": "mean"
        }

        self.df_result_group = df_result.groupby(["idx", "graph"], dropna=False).agg(agg_dict).reset_index()

        new_columns = ["idx", "graph"]
        for col, funcs in agg_dict.items():
            if isinstance(funcs, list):
                for func in funcs:
                    new_col_name = col if func == 'mean' else f"{col}_{func}"
                    new_columns.append(new_col_name)
            else:
                new_columns.append(col)
        self.df_result_group.columns = new_columns
        print(f"Grouped results into {len(self.df_result_group)} unique graphs.")

        self.df_result_group['graph'] = self.df_result_group['graph'].astype(str)
        
        # Remove 'graph_' prefix from graph strings if it exists
        self.df_result_group['graph'] = self.df_result_group['graph'].apply(
            lambda x: x[6:] if x.startswith("graph_") else x
        )
        
        max_len = self.df_result_group['graph'].str.len().max()
        expected_len = len(self.full_edges)
        if max_len < expected_len:
            print(f"Warning: Max graph string length ({max_len}) is less than expected ({expected_len}). Padding to {expected_len}.")
            max_len = expected_len
        elif max_len > expected_len:
             print(f"Warning: Max graph string length ({max_len}) is greater than expected ({expected_len}). Check data integrity.")

        self.df_result_group['graph'] = self.df_result_group['graph'].str.zfill(max_len)
        print(f"Padded 'graph' column strings to length {max_len}.")

        graph_strings = self.df_result_group['graph'].tolist()
        graphs_list_of_tuples = [tuple(map(int, list(g_str))) for g_str in graph_strings]
        self.full_graph_id = self.search_space.get_full_graph_idx(graphs_list_of_tuples)
        print(f"Full graph index: {self.full_graph_id}")

    def len(self) -> int:
        """
        Returns the number of graph configurations in the dataset.

        Returns:
            int: Total number of unique graph configurations after aggregation.
        """
        return len(self.df_result_group)

    def get(self, original_idx: int) -> Data:
        """
        Retrieves a single data sample at the specified index.

        Args:
            original_idx (int): Index of the sample to retrieve from the aggregated results.

        Returns:
            Data: PyTorch Geometric Data object with graph configuration and performance data.
            
            - y (torch.Tensor): Target performance value as a 1D tensor
            - graph_bin_str (str): Binary string representation of the graph configuration
        """
        row = self.df_result_group.iloc[original_idx]

        graph_bin_str = row['graph']
        # Remove 'graph_' prefix if it exists
        if isinstance(graph_bin_str, str) and graph_bin_str.startswith("graph_"):
            graph_bin_str = graph_bin_str[6:]
            
        target = torch.tensor([row[self.target_col]], dtype=torch.float)

        data = Data(y=target, graph_bin_str=graph_bin_str)
        return data

    def __getitem__(self, idx: int) -> Data:
        """
        Enables indexing access to dataset samples.

        This method allows the dataset to be used with standard Python indexing
        syntax (e.g., dataset[0]) and makes it compatible with PyTorch DataLoader.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Data: PyTorch Geometric Data object containing the sample data.
        """
        return self.get(idx)
