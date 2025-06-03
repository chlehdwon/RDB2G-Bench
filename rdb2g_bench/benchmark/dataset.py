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
from sklearn.model_selection import train_test_split

from relbench.base import Dataset as RelBenchDataset, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig


from common.text_embedder import GloveTextEmbedding
from common.search_space.search_space import TotalSearchSpace
from common.search_space.gnn_search_space import GNNNodeSearchSpace, IDGNNLinkSearchSpace

class PerformancePredictionDataset(Dataset):
    def __init__(self,
                 dataset_name: str = "rel-f1",
                 task_name: str = "driver-top3",
                 tag: str | None = None,
                 cache_dir: str = os.path.expanduser("~/.cache/relbench_examples"),
                 result_dir: str = os.path.expanduser("./results"),
                 seed: int = 42,
                 device: str = 'cpu',
                 train_ratio: float = 0.1,
                 valid_ratio: float = 0.5,
                 bidirect: bool = False):
        """
        Initializes the dataset, handling data loading and preprocessing.

        Args:
            dataset_name (str): Name of the RelBench dataset.
            task_name (str): Name of the RelBench task.
            tag (str | None): Identifier for the results sub-directory. If None, tries to find suitable results.
            cache_dir (str): Directory for caching materialized graphs and stypes.
            result_dir (str): Root directory where results are stored.
            seed (int): Random seed for reproducibility (data splitting, etc.).
            device (str): Device ('cpu' or 'cuda:X') for potential GPU operations like text embedding.
            train_ratio (float): Combined proportion of the dataset for the training and validation sets (e.g., 0.2 means 10% train, 10% validation, 80% test).
            valid_ratio (float): Proportion of validation set within the train_ratio portion.
            bidirect (bool): Whether to create bidirectional edges for regular edges.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.tag = tag
        self.cache_dir = cache_dir
        self.result_dir = result_dir
        self.seed = seed
        self.device = torch.device(device)
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.bidirect = bidirect
        if not (0 < train_ratio < 1.0):
             raise ValueError("train_ratio must be between 0 and 1 (exclusive) to represent the combined train+validation proportion.")

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
        self.target_col = "test_tune_metric"
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
            "test_tune_metric",
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
            "test_tune_metric": ["mean", "std"],
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

        node_types_set = set()
        for src, _, dst in self.full_edges:
             node_types_set.add(src)
             node_types_set.add(dst)

        self.node_types = sorted(list(node_types_set))
        self.node_type_map = {name: i for i, name in enumerate(self.node_types)}
        self.num_nodes = len(self.node_types)
        print(f"Total unique node types (tables): {self.num_nodes}")
        print(f"Node types: {self.node_types}")

        if self.task.task_type == TaskType.LINK_PREDICTION:
            self.src_node_idx = self.node_type_map[self.task.src_entity_table]
            self.dst_node_idx = self.node_type_map[self.task.dst_entity_table]
        else:
            self.src_node_idx = self.node_type_map[self.task.entity_table]
            self.dst_node_idx = -1

        self.static_node_features = torch.arange(self.num_nodes, dtype=torch.long)

        indices = self.df_result_group.index.tolist()

        test_size = 1.0 - self.train_ratio
        val_prop_in_trainval = self.valid_ratio

        print(f"Splitting data: Train={self.train_ratio / 2:.3f} / Val={self.train_ratio / 2:.3f} / Test={test_size:.3f} (based on train_ratio={self.train_ratio:.3f})")

        train_val_indices, self.test_indices = train_test_split(
            indices, test_size=test_size, random_state=self.seed
        )

        self.train_indices, self.val_indices = train_test_split(
            train_val_indices, test_size=val_prop_in_trainval, random_state=self.seed
        )

        self.splits = {
            'train': self.train_indices,
            'valid': self.val_indices,
            'test': self.test_indices
        }


    def len(self) -> int:
        return len(self.df_result_group)

    def get(self, original_idx: int) -> Data:
        row = self.df_result_group.iloc[original_idx]

        graph_bin_str = row['graph']
        # Remove 'graph_' prefix if it exists
        if isinstance(graph_bin_str, str) and graph_bin_str.startswith("graph_"):
            graph_bin_str = graph_bin_str[6:]
            
        target = torch.tensor([row[self.target_col]], dtype=torch.float)
        train_time = torch.tensor([row['train_time']], dtype=torch.float)

        edge_list = []
        edge_feature_idx = []
        for i, edge_type_tuple in enumerate(self.full_edges):
            if graph_bin_str[i] == '1':
                src_type, rel, dst_type = edge_type_tuple
                if src_type in self.node_type_map and dst_type in self.node_type_map:
                    src_idx_mapped = self.node_type_map[src_type]
                    dst_idx_mapped = self.node_type_map[dst_type]
                if rel[:4] == "r2e_":
                    edge_type = rel[4:]
                    edge_list.append([src_idx_mapped, dst_idx_mapped])
                    edge_feature_idx.append(self.node_type_map[edge_type])
                    edge_list.append([dst_idx_mapped, src_idx_mapped])
                    edge_feature_idx.append(self.node_type_map[edge_type])
                else:
                    edge_list.append([src_idx_mapped, dst_idx_mapped])
                    edge_feature_idx.append(-1)
                    if self.bidirect:
                        edge_list.append([dst_idx_mapped, src_idx_mapped])
                        edge_feature_idx.append(-1)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_feature_idx, dtype=torch.long)

        data = Data(
            x=self.static_node_features.clone(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target, 
            train_time=train_time,
            num_nodes=self.num_nodes,
            graph_id=torch.tensor([row['idx']], dtype=torch.long),
            graph_bin_str=graph_bin_str
        )
        return data

    def __getitem__(self, idx):
        return self.get(idx)

    def get_split_indices(self, split: str) -> list[int]:
        return self.splits[split]
