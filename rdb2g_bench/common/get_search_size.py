import argparse
import json
import os
# os.environ['XDG_CACHE_HOME'] = '/data/cache'
from pathlib import Path

import pandas as pd
import torch
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from relbench.base import Dataset, EntityTask, RecommendationTask
from relbench.datasets import get_dataset, get_dataset_names
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task, get_task_names

from search_space.gnn_search_space import GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace
from search_space.search_space import TotalSearchSpace
from text_embedder import GloveTextEmbedding

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

device = torch.device('cpu')
if torch.cuda.is_available():
    torch.set_num_threads(1)

columns = ["dataset", "task", "task_type", "src_entity", "dst_entity", "num_graphs", "full_idx"]
csv_path = f"./analysis/search_size_stats.csv"
df_init = pd.DataFrame(columns=columns)
df_init.to_csv(csv_path, header=True, index=False)

for dataset_name in get_dataset_names():
    for task_name in get_task_names(dataset_name):
        dataset: Dataset = get_dataset(dataset_name, download=True)
        task: EntityTask = get_task(dataset_name, task_name, download=True)

        stypes_cache_path = Path(f"{args.cache_dir}/{dataset_name}/stypes.json")
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
            cache_dir=f"{args.cache_dir}/{dataset_name}/materialized",
        )

        if isinstance(task, RecommendationTask):
            src_entity_table, dst_entity_table = task.src_entity_table, task.dst_entity_table
            search_space = TotalSearchSpace(dataset=dataset_name,
                                        task=task_name,
                                        hetero_data=data,
                                        GNNSearchSpace=GNNLinkSearchSpace,
                                        num_layers=2,
                                        src_entity_table=src_entity_table,
                                        dst_entity_table=dst_entity_table)
            all_graphs = search_space.generate_all_graphs()
            full_idx = search_space.get_full_graph_idx(all_graphs)
            
            df_one = pd.DataFrame([[dataset_name, task_name, "gnn_link", src_entity_table, dst_entity_table, len(all_graphs), full_idx]], columns=columns)
            df_one.to_csv(csv_path, mode='a', header=False, index=False)
            search_space = TotalSearchSpace(dataset=dataset_name,
                                        task=task_name,
                                        hetero_data=data,
                                        GNNSearchSpace=IDGNNLinkSearchSpace,
                                        num_layers=2,
                                        src_entity_table=src_entity_table,
                                        dst_entity_table=dst_entity_table)
            all_graphs = search_space.generate_all_graphs()
            full_idx = search_space.get_full_graph_idx(all_graphs)
            
            df_one = pd.DataFrame([[dataset_name, task_name, "idgnn_link", src_entity_table, dst_entity_table, len(all_graphs), full_idx]], columns=columns)
            df_one.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            src_entity_table, dst_entity_table = task.entity_table, None
            search_space = TotalSearchSpace(dataset=dataset_name,
                                        task=task_name,
                                        hetero_data=data,
                                        GNNSearchSpace=GNNNodeSearchSpace,
                                        num_layers=2,
                                        src_entity_table=src_entity_table)
            all_graphs = search_space.generate_all_graphs()
            full_idx = search_space.get_full_graph_idx(all_graphs)
            
            df_one = pd.DataFrame([[dataset_name, task_name, "gnn_node", src_entity_table, None, len(all_graphs), full_idx]], columns=columns)
            df_one.to_csv(csv_path, mode='a', header=False, index=False)
