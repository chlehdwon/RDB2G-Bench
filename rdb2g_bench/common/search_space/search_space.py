from typing import Union, List

import numpy as np
import pandas as pd
import torch    
from torch_geometric.data import HeteroData
from .gnn_search_space import GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace
from .row2ne_search_space import Row2NESearchSpace

class TotalSearchSpace():
    def __init__(self,
        dataset: str,
        task: str,
        hetero_data: HeteroData,
        GNNSearchSpace: Union[GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace],
        num_layers: int,
        src_entity_table: str,
        dst_entity_table: str = None
    ):
        self.dataset = dataset
        self.task = task
        self.data = hetero_data
        self.row2graph = Row2NESearchSpace(dataset, task, hetero_data)
        self.num_layers = num_layers
        self.src_entity_table = src_entity_table
        self.dst_entity_table = dst_entity_table
        self.full_edges = self.row2graph.find_full_edges()
        self.gnn_search_space = GNNSearchSpace(
            dataset=self.dataset,
            task=self.task,
            num_layers=self.num_layers,
            node_types=self.row2graph.node_types,
            edge_types=self.full_edges,
            src_entity_table=self.src_entity_table,
            dst_entity_table=self.dst_entity_table
        )

    def get_possible_edges(self):
        return self.row2graph.find_possible_edges()

    def get_full_edges(self):
        return self.full_edges

    def generate_all_graphs(self): 
        return self.gnn_search_space.generate_all_graphs()

    def get_full_graph_idx(self, graphs: List[tuple]) -> int:
        graphs = np.array(graphs)
        r2n_idx = np.where(np.sum(graphs[:, :len(self.get_possible_edges())], axis=1) == 0)[0]
        return -1 if len(r2n_idx) == 0 else r2n_idx[-1]

    def get_data(self, edges: Union[torch.Tensor, tuple]) -> HeteroData:
        if type(edges) != torch.Tensor:
            edges = torch.tensor(edges)
        converted_data = self.row2graph.convert_row_to_edge(edges)
        return self.gnn_search_space.get_data(edges, converted_data)