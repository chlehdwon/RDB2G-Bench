from typing import Any, Dict, List, Tuple

import copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

class GNNSearchSpace():
    def __init__(
        self,
        dataset: str,
        task: str,
        num_layers: int,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        src_entity_table: str,
        dst_entity_table: str = None,
    ):
        self.dataset = dataset
        self.task = task
        self.src_entity_table = src_entity_table
        self.dst_entity_table = dst_entity_table
        self.num_layers = num_layers
        self.node_types = node_types
        self.edge_types = edge_types
        self.reachable_nodes = set()
        
    def filter_unreachable(self, edges):
        reachable_nodes = self.reachable_nodes.copy()
        layer_reachable = [reachable_nodes.copy()]

        for _ in range(self.num_layers):
            new_reachable = set()
            for i, is_selected in enumerate(edges):
                if is_selected == 1:
                    src, _, dst = self.edge_types[i]
                    if src in reachable_nodes:
                        new_reachable.add(dst)
                    if dst in reachable_nodes:
                        new_reachable.add(src)

            reachable_nodes.update(new_reachable)
            layer_reachable.append(new_reachable)
            
            if not new_reachable:
                break

        filtered_edges = np.copy(edges)
        for i, is_selected in enumerate(edges):
            if is_selected == 1:
                src, _, dst = self.edge_types[i]
                # Check connection between consecutive layers
                for layer in range(len(layer_reachable)-1):
                    if (src in layer_reachable[layer] and dst in layer_reachable[layer+1]) or \
                       (dst in layer_reachable[layer] and src in layer_reachable[layer+1]):
                        break
                else:
                    filtered_edges[i] = 0

        return filtered_edges
    
    def is_valid(self, edges) -> bool:
        raise NotImplementedError

    def is_possible(self, edges) -> bool:
        for i, is_selected in enumerate(edges):
            if is_selected == 1:
                _, rel, _ = self.edge_types[i]
                if rel.startswith('r2e'):
                    rel = rel[4:]
                    for j, edge in enumerate(self.edge_types):
                        if edge[0] == rel or edge[2] == rel:
                            if edges[j] == 1:
                                return False
        return True

    def generate_all_graphs(self):
        num_edge_types = len(self.edge_types)
        possible_graphs = set()
        for mask in range(2 ** num_edge_types):
            # Create binary array of selected edges (1 = included, 0 = excluded)
            selected_edges = np.zeros(num_edge_types, dtype=int)
            for i in range(num_edge_types):
                if mask & (1 << i):
                    selected_edges[i] = 1
            if self.is_possible(selected_edges):
                filtered_edges = self.filter_unreachable(selected_edges)
                key = tuple(filtered_edges)
                if key not in possible_graphs and self.is_valid(filtered_edges):
                    possible_graphs.add(key)
        
        return sorted(list(possible_graphs))

    def get_data(self, edges: np.ndarray, data: HeteroData) -> HeteroData:
        new_data = HeteroData()
        
        involved_nodes = set()
        for i, is_selected in enumerate(edges):
            if is_selected:
                src, rel, dst = self.edge_types[i]
                involved_nodes.add(src)
                involved_nodes.add(dst)
                if rel.startswith("r2e"):
                    involved_nodes.add(rel[4:])
        
        for node in involved_nodes:
            if node in data.node_types:
                for attr_name, attr_value in data[node].items():
                    new_data[node][attr_name] = attr_value

        for i, is_selected in enumerate(edges):
            if is_selected:
                edge_type = self.edge_types[i]
                if edge_type in data.edge_types:
                    for attr_name, attr_value in data[edge_type].items():
                        new_data[edge_type][attr_name] = attr_value
                    rev_edge_type = (edge_type[2], 'rev_' + edge_type[1], edge_type[0])
                    assert rev_edge_type in data.edge_types
                    for attr_name, attr_value in data[rev_edge_type].items():
                        new_data[rev_edge_type][attr_name] = attr_value

        return new_data

    def __repr__(self) -> str:
        return (f"BasicSearchSpace(dataset={self.dataset}, "
                f"task={self.task}, "
                f"src_entity_table={self.src_entity_table}, "
                f"matrix_shape={self.matrix}, "
                f"node_types={self.node_types})")


class GNNNodeSearchSpace(GNNSearchSpace):
    def __init__(
        self,
        dataset: str,
        task: str,
        num_layers: int,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        src_entity_table: str,
        dst_entity_table: str = None,
    ):
        super().__init__(dataset, task, num_layers, node_types, edge_types, src_entity_table, dst_entity_table)
        self.reachable_nodes = {self.src_entity_table}
        if dst_entity_table is not None:
            raise ValueError("dst_entity_table must be None for GNNNodeSearchSpace")


    def filter_unreachable(self, edges: torch.Tensor) -> torch.Tensor:
        return super().filter_unreachable(edges)

    def is_valid(self, edges: torch.Tensor) -> bool:
        # if target entity has no incoming/outcoming edges, the graph is invalid
        src_has_edges = False

        for i, is_selected in enumerate(edges):
            if is_selected == 1:
                src, _, dst = self.edge_types[i]
                if src == self.src_entity_table or dst == self.src_entity_table:
                    src_has_edges = True
                    break
        
        if not src_has_edges:
            return False
            
        return True

    def __repr__(self) -> str:
        return (f"GNNNodeSearchSpace(dataset={self.dataset}, "
                f"task={self.task}, "
                f"src_entity_table={self.src_entity_table}, "
                f"node_types={self.node_types})")


class GNNLinkSearchSpace(GNNSearchSpace):
    def __init__(
        self,
        dataset: str,
        task: str,
        num_layers: int,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        src_entity_table: str,
        dst_entity_table: str,
    ):
        super().__init__(dataset, task, num_layers, node_types, edge_types, src_entity_table, dst_entity_table)
        self.reachable_nodes = {self.src_entity_table, self.dst_entity_table}

    def filter_unreachable(self, edges: np.ndarray) -> np.ndarray:
        return super().filter_unreachable(edges)

    def is_valid(self, edges: np.ndarray) -> bool:
        # Check if source and destination entity tables have any edges
        src_has_edges = False
        dst_has_edges = False
        
        for i, is_selected in enumerate(edges):
            if is_selected == 1:
                src, _, dst = self.edge_types[i]
                if src == self.src_entity_table or dst == self.src_entity_table:
                    src_has_edges = True
                if src == self.dst_entity_table or dst == self.dst_entity_table:
                    dst_has_edges = True
                if src_has_edges and dst_has_edges:
                    break
        
        # If either source or destination entity has no edges, the graph is invalid
        if not src_has_edges or not dst_has_edges:
            return False
            
        return True

    def __repr__(self) -> str:
        return (f"GNNLinkSearchSpace(dataset={self.dataset}, "
                f"task={self.task}, "
                f"src_entity_table={self.src_entity_table}, "
                f"dst_entity_table={self.dst_entity_table}, "
                f"node_types={self.node_types})")


class IDGNNLinkSearchSpace(GNNSearchSpace):
    def __init__(
        self,
        dataset: str,
        task: str,
        num_layers: int,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        src_entity_table: str,
        dst_entity_table: str,
    ):
        super().__init__(dataset, task, num_layers, node_types, edge_types, src_entity_table, dst_entity_table)
        self.reachable_nodes = {self.src_entity_table}

    def filter_unreachable(self, edges: np.ndarray) -> np.ndarray:
        return super().filter_unreachable(edges)

    def is_valid(self, edges: np.ndarray) -> bool:
        # Check if source and destination entity tables have any edges
        src_has_edges = False
        dst_has_edges = False
        
        for i, is_selected in enumerate(edges):
            if is_selected == 1:
                src, _, dst = self.edge_types[i]
                if src == self.src_entity_table or dst == self.src_entity_table:
                    src_has_edges = True
                if src == self.dst_entity_table or dst == self.dst_entity_table:
                    dst_has_edges = True
                if src_has_edges and dst_has_edges:
                    break
        
        # If source or destination entity has no edges, the graph is invalid
        if not src_has_edges or not dst_has_edges:
            return False

        reachable_from_src = {self.src_entity_table}

        for _ in range(len(self.node_types)):
            if self.dst_entity_table in reachable_from_src:
                break
                
            new_reachable = set()
            for i, is_selected in enumerate(edges):
                if is_selected == 1:
                    src, _, dst = self.edge_types[i]
                    if src in reachable_from_src:
                        new_reachable.add(dst)
                    if dst in reachable_from_src:
                        new_reachable.add(src)
            
            reachable_from_src.update(new_reachable)
            
            if not new_reachable:
                break
        
        # If dst_entity_table is not reachable from src_entity_table, the graph is invalid
        if self.dst_entity_table not in reachable_from_src:
            return False
            
        return True

    def __repr__(self) -> str:
        return (f"IDGNNLinkSearchSpace(dataset={self.dataset}, "
                f"task={self.task}, "
                f"src_entity_table={self.src_entity_table}, "
                f"dst_entity_table={self.dst_entity_table}, "
                f"node_types={self.node_types})")
