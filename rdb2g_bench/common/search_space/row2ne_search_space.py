from typing import Any, Dict, List, Tuple, Union

import copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

class Row2NESearchSpace():
    def __init__(self, dataset: str, task: str, hetero_data: HeteroData):
        self.dataset = dataset
        self.task = task
        self.data = hetero_data
        self.node_types = sorted(list(hetero_data.node_types))
        self.edge_types = sorted(list(hetero_data.edge_types))
        self.possible_edge_types = self.find_possible_edges()
        self.full_edges = self.find_full_edges()
        self._cached_r2e_key = None
        self._cached_converted_data = None
    
    def find_possible_edges(self):
        possible_edges = []
        for node_type in self.node_types:
            outgoing_edges = [(src, rel, dst) for src, rel, dst in self.edge_types 
                             if src == node_type and rel.startswith('f2p')]
            incoming_edges = [(src, rel, dst) for src, rel, dst in self.edge_types 
                             if dst == node_type and rel.startswith('f2p')]
            if len(outgoing_edges) == 2 and len(incoming_edges) == 0:
                possible_edges.append(node_type)
        
        return sorted(possible_edges)

    def find_full_edges(self):
        full_edges = []
        
        for node_type in self.possible_edge_types:
            connected_nodes = []
            for src, rel, dst in self.edge_types:
                if src == node_type and rel.startswith('f2p'):
                    connected_nodes.append(dst)
            
            if len(connected_nodes) == 1:
                src_node, dst_node = connected_nodes[0], connected_nodes[0]
            elif len(connected_nodes) == 2:
                src_node, dst_node = connected_nodes[0], connected_nodes[1]
            else:
                raise ValueError(f"Invalid number of connected nodes: {len(connected_nodes)}")

            src_edge_type, dst_edge_type = None, None
            for edge_type in self.edge_types:
                if edge_type[0] == node_type and edge_type[2] == src_node and edge_type != dst_edge_type and src_edge_type is None:
                    src_edge_type = edge_type
                if edge_type[0] == node_type and edge_type[2] == dst_node and edge_type != src_edge_type and dst_edge_type is None:
                    dst_edge_type = edge_type

            if src_edge_type and dst_edge_type:
                full_edges.append((src_node, f"r2e_{node_type}", dst_node))

        return full_edges + [edge for edge in sorted(self.data.edge_types) if edge[1].startswith("f2p")]

    def convert_row_to_edge(self, edges: torch.Tensor):
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges)

        # Determine the cache key based on the row-to-edge part of the edges tensor
        r2e_part = edges[:len(self.possible_edge_types)]
        current_r2e_key = tuple(r2e_part.tolist())
        if current_r2e_key == self._cached_r2e_key:
            print("Using cached converted data")
            return copy.deepcopy(self._cached_converted_data)

        converted_data = HeteroData()
        # Copy node data and ensure tensors are contiguous
        nodes_to_convert = set()
        for i, is_selected in enumerate(r2e_part):
            if is_selected == 1:
                _, rel, _ = self.full_edges[i]
                nodes_to_convert.add(rel[4:])

        for node_type in self.node_types:
            for attr_name, attr_value in self.data[node_type].items():
                if node_type in nodes_to_convert and attr_name == 'time':
                    continue
                if isinstance(attr_value, torch.Tensor):
                    converted_data[node_type][attr_name] = attr_value.clone().detach().contiguous()
                else:
                    converted_data[node_type][attr_name] = attr_value

        # Copy existing edges and ensure tensors are contiguous
        for edge_type in self.edge_types:
            src_node, edge_name, dst_node = edge_type
            if src_node not in nodes_to_convert and dst_node not in nodes_to_convert:
                converted_data[edge_type].edge_index = self.data[edge_type].edge_index.clone().detach().contiguous()

        # Convert nodes to edges
        for node_type in nodes_to_convert:
            connected_nodes = []
            for src, rel, dst in self.edge_types:
                if src == node_type and rel.startswith('f2p'):
                    connected_nodes.append(dst)

            if len(connected_nodes) == 1:
                src_node, dst_node = connected_nodes[0], connected_nodes[0]
            elif len(connected_nodes) == 2:
                src_node, dst_node = connected_nodes[0], connected_nodes[1]
            else:
                raise ValueError(f"Invalid number of connected nodes: {len(connected_nodes)}")

            # Find relevant edge types
            src_edge_type, dst_edge_type = None, None
            for edge_type in self.edge_types:
                if edge_type[0] == node_type and edge_type[2] == src_node and edge_type != dst_edge_type and src_edge_type is None:
                    src_edge_type = edge_type
                if edge_type[0] == node_type and edge_type[2] == dst_node and edge_type != src_edge_type and dst_edge_type is None:
                    dst_edge_type = edge_type

            if src_edge_type and dst_edge_type:
                # Create new edge (from src_node to dst_node)
                new_edge_type = (src_node, f"r2e_{node_type}", dst_node)
                reverse_edge_type = (dst_node, f"rev_r2e_{node_type}", src_node)
                
                src_edges = self.data[src_edge_type].edge_index.clone().detach().contiguous()
                dst_edges = self.data[dst_edge_type].edge_index.clone().detach().contiguous()
                
                # Create mappings using numpy operations
                src_node_ids = src_edges[0].numpy()
                src_table_ids = src_edges[1].numpy()
                src_mapping = dict(zip(src_node_ids, src_table_ids))
                
                dst_node_ids = dst_edges[0].numpy()
                dst_table_ids = dst_edges[1].numpy()
                dst_mapping = dict(zip(dst_node_ids, dst_table_ids))
                
                # Get common node ids that exist in both mappings
                node_ids = np.arange(len(self.data[node_type].tf))
                mask = np.isin(node_ids, list(src_mapping.keys())) & np.isin(node_ids, list(dst_mapping.keys()))
                valid_node_ids = node_ids[mask]
                
                # Construct edge pairs
                if len(valid_node_ids) > 0:
                    # Use vectorized operations to get src and dst ids
                    max_node_id = max(max(src_mapping.keys()), max(dst_mapping.keys())) + 1
                    src_lookup = np.full(max_node_id, -1)
                    dst_lookup = np.full(max_node_id, -1)
                    
                    src_keys = np.array(list(src_mapping.keys()), dtype=np.int64)
                    src_values = np.array(list(src_mapping.values()), dtype=np.int64)
                    src_lookup[src_keys] = src_values
                    
                    dst_keys = np.array(list(dst_mapping.keys()), dtype=np.int64)
                    dst_values = np.array(list(dst_mapping.values()), dtype=np.int64)
                    dst_lookup[dst_keys] = dst_values
                    
                    # Apply vectorized lookups
                    src_ids = src_lookup[valid_node_ids]
                    dst_ids = dst_lookup[valid_node_ids]
                    new_edges = np.stack([src_ids, dst_ids], axis=1)
                    
                    if len(new_edges) > 0:
                        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
                        converted_data[new_edge_type].edge_index = new_edge_index
                        converted_data[new_edge_type].mapped_node_ids = torch.tensor(valid_node_ids, dtype=torch.long).clone().detach().contiguous()
                        
                        reverse_edge_index = torch.stack([new_edge_index[1], new_edge_index[0]], dim=0).contiguous()
                        converted_data[reverse_edge_type].edge_index = reverse_edge_index
                        converted_data[reverse_edge_type].mapped_node_ids = torch.tensor(valid_node_ids, dtype=torch.long).clone().detach().contiguous()

        self._cached_converted_data = converted_data.to('cpu')
        self._cached_r2e_key = current_r2e_key

        return converted_data