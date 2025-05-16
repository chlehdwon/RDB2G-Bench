from typing import Tuple, Set, Optional, Type, Union, List, Dict
import torch
from torch_geometric.data import HeteroData

from common.search_space.search_space import TotalSearchSpace
from common.search_space.gnn_search_space import GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace

class MicroActionSet:
    def __init__(self,
                 dataset: str,
                 task: str,
                 hetero_data: HeteroData,
                 GNNSpaceClass: Type[Union[GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace]],
                 num_layers: int,
                 src_entity_table: str,
                 dst_entity_table: Optional[str] = None):
        self.search_space = TotalSearchSpace(
            dataset=dataset,
            task=task,
            hetero_data=hetero_data,
            GNNSearchSpace=GNNSpaceClass,
            num_layers=num_layers,
            src_entity_table=src_entity_table,
            dst_entity_table=dst_entity_table
        )
        self.full_edges: List[Tuple[str, str, str]] = self.search_space.get_full_edges()
        self.fk_pk_indices: Set[int] = {
            i for i, edge_type in enumerate(self.full_edges) if edge_type[1].startswith("f2p")
        }
        self.r2e_indices: Set[int] = {
            i for i, edge_type in enumerate(self.full_edges) if edge_type[1].startswith("r2e")
        }

        _valid_edge_sets_set: Set[Tuple[int, ...]] = set(self.search_space.generate_all_graphs())
        self.valid_edge_sets_list: List[Tuple[int, ...]] = sorted(list(_valid_edge_sets_set))
        self.valid_edge_sets: Set[Tuple[int, ...]] = set(self.valid_edge_sets_list)

        # Precompute mappings for conversion actions
        self.r2e_to_f2p_map: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        self.f2p_pair_to_r2e_map: Dict[Tuple[int, int], int] = {}

        for r2e_idx in self.r2e_indices:
            src, rel, dst = self.full_edges[r2e_idx]
            node_table = rel.split('_', 1)[1]

            f2p_idx1 = None
            f2p_idx2 = None

            # Find the f2p edge B -> src
            for fk_pk_idx in self.fk_pk_indices:
                fk_src, fk_rel, fk_dst = self.full_edges[fk_pk_idx]
                if fk_src == node_table and fk_dst == src:
                    f2p_idx1 = fk_pk_idx
                    break

            # Find the f2p edge B -> dst
            for fk_pk_idx in self.fk_pk_indices:
                fk_src, fk_rel, fk_dst = self.full_edges[fk_pk_idx]
                if fk_src == node_table and fk_dst == dst:
                    f2p_idx2 = fk_pk_idx
                    break

            # Store the mapping from r2e index to the pair of f2p indices
            self.r2e_to_f2p_map[r2e_idx] = (f2p_idx1, f2p_idx2)

            # If both f2p edges are found, store the reverse mapping
            if f2p_idx1 is not None and f2p_idx2 is not None:
                # Ensure consistent key order by sorting indices
                key = tuple(sorted((f2p_idx1, f2p_idx2)))
                self.f2p_pair_to_r2e_map[key] = r2e_idx

    def add_fk_pk_edge(self,
                       current_edge_set: Tuple[int, ...]
                      ) -> List[Tuple[Tuple[int, ...], int]]:
        possible_next_sets_with_indices = []
        for edge_index_to_add in self.fk_pk_indices:
            if current_edge_set[edge_index_to_add] == 0:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[edge_index_to_add] = 1
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    assert new_edge_set in self.valid_edge_sets_list, f"Edge set {new_edge_set} in set but not in list!"
                    index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append((new_edge_set, index))
        return possible_next_sets_with_indices

    def remove_fk_pk_edge(self,
                          current_edge_set: Tuple[int, ...]
                         ) -> List[Tuple[Tuple[int, ...], int]]:
        possible_next_sets_with_indices = []
        for edge_index_to_remove in self.fk_pk_indices:
            if current_edge_set[edge_index_to_remove] == 1:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[edge_index_to_remove] = 0
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    assert new_edge_set in self.valid_edge_sets_list, f"Edge set {new_edge_set} in set but not in list!"
                    index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append((new_edge_set, index))
        return possible_next_sets_with_indices

    def convert_row_to_edge(self,
                             current_edge_set: Tuple[int, ...]
                             ) -> List[Tuple[Tuple[int, ...], int]]:
        possible_next_sets_with_indices = []
        for f2p_pair, r2e_idx in self.f2p_pair_to_r2e_map.items():
            f2p_idx1, f2p_idx2 = f2p_pair
            if current_edge_set[r2e_idx] == 0 and current_edge_set[f2p_idx1] == 1 and current_edge_set[f2p_idx2] == 1:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[r2e_idx] = 1
                new_edge_set_list[f2p_idx1] = 0
                new_edge_set_list[f2p_idx2] = 0
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append((new_edge_set, index))
        return possible_next_sets_with_indices

    def convert_edge_to_row(self,
                             current_edge_set: Tuple[int, ...]
                             ) -> List[Tuple[Tuple[int, ...], int]]:
        possible_next_sets_with_indices = []
        for f2p_pair, r2e_idx in self.f2p_pair_to_r2e_map.items():
            f2p_idx1, f2p_idx2 = f2p_pair
            if current_edge_set[f2p_idx1] == 0 and current_edge_set[f2p_idx2] == 0 and current_edge_set[r2e_idx] == 1:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[f2p_idx1] = 1
                new_edge_set_list[f2p_idx2] = 1
                new_edge_set_list[r2e_idx] = 0
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append((new_edge_set, index))
        return possible_next_sets_with_indices