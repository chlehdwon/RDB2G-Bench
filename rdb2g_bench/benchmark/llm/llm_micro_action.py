from typing import Tuple, Optional, Type, Union, List
from torch_geometric.data import HeteroData

from ...common.search_space.search_space import TotalSearchSpace
from ...common.search_space.gnn_search_space import GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace
from ..micro_action import MicroActionSet

class LLMMicroActionSet(MicroActionSet):
    def __init__(self,
                 dataset: str,
                 task: str,
                 hetero_data: HeteroData,
                 GNNSpaceClass: Type[Union[GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace]],
                 num_layers: int,
                 src_entity_table: str,
                 dst_entity_table: Optional[str] = None):
        super().__init__(dataset, task, hetero_data, GNNSpaceClass, num_layers, src_entity_table, dst_entity_table)
        
    def add_fk_pk_edge(self,
            current_edge_set: Tuple[int, ...],
            from_table_name: str,
            from_col_name: str,
            to_table_name: str,
        ) -> Tuple[Tuple[int, ...], int, str]:
        graph_idx, error_msg = -1, ""
        edge_to_add = (from_table_name, f"f2p_{from_col_name}", to_table_name)
        if edge_to_add in self.full_edges:
            edge_index = self.full_edges.index(edge_to_add)
            if current_edge_set[edge_index] == 0:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[edge_index] = 1
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    graph_idx = self.valid_edge_sets_list.index(new_edge_set)
                    error_msg = ""
                else:
                    new_edge_set = current_edge_set
                    error_msg = f"Given add_fk_pk_edge action is not valid."
            else:
                new_edge_set = current_edge_set
                error_msg = f"Given edge type({edge_to_add}) between {from_table_name} and {to_table_name} is already connected."
        else:
            new_edge_set = current_edge_set
            error_msg = f"Given edge type({edge_to_add}) between {from_table_name} and {to_table_name} is an invalid edge type."

        return new_edge_set, graph_idx, error_msg
    
    def remove_fk_pk_edge(self,
            current_edge_set: Tuple[int, ...],
            from_table_name: str,
            from_col_name: str,
            to_table_name: str,
        ) -> Tuple[Tuple[int, ...], int, str]:
        graph_idx, error_msg = -1, ""
        edge_to_remove = (from_table_name, f"f2p_{from_col_name}", to_table_name)
        if edge_to_remove in self.full_edges:
            edge_index = self.full_edges.index(edge_to_remove)
            if current_edge_set[edge_index] == 1:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[edge_index] = 0
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    graph_idx = self.valid_edge_sets_list.index(new_edge_set)
                    error_msg = ""
                else:
                    new_edge_set = current_edge_set
                    error_msg = f"Given remove_fk_pk_edge action is not valid."
            else:
                new_edge_set = current_edge_set
                error_msg = f"Given edge type({edge_to_remove}) between {from_table_name} and {to_table_name} is not connected."
        else:
            new_edge_set = current_edge_set
            error_msg = f"Given edge type({edge_to_remove}) between {from_table_name} and {to_table_name} is an invalid edge type."

        return new_edge_set, graph_idx, error_msg

    def convert_row_to_edge(self,
            current_edge_set: Tuple[int, ...],
            table_1_name: str,
            table_2_name: str,
            edge_table_name: str
        ) -> Tuple[Tuple[int, ...], int, str]:
        graph_idx, error_msg = -1, ""
        if (table_1_name, f"r2e_{edge_table_name}", table_2_name) in self.full_edges:
            convert_edge = (table_1_name, f"r2e_{edge_table_name}", table_2_name)
            convert_edge_index = self.full_edges.index(convert_edge)
        elif (table_2_name, f"r2e_{edge_table_name}", table_1_name) in self.full_edges:
            convert_edge = (table_2_name, f"r2e_{edge_table_name}", table_1_name)
            convert_edge_index = self.full_edges.index(convert_edge)
        else:
            return current_edge_set, -1, f"Given edge type({edge_table_name}) between {table_1_name} and {table_2_name} is an invalid edge type."

        if current_edge_set[convert_edge_index] == 0:
            f2p_indices = self.r2e_to_f2p_map.get(convert_edge_index)
            if f2p_indices and f2p_indices[0] is not None and f2p_indices[1] is not None:
                f2p_idx1, f2p_idx2 = f2p_indices
                if current_edge_set[f2p_idx1] == 1 and current_edge_set[f2p_idx2] == 1:
                    new_edge_set_list = list(current_edge_set)
                    new_edge_set_list[convert_edge_index] = 1
                    new_edge_set_list[f2p_idx1] = 0
                    new_edge_set_list[f2p_idx2] = 0
                    new_edge_set = tuple(new_edge_set_list)
                    if new_edge_set in self.valid_edge_sets:
                        graph_idx = self.valid_edge_sets_list.index(new_edge_set)
                        error_msg = ""
                    else:
                        new_edge_set = current_edge_set
                        error_msg = f"Given convert_edge_to_row action is not valid."
                else:
                    new_edge_set = current_edge_set
                    error_msg = f"Given convert_edge_to_row action is not valid."
            else:
                new_edge_set = current_edge_set
                error_msg = f"Given convert_edge_to_row action is not valid."
        else:
            new_edge_set = current_edge_set
            error_msg = f"Given edge type({edge_table_name}) between {table_1_name} and {table_2_name} is already converted to edge."

        return new_edge_set, graph_idx, error_msg

    def convert_edge_to_row(self,
            current_edge_set: Tuple[int, ...],
            table_1_name: str,
            table_2_name: str,
            edge_table_name: str
        ) -> Tuple[Tuple[int, ...], int, str]:
        graph_idx, error_msg = -1, ""
        if (table_1_name, f"r2e_{edge_table_name}", table_2_name) in self.full_edges:
            convert_edge = (table_1_name, f"r2e_{edge_table_name}", table_2_name)
            convert_edge_index = self.full_edges.index(convert_edge)
        elif (table_2_name, f"r2e_{edge_table_name}", table_1_name) in self.full_edges:
            convert_edge = (table_2_name, f"r2e_{edge_table_name}", table_1_name)
            convert_edge_index = self.full_edges.index(convert_edge)
        else:
            return current_edge_set, -1, f"Given edge type({edge_table_name}) between {table_1_name} and {table_2_name} is an invalid edge type."
        
        if current_edge_set[convert_edge_index] == 1:
            f2p_indices = self.r2e_to_f2p_map.get(convert_edge_index)
            if f2p_indices and f2p_indices[0] is not None and f2p_indices[1] is not None:
                f2p_idx1, f2p_idx2 = f2p_indices
                if current_edge_set[f2p_idx1] == 0 and current_edge_set[f2p_idx2] == 0:
                    new_edge_set_list = list(current_edge_set)
                    new_edge_set_list[convert_edge_index] = 0
                    new_edge_set_list[f2p_idx1] = 1
                    new_edge_set_list[f2p_idx2] = 1
                    new_edge_set = tuple(new_edge_set_list)
                    if new_edge_set in self.valid_edge_sets:
                        graph_idx = self.valid_edge_sets_list.index(new_edge_set)
                        error_msg = ""
                    else:
                        new_edge_set = current_edge_set
                        error_msg = f"Given convert_edge_to_row action is not valid."
                else:
                    new_edge_set = current_edge_set
                    error_msg = f"Given convert_edge_to_row action is not valid."
            else:
                new_edge_set = current_edge_set
                error_msg = f"Given convert_edge_to_row action is not valid."
        else:
            new_edge_set = current_edge_set
            error_msg = f"Given edge type({edge_table_name}) between {table_1_name} and {table_2_name} is not converted to edge."

        return new_edge_set, graph_idx, error_msg

    def get_possible_add_fk_pk_edge(self,
                       current_edge_set: Tuple[int, ...]
                      ) -> List[Tuple[Tuple[int, ...], int]]:
        possible_next_sets_with_indices = []
        for edge_index_to_add in self.fk_pk_indices:
            if current_edge_set[edge_index_to_add] == 0:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[edge_index_to_add] = 1
                # new_edge_set_list = self.search_space.gnn_search_space.filter_unreachable(new_edge_set_list)
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    assert new_edge_set in self.valid_edge_sets_list, f"Edge set {new_edge_set} in set but not in list!"
                    # index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append(new_edge_set)
        return possible_next_sets_with_indices

    def get_possible_remove_fk_pk_edge(self,
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
                    possible_next_sets_with_indices.append(new_edge_set)
        return possible_next_sets_with_indices

    def get_possible_convert_row_to_edge(self,
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
                    possible_next_sets_with_indices.append(new_edge_set)
        return possible_next_sets_with_indices

    def get_possible_convert_edge_to_row(self,
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
                    possible_next_sets_with_indices.append(new_edge_set)
        return possible_next_sets_with_indices