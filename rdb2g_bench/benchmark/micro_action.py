from typing import Tuple, Set, Optional, Type, Union, List, Dict
import torch
from torch_geometric.data import HeteroData

from ..common.search_space.search_space import TotalSearchSpace
from ..common.search_space.gnn_search_space import GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace

class MicroActionSet:
    """
    Defines micro actions for graph construction transformation.
    
    This class provides atomic operations to transform graph constructions by
    adding/removing edges and converting between different representations.
    It maintains mappings between different edge types and ensures that all
    transformations result in valid graph configurations.
    
    Attributes:
        search_space (TotalSearchSpace): The search space containing all valid graphs
        full_edges (List[Tuple[str, str, str]]): List of all possible edge types
        fk_pk_indices (Set[int]): Indices of foreign key-primary key edges
        r2e_indices (Set[int]): Indices of row-to-edge transformation edges
        valid_edge_sets_list (List[Tuple[int, ...]]): Sorted list of all valid edge sets
        valid_edge_sets (Set[Tuple[int, ...]]): Set of all valid edge sets for fast lookup
        r2e_to_f2p_map (Dict[int, Tuple[Optional[int], Optional[int]]]): Mapping from 
            r2e indices to corresponding f2p index pairs
        f2p_pair_to_r2e_map (Dict[Tuple[int, int], int]): Mapping from f2p index pairs
            to corresponding r2e indices
    
    Example:
        >>> micro_actions = MicroActionSet(
        ...     dataset="rel-f1",
        ...     task="driver-top3", 
        ...     hetero_data=data,
        ...     GNNSpaceClass=GNNNodeSearchSpace,
        ...     num_layers=2,
        ...     src_entity_table="drivers"
        ... )
        >>> current_set = (1, 0, 1, 0)
        >>> new_sets = micro_actions.add_fk_pk_edge(current_set)
    """
    
    def __init__(self,
                 dataset: str,
                 task: str,
                 hetero_data: HeteroData,
                 GNNSpaceClass: Type[Union[GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace]],
                 num_layers: int,
                 src_entity_table: str,
                 dst_entity_table: Optional[str] = None):
        """
        Initialize the MicroActionSet.
        
        Args:
            dataset (str): Name of the dataset (e.g., "rel-f1")
            task (str): Name of the task (e.g., "driver-top3")
            hetero_data (HeteroData): Heterogeneous graph data object
            GNNSpaceClass (Type): GNN search space class to use
            num_layers (int): Number of GNN layers
            src_entity_table (str): Source entity table name
            dst_entity_table (Optional[str]): Destination entity table name for link tasks
        """
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
        """
        Add a foreign key-primary key edge to the current edge set.
        
        This action explores adding table connections by including foreign key
        to primary key relationships that are currently not in the graph.
        
        Args:
            current_edge_set (Tuple[int, ...]): Current binary edge set representation
                where 1 indicates edge is included, 0 indicates it's not
                
        Returns:
            List[Tuple[Tuple[int, ...], int]]: List of possible next edge sets with their indices.
            
            Each tuple contains:

            - new_edge_set (Tuple[int, ...]): The modified edge set with an added FK-PK edge
            - index (int): Index of the new edge set in the valid_edge_sets_list
                
        Example:
            >>> current = (1, 0, 1, 0)  # Some edges included, some not
            >>> new_sets = micro_actions.add_fk_pk_edge(current)
            >>> # Returns: [((1, 1, 1, 0), 5), ((1, 0, 1, 1), 7)]
        """
        possible_next_sets_with_indices = []
        for edge_index_to_add in self.fk_pk_indices:
            if current_edge_set[edge_index_to_add] == 0:  # Edge not currently included
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[edge_index_to_add] = 1  # Add the edge
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    assert new_edge_set in self.valid_edge_sets_list, f"Edge set {new_edge_set} in set but not in list!"
                    index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append((new_edge_set, index))
        return possible_next_sets_with_indices

    def remove_fk_pk_edge(self,
                          current_edge_set: Tuple[int, ...]
                         ) -> List[Tuple[Tuple[int, ...], int]]:
        """
        Remove a foreign key-primary key edge from the current edge set.
        
        This action explores simplifying the graph by removing table connections
        that are currently included in the graph.
        
        Args:
            current_edge_set (Tuple[int, ...]): Current binary edge set representation
                where 1 indicates edge is included, 0 indicates it's not
                
        Returns:
            List[Tuple[Tuple[int, ...], int]]: List of possible next edge sets with their indices.
            
            Each tuple contains:

            - new_edge_set (Tuple[int, ...]): The modified edge set with a removed FK-PK edge
            - index (int): Index of the new edge set in the valid_edge_sets_list
                
        Example:
            >>> current = (1, 1, 1, 0)  # Some edges included
            >>> new_sets = micro_actions.remove_fk_pk_edge(current)
            >>> # Returns: [((0, 1, 1, 0), 2), ((1, 0, 1, 0), 3)]
        """
        possible_next_sets_with_indices = []
        for edge_index_to_remove in self.fk_pk_indices:
            if current_edge_set[edge_index_to_remove] == 1:  # Edge currently included
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[edge_index_to_remove] = 0  # Remove the edge
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    assert new_edge_set in self.valid_edge_sets_list, f"Edge set {new_edge_set} in set but not in list!"
                    index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append((new_edge_set, index))
        return possible_next_sets_with_indices

    def convert_row_to_edge(self,
                             current_edge_set: Tuple[int, ...]
                             ) -> List[Tuple[Tuple[int, ...], int]]:
        """
        Convert row representation to edge representation.
        
        This action transforms a graph that uses table rows (via two FK-PK edges)
        to represent relationships into a direct edge representation. It replaces
        two foreign key connections with a single direct edge.
        
        Args:
            current_edge_set (Tuple[int, ...]): Current binary edge set representation
                
        Returns:
            List[Tuple[Tuple[int, ...], int]]: List of possible next edge sets with their indices.
            
            Each tuple contains:

            - new_edge_set (Tuple[int, ...]): The modified edge set with row-to-edge conversion
            - index (int): Index of the new edge set in the valid_edge_sets_list
                
        Note:
            This conversion is only possible when both required FK-PK edges are present
            and the corresponding R2E edge is not yet included.
            
        Example:
            >>> # Current set has FK edges to represent relationship via table row
            >>> current = (1, 1, 0, 0)  # Two FK-PK edges, no direct edge
            >>> new_sets = micro_actions.convert_row_to_edge(current)
            >>> # Returns: [((0, 0, 1, 0), 4)]  # Direct edge, no FK edges
        """
        possible_next_sets_with_indices = []
        for f2p_pair, r2e_idx in self.f2p_pair_to_r2e_map.items():
            f2p_idx1, f2p_idx2 = f2p_pair
            # Check if conversion is possible: r2e not present, both f2p present
            if current_edge_set[r2e_idx] == 0 and current_edge_set[f2p_idx1] == 1 and current_edge_set[f2p_idx2] == 1:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[r2e_idx] = 1    # Add direct edge
                new_edge_set_list[f2p_idx1] = 0   # Remove first FK edge
                new_edge_set_list[f2p_idx2] = 0   # Remove second FK edge
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append((new_edge_set, index))
        return possible_next_sets_with_indices

    def convert_edge_to_row(self,
                             current_edge_set: Tuple[int, ...]
                             ) -> List[Tuple[Tuple[int, ...], int]]:
        """
        Convert edge representation to row representation.
        
        This action transforms a graph that uses direct edges to represent
        relationships into a table row representation. It replaces a single
        direct edge with two foreign key connections through an intermediate table.
        
        Args:
            current_edge_set (Tuple[int, ...]): Current binary edge set representation
                
        Returns:
            List[Tuple[Tuple[int, ...], int]]: List of possible next edge sets with their indices.
            
            Each tuple contains:

            - new_edge_set (Tuple[int, ...]): The modified edge set with edge-to-row conversion  
            - index (int): Index of the new edge set in the valid_edge_sets_list
                
        Note:
            This conversion is only possible when the R2E edge is present and
            both corresponding FK-PK edges are not yet included.
            
        Example:
            >>> # Current set has direct edge representation
            >>> current = (0, 0, 1, 0)  # Direct edge, no FK edges
            >>> new_sets = micro_actions.convert_edge_to_row(current)
            >>> # Returns: [((1, 1, 0, 0), 2)]  # Two FK edges, no direct edge
        """
        possible_next_sets_with_indices = []
        for f2p_pair, r2e_idx in self.f2p_pair_to_r2e_map.items():
            f2p_idx1, f2p_idx2 = f2p_pair
            # Check if conversion is possible: both f2p not present, r2e present
            if current_edge_set[f2p_idx1] == 0 and current_edge_set[f2p_idx2] == 0 and current_edge_set[r2e_idx] == 1:
                new_edge_set_list = list(current_edge_set)
                new_edge_set_list[f2p_idx1] = 1   # Add first FK edge
                new_edge_set_list[f2p_idx2] = 1   # Add second FK edge
                new_edge_set_list[r2e_idx] = 0    # Remove direct edge
                new_edge_set = tuple(new_edge_set_list)
                if new_edge_set in self.valid_edge_sets:
                    index = self.valid_edge_sets_list.index(new_edge_set)
                    possible_next_sets_with_indices.append((new_edge_set, index))
        return possible_next_sets_with_indices