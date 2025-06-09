from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType
from .sage_conv_edge import SAGEConvEdge
from .gin_conv import GINConv, GINEConv
from .gps_conv import GPSConv


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConvEdge((channels, channels), channels, aggr=aggr)
                    if edge_type[1].startswith('r2e_') or edge_type[1].startswith('rev_r2e_')
                    else SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            edge_attr_dict = {}
            
            for edge_type in edge_index_dict.keys():
                if edge_type in edge_dict:
                    edge_attr_dict[edge_type] = edge_dict[edge_type]
            
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGIN(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn_gin = torch.nn.Sequential(
                torch.nn.Linear(channels, channels * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(channels * 2, channels),
            )
            conv = HeteroConv(
                {
                    edge_type: GINEConv(nn=nn_gin, train_eps=True)
                    if edge_type[1].startswith('r2e_') or edge_type[1].startswith('rev_r2e_')
                    else GINConv(nn=nn_gin, train_eps=True)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            edge_attr_dict = {}
            
            for edge_type in edge_index_dict.keys():
                if edge_type in edge_dict:
                    edge_attr_dict[edge_type] = edge_dict[edge_type]
            
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGPS(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        num_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.0,
        attn_type: str = 'multihead',
        aggr: str = "mean",
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                local_mpnn = SAGEConvEdge(
                    in_channels=(channels, channels),
                    out_channels=channels,
                    aggr=aggr
                )
                gps_layer = GPSConv(
                    channels=channels,
                    conv=local_mpnn,
                    heads=heads,
                    dropout=dropout,
                    attn_type=attn_type,
                    norm=None,
                )
                conv_dict[edge_type] = gps_layer
            
            conv = HeteroConv(conv_dict, aggr="sum")
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for conv, norm_dict in zip(self.convs, self.norms):
            edge_attr_dict = {}
            for edge_type in edge_index_dict.keys():
                if edge_type in edge_dict:
                    edge_attr_dict[edge_type] = edge_dict[edge_type]
            
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
