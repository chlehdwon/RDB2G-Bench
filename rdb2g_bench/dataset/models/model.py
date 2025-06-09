# Reference: https://github.com/snap-stanford/relbench/blob/main/examples/model.py

from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from .modules.nn import HeteroGraphSAGE, HeteroGIN,  HeteroGPS
from ..utils import divide_node_edge_dict

class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        gnn: str = 'GraphSAGE',
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        
        # Create temporal encoder if time attributes exist
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        
        # Initialize GNN based on specified type
        if gnn == "GraphSAGE":
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers,
            )
        elif gnn == "GIN":
            self.gnn = HeteroGIN(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
            )
        elif gnn == "GPS":
            self.gnn = HeteroGPS(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn}")

        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )
        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time if hasattr(batch[entity_table], "seed_time") else None
        x_dict = self.encoder(batch.tf_dict)
        if seed_time is not None:
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict, edge_dict = divide_node_edge_dict(batch, x_dict)

        x_dict = self.gnn(
            x_dict,
            edge_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        if seed_time is not None:
            return self.head(x_dict[entity_table][: seed_time.size(0)])
        else:
            return self.head(x_dict[entity_table][: batch[entity_table]['batch_size']])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time if hasattr(batch[entity_table], "seed_time") else None
        x_dict = self.encoder(batch.tf_dict)
        if seed_time is not None:
            x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict, edge_dict = divide_node_edge_dict(batch, x_dict)

        x_dict = self.gnn(
            x_dict,
            edge_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])