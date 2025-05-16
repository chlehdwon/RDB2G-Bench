import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GPSConv, NNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from relbench.base.task_base import TaskType

class PerformancePredictorGNN(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 task_type: TaskType,
                 src_node_idx: int,
                 dst_node_idx: int,
                 embedding_dim: int,
                 gnn_hidden_dim: int,
                 num_gnn_layers: int,
                 mlp_hidden_dim: int,
                 output_dim: int = 1,
                 gnn_type: str = "GCN",
                 gat_heads: int = 1,
                 dropout_rate: float = 0.5,
                 embedding_type: str = "onehot",
                 pooling_type: str = "mean",
                 alpha: float = 1.0):
        super().__init__()
        self.num_node_types = num_node_types
        self.gnn_type = gnn_type.upper()
        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type
        self.alpha = alpha
        self.src_node_idx = src_node_idx
        self.dst_node_idx = dst_node_idx
        self.task_type = task_type

        if embedding_type == "onehot":
            self.node_emb = nn.Embedding(num_node_types, num_node_types)
            self.node_emb.weight.data = torch.eye(num_node_types)
            self.node_emb.weight.requires_grad = False
            in_channels = num_node_types
        elif embedding_type == "random":
            self.node_emb = nn.Embedding(num_node_types, embedding_dim)
            self.node_emb.weight.data = torch.randn(num_node_types, embedding_dim)
            in_channels = embedding_dim

        # GNN Layers (Dynamically created based on gnn_type)
        self.gnn_layers = nn.ModuleList()
        out_channels = gnn_hidden_dim

        for i in range(num_gnn_layers):
            if i > 0:
                in_channels = gnn_hidden_dim
                if self.gnn_type == 'GAT':
                     in_channels = gnn_hidden_dim * gat_heads

            # Determine if this is the last GNN layer
            is_last_layer = (i == num_gnn_layers - 1)

            if self.gnn_type == 'GCN':
                layer = GCNConv(in_channels, out_channels)
            elif self.gnn_type == 'GAT':
                layer = GATConv(in_channels, out_channels, heads=gat_heads, dropout=dropout_rate)
                if is_last_layer:
                    out_channels = out_channels * gat_heads
            elif self.gnn_type == 'GRAPHSAGE':
                # SAGEConv(in_channels, out_channels, aggr='mean', **kwargs) # aggr can be mean, lstm, etc.
                layer = SAGEConv(in_channels, out_channels)
            elif self.gnn_type == 'GIN':
                # GIN requires an MLP for each layer
                gin_mlp = nn.Sequential(
                    nn.Linear(in_channels, gnn_hidden_dim), # Use gnn_hidden_dim for internal MLP processing
                    nn.ReLU(),
                    nn.Linear(gnn_hidden_dim, out_channels)
                )
                layer = GINConv(gin_mlp, train_eps=False) # train_eps=False is common
            elif self.gnn_type == 'GPS':
                raise ValueError("GPS is not supported yet.")
            elif self.gnn_type == 'NN':
                edge_feature_dim = self.node_emb.embedding_dim
                nn_mlp = nn.Sequential(
                    nn.Linear(edge_feature_dim, gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gnn_hidden_dim, in_channels * out_channels)
                )
                layer = NNConv(in_channels, out_channels, nn=nn_mlp, aggr='mean')
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}. Choose from 'GCN', 'GAT', 'GraphSAGE', 'GIN', 'GPS', 'NN'.")

            self.gnn_layers.append(layer)

        # Skip connections projections
        self.skip_projections = nn.ModuleList()
        current_h_channels = embedding_dim if embedding_type == "random" else num_node_types
        for i in range(num_gnn_layers):
            layer = self.gnn_layers[i]
            hf_channels = layer.out_channels
            if isinstance(layer, GATConv):
                 hf_channels *= layer.heads # GAT output is heads * out_channels

            projection = nn.Identity()
            if current_h_channels != hf_channels:
                 projection = nn.Linear(current_h_channels, hf_channels)
            self.skip_projections.append(projection)
            current_h_channels = hf_channels # Update for next layer's input h dimension

        # MLP Head for Regression
        mlp_input_dim = current_h_channels # Input to MLP is the output dim of the last GNN layer
        
        if self.pooling_type == "target" and self.task_type == TaskType.LINK_PREDICTION:
            mlp_input_dim = mlp_input_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """
        Forward pass of the GNN model.

        Args:
            data (Data | Batch): A PyTorch Geometric Data or Batch object containing:
                                 - x: Node features (indices in this case).
                                 - edge_index: Edge connectivity.
                                 - batch: Batch assignment vector (if processing a Batch).

        Returns:
            torch.Tensor: The predicted performance value(s) (shape: [batch_size, output_dim]).
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch_vector = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Set node embeddings
        h = self.node_emb(x)

        # Set edge attributes
        if edge_attr is not None:
            valid_mask = edge_attr != -1
            valid_indices = edge_attr[valid_mask]
            edge_features = torch.zeros(edge_attr.size(0), self.node_emb.embedding_dim, device=edge_attr.device, dtype=h.dtype)
            if valid_indices.numel() > 0:
                valid_embeddings = self.node_emb(valid_indices)
                edge_features[valid_mask] = valid_embeddings
            edge_attr = edge_features
        else:
            edge_attr = None

        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            h_input = h # Store input for skip connection

            if self.gnn_type == 'GAT':
                # GATConv signature usually (x, edge_index), edge_attr optional depending on init
                # Assuming basic GAT for now, not using edge_attr in the call
                hf = layer(h_input, edge_index)
                h_proj = self.skip_projections[i](h_input)
                h = (1 - self.alpha) * h_proj + self.alpha * hf
                # Activation/dropout usually internal to GATConv or applied differently
            elif self.gnn_type == 'NN':
                # NNConv requires edge_attr through nn module, but call signature is (x, edge_index, edge_weight/edge_attr)
                # Assuming edge_attr here corresponds to edge features needed by nn
                hf = layer(h_input, edge_index, edge_attr)
                h_proj = self.skip_projections[i](h_input)
                h = (1 - self.alpha) * h_proj + self.alpha * hf
                if i < len(self.gnn_layers) - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout_rate, training=self.training)
            else: # GCN, SAGE, GIN
                # These typically take (x, edge_index). edge_attr might be used for weighting if layer supports it.
                # Assuming standard usage without edge_attr in the call
                hf = layer(h_input, edge_index)
                h_proj = self.skip_projections[i](h_input)
                h = (1 - self.alpha) * h_proj + self.alpha * hf
                if i < len(self.gnn_layers) - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout_rate, training=self.training)


        # Pooling (readout)
        if self.pooling_type == "target":
            # Get node pointer for batch processing
            ptr = data.ptr
            global_src_indices = ptr[:-1] + self.src_node_idx

            if self.task_type == TaskType.LINK_PREDICTION:
                global_dst_indices = ptr[:-1] + self.dst_node_idx
                src_embeddings = h[global_src_indices]
                dst_embeddings = h[global_dst_indices]
                graph_embedding = torch.cat([src_embeddings, dst_embeddings], dim=1)
            else:
                graph_embedding = h[global_src_indices]
        else: # Global pooling
            # Only for connected nodes
            unique_connected_nodes = torch.unique(edge_index.flatten())
            connected_mask = torch.zeros(h.size(0), device=h.device, dtype=torch.bool)
            connected_mask[unique_connected_nodes] = True

            h = h[connected_mask]
            batch_vector = batch_vector[connected_mask]

            if self.pooling_type == "mean":
                graph_embedding = global_mean_pool(h, batch_vector)
            elif self.pooling_type == "sum":
                graph_embedding = global_add_pool(h, batch_vector)

        # MLP Head for prediction
        prediction = self.mlp(graph_embedding)

        return prediction

    def margin_loss(self, pred: torch.Tensor, actual: torch.Tensor, margin: float) -> torch.Tensor:
        pred, actual = pred.squeeze(), actual.squeeze()
        actual = actual.cpu().detach().numpy()
        perf_diff = actual[:, None] - actual
        perf_abs_diff_matrix = np.triu(np.abs(perf_diff), 1)
        ex_thresh_inds = np.where(perf_abs_diff_matrix > 0.0)
        
        better_labels = (perf_diff > 0)[ex_thresh_inds]

        s_1 = pred[ex_thresh_inds[1]]
        s_2 = pred[ex_thresh_inds[0]]

        better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
        zero_, margin = s_1.new([0.0]), s_1.new([margin])

        loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))

        return loss