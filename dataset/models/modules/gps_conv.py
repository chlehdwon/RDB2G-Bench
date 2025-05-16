# Custom implementation of GPSConv based on torch_geometric.nn.conv.GPSConv

import inspect
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.utils import to_dense_batch


class GPSConv(torch.nn.Module):
    r"""The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper.

    The GPS layer is based on a 3-part recipe:

    1. Inclusion of positional (PE) and structural encodings (SE) to the input
       features (done in a pre-processing step via
       :class:`torch_geometric.transforms`).
    2. A local message passing layer (MPNN) that operates on the input graph.
    3. A global attention layer that operates on the entire graph.

    .. note::

        For an example of using :class:`GPSConv`, see
        `examples/graph_gps.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        graph_gps.py>`_.

    Args:
        channels (int): Size of each input sample.
        conv (MessagePassing, optional): The local message passing layer.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        dropout (float, optional): Dropout probability of intermediate
            embeddings. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        attn_type (str): Global attention type, :obj:`multihead` or
            :obj:`performer`. (default: :obj:`multihead`)
        attn_kwargs (Dict[str, Any], optional): Arguments passed to the
            attention layer. (default: :obj:`None`)
    """
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'performer':
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            # TODO: Support BigBird
            raise ValueError(f'{attn_type} is not supported')

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        
        # Determine input for residual connections and global attention based on x type
        x_for_residual_and_global = x[1] if isinstance(x, tuple) else x

        if self.conv is not None:  # Local MPNN.
            # Pass edge_attr to the local message passing layer
            h_local = self.conv(x, edge_index, edge_attr=edge_attr, **kwargs)
            h_local = F.dropout(h_local, p=self.dropout, training=self.training)
            h_local = h_local + x_for_residual_and_global
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h_local = self.norm1(h_local, batch=batch)
                else:
                    h_local = self.norm1(h_local)
            hs.append(h_local)

        # Global attention transformer-style model.
        # Use x_for_residual_and_global for global attention input
        h_global_in, mask = to_dense_batch(x_for_residual_and_global, batch)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h_global, _ = self.attn(h_global_in, h_global_in, h_global_in, key_padding_mask=~mask,
                             need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h_global = self.attn(h_global_in, mask=mask)

        h_global = h_global[mask]
        h_global = F.dropout(h_global, p=self.dropout, training=self.training)
        h_global = h_global + x_for_residual_and_global  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h_global = self.norm2(h_global, batch=batch)
            else:
                h_global = self.norm2(h_global)
        hs.append(h_global)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads}, '
                f'attn_type={self.attn_type})')