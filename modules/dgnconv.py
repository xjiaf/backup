from typing import List, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch_geometric.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import spmm
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor
)


class DGNConv(MessagePassing):
    def __init__(self, in_channels: Optional[Union[int, float]],
                 out_channels: Optional[Union[int, float]],
                 aggr: Optional[Union[str, List[str], Aggregation]] = "add"):
        """
        in_channels: the dim of per node features
        out_channels: the dim of the transformed per node features
        """

        super().__init__(aggr=aggr)  # "Add" aggregation
        self.delta = Parameter(torch.tensor(1.0))
        self.ln = nn.LayerNorm(1, elementwise_affine=True)  # Using LayerNorm here
        self.W_self = Linear(in_channels, out_channels,
                             weight_initializer="glorot",
                             bias_initializer=None)
        self.W_hist = Linear(in_channels, out_channels,
                             weight_initializer="glorot",
                             bias_initializer=None)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.W_self.reset_parameters()
        self.W_hist.reset_parameters()
        self.delta.data.uniform_(0, 1)
        self.ln.reset_parameters()  # Reset LayerNorm parameters

    def forward(self, x: Tensor, edge_index: Adj, edge_time: Tensor,
                node_time: Tensor, edge_weight: OptTensor = None) -> Tensor:

        coeff = self.calculate_coeff(edge_time, node_time, edge_weight)
        out = self.propagate(edge_index, x=x, coeff=coeff)

        return out

    def calculate_coeff(self, edge_time, node_time, edge_weight):
        # Calculate time differences
        time_diff = node_time - edge_time
        time_diff = time_diff.float()  # Convert to float

        # Create a mask for the condition
        mask = (node_time >= edge_time).bool()

        # Normalize edge weights
        if edge_weight is not None:
            normalized_edge_weight = torch.zeros_like(edge_weight)
            valid_edge_weights = torch.masked_select(
                edge_weight.float(), mask)  # Convert to float
            normalized_valid_edge_weights = self.ln(
                valid_edge_weights.unsqueeze(1)).squeeze(1)
            normalized_edge_weight.masked_scatter_(
                mask, normalized_valid_edge_weights)

        # Only calculate time decay for the valid time_diffs
        kappa_time_diff = torch.zeros_like(time_diff)  # Ensure a float tensor
        valid_time_diffs = torch.masked_select(time_diff, mask)
        kappa_valid_time_diffs = torch.softmax(
            -self.delta * valid_time_diffs, dim=0)
        kappa_time_diff.masked_scatter_(mask, kappa_valid_time_diffs)

        if edge_weight is not None:
            coeff = normalized_edge_weight * kappa_time_diff
        else:
            coeff = kappa_time_diff

        return coeff

    def message(self, x_j: Tensor, coeff: Tensor):
        coeff = coeff.unsqueeze(-1)
        return coeff * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor, coeff: Tensor) -> Tensor:
        x = x * coeff.unsqueeze(-1)  # Apply coefficients to features
        return spmm(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out, x):
        return self.W_self(x) + self.W_hist(aggr_out)
