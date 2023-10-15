from typing import List, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import init, Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import (
    Adj,
    OptTensor,
)


class POSConv(MessagePassing):
    def __init__(self, in_channels: Optional[Union[int, float]],
                 out_channels: Optional[Union[int, float]],
                 aggr: Optional[Union[str, List[str], Aggregation]] = "add"):
        super(DGNConv, self).__init__(aggr=aggr)  # "Add" aggregation
        self.delta = Parameter(torch.tensor(1.0))
        self.bn = nn.BatchNorm1d(1, affine=True)  # Added for edge weights normalization
        self.w_self = Linear(in_channels, out_channels)
        self.w_hist = Linear(in_channels, out_channels)

        # Initialize parameters
        init.xavier_normal_(self.delta)
        init.xavier_normal_(self.w_self.weight)
        init.xavier_normal_(self.w_hist.weight)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_time, node_time,
                edge_weight: OptTensor = None) -> Tensor:
        """
        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_time: Edge time
            node_time: Node time
            edge_weight: Edge weights
        Returns:
            out: Output features, size [num_nodes, out_channels]
        """
        if torch.is_tensor(node_time) and node_time.dim() == 0 or not torch.is_tensor(node_time):
            node_time = torch.full_like(edge_time, node_time, dtype=torch.float)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             edge_time=edge_time, node_time=node_time)
        return out

    def message(self, x_j, edge_time, node_time, edge_weight, position):
        # Calculate time differences
        time_diff = node_time - edge_time

        # Create a mask for the condition
        mask = (node_time >= edge_time).bool()

        # Normalize edge weights
        if edge_weight is not None:
            normalized_edge_weight = torch.zeros_like(edge_time)
            normalized_edge_weight[mask] = self.bn(edge_weight[mask].unsqueeze(1)).squeeze(1)

        # Only calculate time decay for the valid time_diffs
        kappa_time_diff = torch.zeros_like(time_diff)
        kappa_time_diff[mask] = F.softmax(-self.delta * time_diff[mask], dim=0)

        if edge_weight is not None:
            feat = normalized_edge_weight * kappa_time_diff * mask * x_j
        else:
            feat = kappa_time_diff * mask * x_j
        
        # Calculate position info
        step = torch.tensor(2.0)
        position = (position) * step * torch.exp(-time_diff) * mask
        return feat, position

    def update(self, aggr_out, x, n_id):
        feat_out = self.w_self(x) + self.w_hist(aggr_out[0])
        position_out = aggr_out[1] + n_id
        return feat_out, position_out
