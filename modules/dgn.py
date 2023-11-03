from typing import Optional, Union

import torch
from torch import nn, Tensor
from torch_geometric.typing import (
    Adj,
    OptTensor,
)

from modules.dgnconv import DGNConv


class DGN(nn.Module):
    def __init__(self, in_channels: Optional[Union[int, float]],
                 out_channels: Optional[Union[int, float]]):
        super().__init__()
        self.conv1 = DGNConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj, edge_time: Tensor,
                node_time: Tensor, edge_weight: OptTensor = None) -> Tensor:

        out_nodes = []

        for i in range(node_time.shape[0]):
            ntime = node_time[i]

            if x.dim() == 2:
                x = x.repeat(node_time.shape[0], 1, 1)
            elif x.dim() == 3:
                assert x.shape[0] == node_time.shape[0]

            node_emb = self.conv1(x=x[i], node_time=ntime,
                                  edge_index=edge_index,
                                  edge_time=edge_time,
                                  edge_weight=edge_weight)

            # Store the result for the current node
            out_nodes.append(node_emb)

        # Stack results for all nodes in node_id
        out = torch.stack(out_nodes, dim=0)  # (batch_size, x.shape[0], out_channels)

        return out
