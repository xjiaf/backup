import torch
from torch import nn

from module.dgnconv import DGNConv


class DGNN(nn.Module):
    def __init__(self, params, in_channels, out_channels):
        super(DGNN, self).__init__()
        self.params = params
        self.conv1 = DGNConv(in_channels, 128)
        self.conv2 = DGNConv(128, 128, dropout=params['dropout'])
        self.fc = nn.Sequential(nn.BatchNorm1d(128),
                                nn.LeakyReLU(),  # nn.ReLU()
                                nn.Linear(128, 90),
                                nn.LeakyReLU(),  # nn.ReLU()
                                nn.Linear(90, out_channels),
                                nn.Dropout(params['dropout'])
                                )

    def forward(self, x, edge_index, edge_time, node_time,
                edge_weight=None, edge_attr=None) -> torch.Tensor:
        node_emb = self.conv1(x=x, edge_index=edge_index, edge_time=edge_time,
                              node_time=node_time, edge_weight=edge_weight)

        node_emb = self.conv2(x=node_emb, edge_index=edge_index, edge_time=edge_time,
                              node_time=node_time, edge_weight=edge_weight)

        node_emb = self.fc(node_emb)
        return node_emb
