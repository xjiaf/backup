import torch
from torch import nn
from torch_geometric.nn import Sequential, Linear

from module.dgnconv import DGNConv


class DGNN(nn.Module):
    def __init__(self, params, in_channels, out_channels):
        super(DGNN, self).__init__()
        self.params = params
        self.dgnn = Sequential('x, edge_index, edge_time,'
                               'node_time, edge_attr',
                               [(DGNConv(in_channels, 128), 'x, edge_index,'
                                 'edge_time, node_time, edge_attr -> x'),
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(),
                                (DGNConv(128, 128), 'x, edge_index,'
                                 'edge_time, node_time, edge_attr -> x'),
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(),
                                (Linear(128, 90, weight_initializer="glorot",
                                        bias_initializer="zeros"), 'x -> x'),
                                nn.LeakyReLU(),
                                (Linear(90, out_channels,
                                        weight_initializer="glorot",
                                        bias_initializer="zeros"), 'x -> x'),
                                nn.Dropout(params['dropout'])
                                ])

    def forward(self, x, edge_index, edge_time, node_time,
                edge_attr=None) -> torch.Tensor:
        node_emb = self.dgnn(x, edge_index, edge_time, node_time, edge_attr)

        return node_emb
