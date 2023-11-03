import torch
from torch import nn

from modules.dgn import DGN


class DGNN(nn.Module):
    def __init__(self, params, in_channels, out_channels):
        super(DGNN, self).__init__()
        self.params = params
        self.dgn1 = DGN(in_channels, in_channels)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(),  # nn.ReLU()
            nn.Linear(in_channels, params['dgnn_hidden_dim']))
        self.dgn2 = DGN(params['dgnn_hidden_dim'], params['dgnn_hidden_dim'])
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(params['dgnn_hidden_dim']),
            nn.LeakyReLU(),  # nn.ReLU()
            nn.Linear(params['dgnn_hidden_dim'], out_channels),
            nn.LeakyReLU(),  # nn.ReLU()
            nn.Linear(out_channels, out_channels),
            nn.Dropout(params['dropout'])
            )

    def forward(self, x, edge_index, edge_time,
                node_time, edge_weight=None) -> torch.Tensor:
        node_emb = self.dgn1(x=x, edge_index=edge_index,
                             edge_time=edge_time,
                             node_time=node_time,
                             edge_weight=edge_weight)
        node_emb = self.fc1(node_emb.view(-1, node_emb.shape[-1])).view(
            node_time.shape[0], x.shape[0], -1)

        node_emb = self.dgn2(x=node_emb, edge_index=edge_index,
                             edge_time=edge_time,
                             node_time=node_time,
                             edge_weight=edge_weight)
        node_emb = self.fc2(node_emb.view(-1, node_emb.shape[-1])).view(
            node_time.shape[0], x.shape[0], -1)

        return node_emb