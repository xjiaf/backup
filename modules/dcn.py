import torch
import torch.nn as nn


class DCN(nn.Module):
    def __init__(self, in_channels, num_layers, mlp_dims, dropout):
        super().__init__()

        # DNN layer
        dnn_layers = []
        dnn_in = in_channels
        for mlp_dim in mlp_dims:
            dnn_layers.append(
                nn.Linear(dnn_in, mlp_dim))
            dnn_layers.append(nn.BatchNorm1d(mlp_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(p=dropout))
            dnn_in = mlp_dim
        self.mlp = nn.Sequential(*dnn_layers)

        # Corss Net layer
        self.num_layers = num_layers
        self.cross_w = nn.ModuleList(
            [nn.Linear(in_channels, 1, bias=False) for _ in range(num_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(
            torch.zeros((in_channels,))) for _ in range(num_layers)])

        # LR layer
        self.lr = nn.Linear(mlp_dims[-1] + in_channels, 1)

    def forward(self, x):
        # DNN out
        mlp_part = self.mlp(x)

        # Cross Net out
        x0 = x
        cross = x
        for i in range(self.num_layers):
            xw = self.cross_w[i](cross)
            cross = x0 * xw + self.cross_b[i] + cross

        # stack output
        out = torch.cat([cross, mlp_part], dim=1)

        # LR out
        out = self.lr(out)
        out = torch.sigmoid(out)

        return out
