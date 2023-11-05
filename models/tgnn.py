import torch
from torch.nn import Module
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

from modules.link_predictor import LinkPredictor


class TGNModel(Module):
    def __init__(self, num_nodes, msg_dim,
                 memory_dim, time_dim, embedding_dim):
        super(TGNModel, self).__init__()

        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
        )

        self.link_pred = LinkPredictor(in_channels=embedding_dim)

        # Helper vector to map global node indices to local ones.
        self.assoc = torch.empty(num_nodes, dtype=torch.long)

    def forward(self, src, dst, neg_dst, n_id, edge_index,
                e_id, last_update, msg, t):
        # Update memory and get node embeddings
        z, last_update = self.memory(n_id)
        z = self.gnn(z, last_update, edge_index, t[e_id], msg[e_id])
        z_src, z_dst, z_neg_dst = z[
            self.assoc[src]], z[self.assoc[dst]], z[self.assoc[neg_dst]]

        # Link prediction
        pos_out = self.link_pred(z_src, z_dst)
        neg_out = self.link_pred(z_src, z_neg_dst)

        return pos_out, neg_out

    def reset_state(self):
        self.memory.reset_state()

    def detach(self):
        self.memory.detach()

    def update_state(self, src, dst, t, msg):
        self.memory.update_state(src, dst, t, msg)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)
