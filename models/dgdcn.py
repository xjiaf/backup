import torch
from torch import nn, Tensor
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput
from torch_geometric.nn.models import MLP

from models.dgnn import DGNN
from datasets.temporal_graph import TemporalGraph
# from utils.temporal_walker import TemporalWalker


class DGNN(nn.Module):
    def __init__(self, params, graph_in_channels,
                 graph_out_channels, mlp_dims=[128, 64],
                 dropout=0.3):
        super().__init__()
        self.params = params
        self.dgnn = DGNN(params, in_channels=graph_in_channels,
                         out_channels=graph_out_channels)
        self.mlp = MLP(channel_list=[graph_out_channels, *mlp_dims],
                       dropout=dropout, act='relu', act_first=True,
                       act_kwargs=None, norm='batch_norm',
                       norm_kwargs=None, plain_last=True, bias=True)

    def forward(self, sampler: NeighborSampler, graph: TemporalGraph,
                node_id: Tensor, node_time: Tensor):

        # Preprocess the input tensors
        device = node_id.device
        batch_size = node_id.size(0)
        hist_len = node_id.size(1)
        valid_indices = (node_id != -1)
        flat_item_ids = torch.masked_select(node_id, valid_indices)
        flat_item_times = torch.masked_select(node_time, valid_indices)
        original_positions = torch.nonzero(valid_indices).squeeze(-1)[:, 0]
        
        # Sample the neighborhood of the target node
        sampler_output = sampler.sample_from_nodes(
            NodeSamplerInput(input_id=None, node=flat_item_ids,
                             time=flat_item_times))
        edge_index = torch.stack([sampler_output.row,
                                  sampler_output.col], dim=0).to(device)
        x = graph.x[sampler_output.node].to(device)
        edge_time = graph.edge_time[sampler_output.edge].to(device)
        edge_weight = graph.edge_weight[sampler_output.edge].to(device)

        # Get the node embedding
        node_emb = self.dgnn(x=x, edge_index=edge_index,
                             edge_time=edge_time, node_time=flat_item_times,
                             edge_weight=edge_weight)

        target_node_indice = torch.nonzero(
            sampler_output.node[:, None].to(
                device) == flat_item_ids, as_tuple=True)[1]
        target_node_emb = []
        for emb, idx in zip(node_emb, target_node_indice):
            target_node_emb.append(emb[idx])
        else:
            target_node_emb = torch.stack(target_node_emb, dim=0)

        # Remap the node embedding to the original shape
        target_node_emb = remap_embeddings(
            target_node_emb, original_positions,
            batch_size, hist_len, device)

        # Input into DCN
        out = self.mlp(target_node_emb)
        y_pred = torch.sigmoid(out)
        return y_pred


def remap_embeddings(embeddings, original_positions: Tensor,
                     batch_size: int, hist_len: int, device):
    """
    Reshape embeddings back to original shape and then compress to a 2D tensor.
    """
    embedding_dim = embeddings.size(-1)
    output_tensor = torch.zeros(batch_size, hist_len,
                                embedding_dim, device=device)
    for idx, position in enumerate(original_positions):
        # Ensure position is within bounds
        if position >= batch_size:
            raise ValueError(f"Position {position} is out of"
                             "bounds for batch size {batch_size}")

        zero_rows = (output_tensor[position] == 0).all(dim=1).nonzero(
            as_tuple=True)

        if zero_rows[0].size(0) == 0:
            raise ValueError(f"No zero row found for position {position}")

        zero_row_idx = zero_rows[0][0]
        output_tensor[position][zero_row_idx] = embeddings[idx]

    # Compressing the tensor to 2D
    return output_tensor.view(batch_size, -1)
