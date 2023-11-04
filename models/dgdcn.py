
import torch
from torch import nn, Tensor
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

from model.dgnn import DGNN
from model.dcnn import DCNN
from datasets.temporal_graph import TemporalGraph
# from utils.temporal_walker import TemporalWalker


class DGDCN(nn.Module):
    def __init__(self, params, graph_in_channels,
                 graph_out_channels, dense_dim=0, sparse_dim=0,
                 sparse_emb_dim=0,
                 num_layers=2, mlp_dims=[128, 64]):
        super(DGDCN, self).__init__()
        self.params = params
        self.dgnn = DGNN(params, in_channels=graph_in_channels,
                         out_channels=graph_out_channels)
        self.dcnn = DCNN(params=params, dense_dim=(
            graph_out_channels * params['hist_len'] + dense_dim),
            sparse_dim=sparse_dim,
            num_layers=num_layers,
            sparse_emb_dim=sparse_emb_dim,
            mlp_dims=mlp_dims)

    def forward(self, sampler: NeighborSampler, graph: TemporalGraph,
                node_id: Tensor, node_time: Tensor,
                dense_features: Tensor = None,
                sparse_features: Tensor = None):
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

        # Concatenate the node embedding with the features
        if dense_features is not None:
            dense_features = torch.cat(
                (dense_features, target_node_emb), dim=1).float()
        else:
            dense_features = target_node_emb.float()
        # Input into DCN
        y_pred = self.dcnn(dense_features=dense_features,
                           sparse_features=sparse_features)

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
