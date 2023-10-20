from __future__ import annotations
import os
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor
from torch_geometric.data import Data


class TemporalGraph(Data):
    def __init__(self, edge=None, x=None, directed=True, **kwargs):
        """
        Parameters:
        - edge : the edge data file_path or np/tensor object.
            Expected format of loaded data is a 2D array where
            each row represents an edge with columns:
            [source_node_id (long), target_node_id (long),
                    time_stamp (long), edge_weight (float)/edge_attr(tensor float)].

        - x: Path to the node feature file or np/tensor object.
            Expected format of loaded data is a 2D array
            where each row represents a node with the row index being
            node_id (long) and columns are feature values.

        - directed (bool): Indicates whether the graph is
            directed or undirected. Default is False.

        Attributes:
        - neighbor_sequence (defaultdict of list): Maps each
            node to a list of tuples, where each tuple represents a
            (historical neighbors, time, weight). the historical
            neighbors of each node ordered by ascending
            Format: {node_1: [(neighbor_node_a1, time_1a,
                edge_attr_1a), ...], node_2:[...],...}.

        - stream_graph (defaultdict of defaultdict of list):
            Maps each node to a dictionary where keys are
            time stamps (ascending ordered) and values are lists of
            neighboring nodes at that time. Format:
            {node_1: {time_1: [(neighbor_node_a, edge_attr_a), ...],
                time_2: [(neighbor_node_b, edge_attr_b), ...], ...}, ...}.
        """
        # Call base class init
        super().__init__(directed=directed, edge_time=None, edge_attr=None,
                         neighbor_sequence=None, stream_graph=None, **kwargs)

        # Load and process data
        if edge is not None:
            self.edge_index, self.edge_time, self.edge_attr = self.process_edge(edge)

        if x is not None:
            self.x = self._load_node_features(x)

        # self.directed = directed
        self.neighbor_sequence = defaultdict(list)
        self.stream_graph = {}  # stream graph data

    def process_edge(self, edge_data: Tensor) -> (Tensor, Tensor, Tensor):
        edge_tuple = self._load_edge(edge_data)
        if not self.directed:
            edge_tuple = to_undirected(*edge_tuple)
            return remove_duplicate_edges(*edge_tuple)
        else:
            return edge_tuple

    def update_further_graph(self, new_edge_index: Tensor = None,
                             new_edge_time: Tensor = None, new_edge_attr=None):
        """
        Update the TemporalGraph with new edges, timestamps, and weights.

        Parameters:
        - new_edge_index: A 2xN tensor where N is the number of edges.
                        The first row contains source nodes,
                        and the second row contains target nodes.
        - new_edge_time: A 1D tensor of length N, representing
                        the timestamps for each edge.
        - new_edge_attr: A 1D tensor of length N,
                        representing the weights for each edge.

        This function will:
        1. Update the neighbor_sequence dictionary with new edges.
        2. Update the stream_graph dictionary to maintain
                edges in a temporal order.
        """
        # Determine the iterators based on the presence of new_edge_attr
        if not (new_edge_index and new_edge_time):
            new_edge_index, new_edge_time, new_edge_attr = self.edge_index, self.edge_time, self.edge_attr

        iterables = [new_edge_index[0], new_edge_index[1], new_edge_time]
        if new_edge_attr is not None:
            iterables.append(new_edge_attr)

        for data in zip(*iterables):
            source_node, target_node, time_stamp = data[:3]
            edge_attr = data[3].item() if len(data) > 3 else None

            # Update the neighbor sequence
            if edge_attr is None:
                self.neighbor_sequence.setdefault(source_node.item(), []).append((
                    target_node.item(), time_stamp.item()))
            else:
                self.neighbor_sequence.setdefault(source_node.item(), []).append((
                    target_node.item(), time_stamp.item(), edge_attr))

            # Update the stream graph
            if source_node not in self.stream_graph:
                self.stream_graph[source_node] = OrderedDict()

            if time_stamp not in self.stream_graph[source_node]:
                # Insert the time_stamp in sorted order
                self.stream_graph[source_node][time_stamp] = []
                # Sort the keys (time_stamps) of the inner dictionary
                self.stream_graph[source_node] = OrderedDict(sorted(
                    self.stream_graph[source_node].items()))

            # Decide whether to add edge_attr or not
            if edge_attr is None:
                self.stream_graph[source_node][time_stamp].append(target_node)
            else:
                self.stream_graph[source_node][time_stamp].append((target_node, edge_attr))

        # Sort the neighbor sequence
        for source_node, hist in self.neighbor_sequence.items():
            sorted_hist = sorted(hist, key=lambda x: x[1])
            self.neighbor_sequence[source_node] = sorted_hist

    def update(self, new_edge=None, new_edge_index=None,
               new_edge_time=None, new_edge_attr=None):
        """
        Update the TemporalGraph with new edge_index, timestamps, and weights.
        The input can be either a combined new edge tensor or
            new edge_index, new_edge_time and new_edge_attr tensors.
        """
        if new_edge is not None:
            new_edge_index, new_edge_time, new_edge_attr = self.process_edge(new_edge)

        # Append the new_edge_index to the existing edge_index
        edge_index = torch.cat([self.edge_index, new_edge_index], dim=1)
        edge_time = torch.cat([self.edge_time, new_edge_time])
        edge_attr = torch.cat([self.edge_attr, new_edge_attr]) if new_edge_attr else None

        self.edge_index, self.edge_time, self.edge_attr = remove_duplicate_edges(
            edge_index, edge_time, edge_attr)
        self.update_further_graph(new_edge_index, new_edge_time, new_edge_attr)

    @property
    def max_time(self):
        return self.edge_time.max()

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    def _adaptive_load(self, filepath) -> Tensor:
        """
        Load .npy or .pt file based on its extension and return as Tensor.
        """
        _, file_extension = os.path.splitext(filepath)

        if file_extension == '.npy':
            return Tensor(np.load(filepath))
        elif file_extension == '.pt':
            return torch.load(filepath)
        elif file_extension == '.pkl':
            return Tensor(pd.read_pickle(filepath))
        else:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. "
                f"Supported extensions are .npy and .pt."
            )

    def _load_node_features(self, node_feat) -> Tensor:
        print('loading node feature data...')
        x = self._adaptive_load(node_feat) if isinstance(
            node_feat, (str, Path)) else Tensor(node_feat)
        x = Tensor(StandardScaler().fit_transform(x))
        return x

    def _load_edge(self, edge) -> (Tensor, Tensor, Tensor):
        """
        Load edge data from the given path. Depending on the
            number of columns in the file:
        - 1 and 2 columns: Assumes edge indices.
        - 3 columns: Assumes edge indices followed by timestamps.
        - 4 column and after: Assumes edge indices followed by weights or edge attributes.

        Args:
        - edge (str or tensor): Path or edge tensor to the file containing edge data.

        Returns:
        - edge_index (Tensor): Edge indices.
        - edge_time (Tensor): Timestamps for the edges.
            It must be present in the file.
        - edge_attr (Tensor or None): Weights for the edges.
            Returns None if not present in the file.
        """
        print('loading edge data...')
        edge_data = self._adaptive_load(edge) if isinstance(
            edge, (str, Path)) else Tensor(edge)
        if edge_data.shape[1] < 3:  # minimum columns required: source, target, time
            raise ValueError("The edge data must have at"
                             "least 3 columns: source, target, and time.")

        edge_index = edge_data[:, :2].t().long()

        # need to be long type for temporal linker exampler, maybe a bug of PyG
        edge_time = edge_data[:, 2].long()

        # Check for the weight column
        if edge_data.shape[1] > 3:
            edge_attr = edge_data[:, 3:].float()
        else:
            edge_attr = None
        return edge_index, edge_time, edge_attr


def remove_duplicate_edges(edge_index: Tensor, edge_time: Tensor,
                           edge_attr: Tensor = None) -> (Tensor, Tensor, Tensor):
    """
    Remove duplicate edges from the given
        edge_index, edge_time, and edge_attr.
    """
    # Create a unique hash for each edge based on its attributes
    # combine edge_index, edge_time and edge_attr
    combined_tensor = torch.cat([edge_index.t().float(), edge_time.unsqueeze(-1)], dim=1)
    if edge_attr is not None:
        # Ensure that the number of edges and edge attributes are the same
        assert edge_index.shape[1] == edge_attr.shape[0]
        combined_tensor = torch.cat([combined_tensor, edge_attr.unsqueeze(-1)], dim=1)

    # Get the hashes for each row
    hashes = [tensor_to_hashes(row) for row in combined_tensor]

    # Get the unique indices based on the hashes
    _, unique_indices = torch.unique(torch.tensor(hashes), return_inverse=True)

    # Update the edge attributes based on unique indices
    edge_index = edge_index[:, unique_indices]
    edge_time = edge_time[unique_indices]
    if edge_attr is not None:
        if edge_attr.dim() == 1:
            edge_attr = edge_attr[unique_indices]
        else:
            edge_attr = edge_attr[unique_indices, :]

    return edge_index, edge_time, edge_attr


def to_undirected(edge_index, edge_time, edge_attr=None):
    # Ensure that edge_time and edge_attr have the same length
    assert edge_index.shape[1] == edge_time.shape[0]
    assert edge_attr is None or edge_index.shape[1] == edge_attr.shape[0]

    # Create tensors for reversed edges directly during concatenation
    all_edge_index = torch.cat(
        [edge_index, torch.flip(edge_index, dims=[0])], dim=1)
    all_edge_time = torch.cat([edge_time, edge_time])
    if edge_attr is not None:
        all_edge_attr = torch.cat([edge_attr, edge_attr])

    # Remove duplicates
    return remove_duplicate_edges(
        all_edge_index, all_edge_time, all_edge_attr)


def tensor_to_hashes(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
