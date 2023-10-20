from pathlib import Path
import pandas as pd
import numpy as np
import torch


class WikiLoader:
    def __init__(self, params) -> None:
        self.params = params

    def process_data(self):
        """
        1. `x`: 
        - Shape: `(max(item_id) + 1, number_of_features)`
        - This tensor is initialized with zeros. Its length is determined by 
            the number of unique users and items in the dataset.
        
        2. `edge`: 
        - Shape: `(3, number_of_edges)`
        - This tensor represents the edges in the dataset.
            - Col 0: `user_id` for the edge.
            - Col 1: `item_id` for the edge (after renumbering).
            - Col 2: `timestamp` of the interaction.

        3. `edge_attr`: 
        - Shape: `(number_of_edges, number_of_features)`
        - The features are extracted from the `comma_separated_list_of_features` column.

        Returns:
        - Three tensor of (`x`, `edge`, and `edge_attr`).
        """

        # 1. load data
        data = pd.read_csv(Path(
            self.params['raw_data_path'],
            self.params['data_path']), sep=',',
            header=None, skiprows=1)

        # 2. Get the maximum value of user_id and renumber item_id
        max_user_id = data.iloc[:, 0].max()
        data.iloc[:, 1] = data.iloc[:, 1] + max_user_id + 1
        feature_df = data.iloc[:, 4:]

        # 3. x.pt
        x = torch.zeros((int(data.iloc[:, 1].max()) + 1, feature_df.shape[1]))

        # edge
        edge_values = np.array(data.iloc[:, :3].values)
        edge = torch.tensor(edge_values, dtype=torch.int64)

        # edge_attr
        features = np.array(feature_df, dtype=float).reshape(-1, feature_df.shape[1])
        edge_attr = torch.tensor(features, dtype=torch.float)

        return x, edge, edge_attr
