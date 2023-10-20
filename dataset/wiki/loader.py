from pathlib import Path
import pandas as pd
import numpy as np
import torch


class WikiLoader:
    """
    This class is used to load the Wiki dataset.
    """
    def __init__(self, params) -> None:
        self.params = params

    def process_data(self):
        """
        Returns:
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
            - Col 4 and after: `comma_separated_list_of_features` for the edge.
        """

        # 1. load data
        data = pd.read_csv(Path(
            self.params['raw_data_path'],
            self.params['data_path']), sep=',',
            header=None, skiprows=1)
        data.drop(data.columns[3], axis=1, inplace=True)

        # 2. Get the maximum value of user_id and renumber item_id
        max_user_id = data.iloc[:, 0].max()
        data.iloc[:, 1] = data.iloc[:, 1] + max_user_id + 1
        feature_df = data.iloc[:, 3:]
        # 3. node feature
        x = torch.zeros((int(data.iloc[:, 1].max()) + 1, feature_df.shape[1]))

        # 4. edge and edge feature
        edge_index = data.iloc[:, :3].to_numpy(np.int64)
        features = feature_df.to_numpy(dtype=float)
        edge = torch.from_numpy(np.hstack((edge_index, features)))

        return x, edge
