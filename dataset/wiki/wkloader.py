from pathlib import Path
import pandas as pd
import torch


class WikiLoader:
    def __init__(self, params) -> None:
        self.params = params
        self.save_path = Path(params['processed_data_path'], params['dataset'])
    
    def process_data(self):
        # 1. load data
        data = pd.read_csv(Path(self.params['raw_data_path'], 'wikipedia.csv'))

        # 2. Get the maximum value of user_id and renumber item_id
        max_user_id = data["user_id"].max()
        data["item_id"] = data["item_id"] + max_user_id + 1

        # 3. Create tensors based on the specified conditions
        # Create x.pt, length is the maximum value of item_id + 1
        x = torch.zeros((data["item_id"].max() + 1, len(data.filter(like="comma_separated_list_of_features").iloc[0])))

        # edge.pt
        edge = torch.tensor([data["user_id"].values, data["item_id"].values, data["timestamp"].values.astype('int64')], dtype=torch.int64)

        # edge_attr.pt
        features = data.filter(like="comma_separated_list_of_features").values.tolist()
        edge_attr = torch.tensor(features, dtype=torch.float)

        # 4. Save the created tensors as `.pt` files
        torch.save(x, "x.pt")
        torch.save(edge, "edge.pt")
        torch.save(edge_attr, "edge_attr.pt")
