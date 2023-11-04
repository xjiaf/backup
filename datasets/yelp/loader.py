import pickle
from pathlib import Path
import pandas as pd
from pandas import DataFrame as DF

import torch
from sklearn.model_selection import train_test_split


class YelpLoader:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.save_path = Path(params['processed_data_path'], params['dataset'])
        self.save_path.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> (DF, DF, DF):
        raw_file = [self.params['review_file'], self.params['item_file'], self.params['user_file']]
        df_list = []
        for i in range(3):
            print("Loading {}".format(raw_file[i]))

            raw_path = Path(self.params['raw_data_path'], raw_file[i])
            size = 500000
            data = pd.read_json(raw_path, lines=True, chunksize=size)

            chunk_list = []
            for chunk in data:
                chunk_list.append(chunk)
            else:
                df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)
            df_list.append(df)

        # raw review_df, business_df, user_df
        return df_list[0], df_list[1], df_list[2]

    def preprocess_data(self):
        review_df, item_df, user_df = self.load_data()
        # Filter reviews that are only for businesses present in item_df
        valid_business_ids = item_df['business_id'].unique()
        review_df = review_df[review_df['business_id'].isin(valid_business_ids)][['user_id', 'business_id', 'date', 'stars']]
        review_df.rename(columns={'stars': 'label'}, inplace=True)

        # if stars > 0 the label is 1 else 0
        review_df['label'] = review_df['label'].apply(lambda x: 1 if x >= 4 else 0)

        # Create a mapping from user_id to a unique index starting from 0y
        user_map = review_df[['user_id']].drop_duplicates().reset_index(drop=True)
        user_map['user_index'] = user_map.index
        user_num_col = ["user_id", "user_index"] + list(self.params['sparse_features'])
        user_df = user_df[user_df['user_id'].isin(user_map['user_id'])].merge(user_map, on='user_id')[user_num_col]

        # Simplify item_df and add a unique index for each business
        item_df = item_df[['business_id', 'latitude', 'longitude', 'stars', 'review_count', 'is_open']]
        item_df.rename(columns={'review_count': 'business_review_count'}, inplace=True)
        item_df = item_df.fillna(-1)
        item_df['business_index'] = item_df.index

        # Generate item node features df, which will be saved as tensor
        self.node_feat = item_df[self.params['dense_features']]

        # Merge review_df with user_df and item_df to replace user_id and business_id with their respective indices
        review_df = review_df.merge(user_df, on='user_id').merge(item_df, on='business_id')
        review_df = review_df.drop(columns=["business_id", "user_id"])

        # Convert date column to datetime format and sort it
        review_df['date'] = pd.to_datetime(review_df['date'], format='%Y-%m-%d %H:%M:%S')
        review_df = review_df.sort_values(by=['user_index', 'date'])

        # Create sequence columns
        review_df['sequence'] = review_df.groupby('user_index')['business_index'].transform(accumulate_sequence)

        # Create 'last_five_business' column based on 'sequence', if the sequence is less than 5, fill with -1
        review_df['last_five_business'] = review_df['sequence'].apply(lambda x: (x[-6:-1] + [-1]*5)[:5])

        # Create 'user_time'
        review_df['user_time'] = review_df['sequence'].apply(len)

        self.review_df = review_df.sort_values(by='date', ascending=True)

    def construct_dataset(self):
        # Split train and test
        train, test = train_test_split(
            self.review_df, test_size=0.15, shuffle=False)

        # Group by 'user_index' and get the last entry for each user
        self.seq_df_train = train.sort_values('date').groupby(
            'user_index').tail(1)[['user_index', 'sequence']].reset_index(drop=True)

        # Adjust the labels based on the number of bins
        business_freq = train['business_index'].value_counts()
        quartiles = business_freq.quantile([0.25, 0.5, 0.75, 1.0])

        # Define the bins based on quartiles
        bins = [0] + quartiles.tolist()
        unique_bins = sorted(list(set(bins)))

        # Adjust the labels based on the number of unique bins
        adjusted_labels = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"][:len(unique_bins)-1]

        # Use cut to determine the quartile bin for each business_index frequency
        quartile_bins = pd.cut(business_freq, bins=unique_bins, labels=adjusted_labels, right=False, include_lowest=True)

        # Extract business_index values for each quartile range
        self.quartile_dict = {}
        for label in quartile_bins.cat.categories:
            self.quartile_dict[label] = business_freq[quartile_bins == label].index.tolist()

        # build the train and test dataset
        dataset_col = self.params['dense_features'] + self.params['sparse_features'] + ['last_five_business', 'user_time', 'label']
        self.train_dataset = train[dataset_col]
        self.test_dataset = test[dataset_col]

    def seve_data(self):
        # Save quartile_dict
        with open(Path(self.save_path, self.params['quartile_dict_file']), 'wb') as file:
            pickle.dump(self.quartile_dict, file)

        # Save train and test dataset
        torch.save(self.train_dataset, Path(self.save_path, self.params['train_dataset_file']))
        torch.save(self.test_dataset, Path(self.save_path, self.params['test_dataset_file']))

        # save node feature
        torch.save(self.node_feat, Path(self.save_path, self.params['x_file']))

        # save sequence
        self.seq_df_train.to_pickle(Path(self.save_path, self.params['user_seq_train_file']))

    def process_data(self):
        self.preprocess_data()
        self.construct_dataset()
        self.seve_data()


# Create a helper function to generate sequence
def accumulate_sequence(series):
    accumulated = []
    current_sequence = []
    for val in series:
        current_sequence.append(val)
        accumulated.append(list(current_sequence))
    return accumulated