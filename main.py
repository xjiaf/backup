import random
import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.temporal_graph import TemporalGraph
from train_test import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(params):
    setup_seed(params['seed'])

    if params['model'] == 'dgdcn':
        train_model(params, train_loader, device=device)

    # save model and result
#     # store result embeddings
#     emb_dict = {}

#     for j in range(params['epoch_num']):
#         loader = DataLoader(Data, batch_size=params.batch_size, shuffle=True)
#         for i_batch, sample_batched in enumerate(loader):
#             # TODO: build model
#             loss, s_emb, _, _, _ = model.forward()
#             if j == 0:
#                 if i_batch % 100 == 0:
#                     print('batch_{} event_loss:'.format(i_batch), loss)

#             s_node_np = sample_batched['s_node'].cpu().numpy()
#             s_emb_np = s_emb.numpy()
#             e_time_np = sample_batched['event_time'].cpu().numpy()

#             update emb dict
#             for sn, et, se in zip(s_node_np.flatten(), e_time_np.flatten(), s_emb_np):
#                 emb_dict[(sn, et)] = se

#         print('ep_{}_event_loss:'.format(j + 1), loss)

#         # save as pkl
#         all_emb_df = pd.DataFrame([
#             {'s_node': k[0], 'e_time': k[1], 's_emb': v}
#             for k, v in emb_dict.items()
#         ])

#         all_emb_df.to_pickle(params.emb_save_path)
#         torch.save(model.state_dict(), params.model_save_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_params(args, config):
    # Merge default params with dataset-specific params
    params = {**config['default'], **config['datasets'][args.dataset]}

    # If model argument is provided, merge model-specific params
    if args.model in config['models']:
        params = {**params, **config['models'][args.model]}

    # Ensure paths are cross-platform compatible
    params['raw_data_path'] = Path(params['raw_data_path'])
    params['processed_data_path'] = Path(params['processed_data_path'])
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training and Testing TGSL.")
    parser.add_argument('--dataset', type=str, choices=['yelp', 'huawei'], default='yelp', help='Which dataset to use.')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'dev'], default='dev')
    parser.add_argument('--model', type=str, choices=['dgdcn', 'dgpredcn'], default='dgdcn', help='Which model to use.')

    args = parser.parse_args()

    config = load_config('config.yaml')
    params = get_params(args, config)

    main(params)