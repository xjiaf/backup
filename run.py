import random
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch

from train_test import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(params):
    setup_seed(params['seed'])
    save_path = params['result_path'] / params['dataset'] / params['model']
    save_path.mkdir(parents=True, exist_ok=True)

    if params['mode'] == 'train':
        trainer = Trainer(params, device=device)
        trainer.train()

    if params['model'] == 'dgnn':
        emb_save_path = save_path / 'node_emb.pt'
        torch.save(trainer.node_emb, emb_save_path)

    model_save_path = save_path / 'model.pt'
    torch.save(trainer.model.state_dict(), model_save_path)


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
    params = {**params, **config['models'][args.model]}

    # Ensure paths are cross-platform compatible
    params['raw_data_path'] = Path(params['raw_data_path'])
    params['processed_data_path'] = Path(params['processed_data_path'])
    params['result_path'] = Path(params['result_path'])
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training and Testing TGSL.")
    parser.add_argument('--dataset', type=str, choices=['yelp', 'wiki'],
                        default='yelp', help='Which dataset to use.')
    parser.add_argument('--mode', type=str,
                        choices=['train', 'test'], default='train')
    parser.add_argument('--model', type=str, choices=['dgnn', 'dgdcn'],
                        default='dgnn', help='Which model to use.')

    args = parser.parse_args()

    config = load_config('config.yaml')
    params = get_params(args, config)

    main(params)
