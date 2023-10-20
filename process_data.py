import yaml
import argparse
from pathlib import Path
import torch

from dataset.wiki.loader import WikiLoader


def process_data(params: dict):
    edge_file = Path(
        params['processed_data_path'], params['dataset'], params['edge_path'])
    x_file = Path(
        params['processed_data_path'], params['dataset'], params['x_path'])
    edge_attr_file = Path(
        params['processed_data_path'], params['dataset'], params['edge_attr_path'])
    edge_file.parent.mkdir(parents=True, exist_ok=True)
    # Load user behavior seq and construct graph
    # if edge_file.exists() and x_file.exists() and edge_attr_file.exists():
    #     print("{0} graph already exists, skip".format(params['dataset']))
    # else:
    print("start to process {0} data and generate graph".format(
        params['dataset']))

    if params['dataset'] == 'wiki':
        loader = WikiLoader(params=params)

    x, edge, edge_attr = loader.process_data()
    print("finished {0} graph constructiong".format(params['dataset']))
    # 4. Save the created tensors as `.pt` files
    torch.save(x, x_file)
    torch.save(edge, edge_file)
    torch.save(edge_attr, edge_attr_file)
    print("finished saving {0} graph".format(params['dataset']))


def main(params: dict):
    process_data(params)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_params(args, config):
    # Merge default params with dataset-specific params
    params = {**config['default'], **config['datasets'][args.dataset]}

    # Ensure paths are cross-platform compatible
    params['raw_data_path'] = Path(params['raw_data_path'])
    params['processed_data_path'] = Path(params['processed_data_path'])
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construct datasets")
    parser.add_argument('--dataset', type=str, choices=['yelp', 'wiki'],
                        default='wiki', help='Which dataset to use.')
    args = parser.parse_args()

    config = load_config('config.yaml')
    params = get_params(args, config)

    main(params)
