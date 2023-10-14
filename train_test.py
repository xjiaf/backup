trainer怎么样
import torch
from torch import nn, optim
from torch_geometric.loader import LinkLoader, LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from model.dgdcn import DGDCN
from model.dgnn import DGNN
from datasets.temporal_graph import TemporalGraph
from utils.early_stopping import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, params, device=device):
        self.params = params
        # Load graph
        origin_edge_path = (params['processed_data_path'] / params[
            'dataset'] / params['origin_edge_file'])
        x_path = (params['processed_data_path'] / params[
            'dataset'] / params['x_file'])
        self.graph = TemporalGraph(edge=origin_edge_path,
                                    x=x_path, directed=params['directed'])

        # Initialize
        self.model = self.init_model().to(device)
        self.optimizer = optim.Adam([{'params': self.model.parameters()}],
                                    lr=params['lr'],
                                    weight_decay=params[
                                        'weight_decay']).to(device)
        self.criterion = self.model.criterion.to(device)

        self.model.train()

    def init_model(self):
        """Initialize the model"""
        if self.params['model'] == 'dgdcn':
            linear_feature_columns = self.params['linear_feature_columns']
            dnn_feature_columns = self.params['dnn_feature_columns']
            model = DGDCN(self.params, self.graph.num_node_features,
                          linear_feature_columns, dnn_feature_columns)
        elif self.params['model'] == 'dgnn':
            model = model = DGNN(self.params,
                                 self.graph.num_node_features,
                                 self.params['out_channels'])

        # if pytorch version >= 2.0.0 and cuda is available
        if (int(torch.__version__.split('.')[0]) >= 2) and torch.cuda.is_available():
            model = torch.compile(model)  # pytorch compile to accelerate
        return model

    def get_train_loader(self):
        if self.params['model'] == 'dgnn':
            if self.params['loader'] == 'LinkNeighborLoader':
                loader = LinkNeighborLoader(data=self.graph,
                                            batch_size=self.params['batch_size'],
                                            num_neighbors=self.params[
                                                'num_neighbors'],
                                            neg_sampling=NegativeSampling(
                                                mode='triplet'),
                                            temporal_strategy='uniform',
                                            shuffle=self.params['shuffle'],
                                            neg_sampling_ratio=self.params[
                                                'neg_sampling_ratio'],
                                            num_workers=self.params['num_workers'],
                                            edge_label_time=self.graph.edge_time,
                                            time_attr='edge_time',
                                            weight_attr='edge_weight')

            elif self.params['loader'] == 'LinkLoader':
                loader = LinkLoader(data=self.graph,
                                    shuffle=self.params['shuffle'],
                                    neg_sampling=NegativeSampling(
                                        mode='triplet'),
                                    batch_size=self.params['batch_size'],
                                    neg_sampling_ratio=self.params[
                                        'neg_sampling_ratio'],
                                    num_workers=self.params['num_workers'],
                                    edge_label_time=self.graph.edge_time,
                                    time_attr='edge_time')
        return loader

    def train(self, epoch_num=None, is_early_stopping=True):
        loader = self.get_train_loader()
        if epoch_num is None:
            epoch_num = self.params['epoch_num']
        if is_early_stopping:
            ckpt_path = (self.params['result_path'] / self.params[
                'dataset'] / self.params['model'] / 'checkpoint.pt')
            early_stopping = EarlyStopping(patience=self.params[
                'patience'], verbose=True, path=ckpt_path)

        if self.params['model'] == 'dgnn':
            self.node_emb = torch.zeros_like((self.graph.num_nodes,
                                              self.params['out_channels']))
        for epoch_idx in range(epoch_num):
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(device)
                loss = self.model_forward(batch)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(parameters=self.model.parameters(),
                                        max_norm=5, norm_type=2.0)
                self.optimizer.step()
                if epoch_idx == 0:
                    if batch_idx % 10 == 0:
                        print('batch_{} event_loss:'.format(batch_idx), loss)
            else:
                print('epoch_{}_loss:'.format(epoch_idx + 1), loss)

                if is_early_stopping:
                    early_stopping(loss, self.model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

        if self.params['model'] == 'dgnn':
            return self.node_emb.cpu().detach().clone(), loss.cpu().detach().clone()

    def model_forward(self, batch):
        batch = batch.to(device)
        if self.params['model'] == 'dgnn':
            self.node_emb[batch.n_id] = self.model(batch.x,
                                                   batch.edge_index,
                                                   batch.edge_time,
                                                   batch.node_time,
                                                   batch.edge_weight)
            loss = self.criterion(self.node_emb[batch.src_index],
                                  self.node_emb[batch.dst_pos_index],
                                  self.node_emb[batch.dst_neg_index])
        return loss


class Tester:
    def __init__(self, params, device=device):
        pass
        # model.eval()
        with torch.no_grad():
            pass
