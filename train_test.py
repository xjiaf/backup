import torch
from torch import nn, optim
from torch_geometric.loader import LinkLoader, LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from model.dgnn import DGNN
from dataset.temporal_graph import TemporalGraph
from utils.early_stopping import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dataset:
    def __init__(self, params):
        self.params = params
        # Load graph
        origin_edge_path = (params['processed_data_path'] / params[
            'dataset'] / params['origin_edge_file'])
        x_path = (params['processed_data_path'] / params[
            'dataset'] / params['x_file'])
        self.graph = TemporalGraph(edge=origin_edge_path,
                                   x=x_path, directed=params['directed'])

        self.train, self.test = self.train_test_split()

        # for induction


    def train_test_split(self):
        pass

    def get_train_loader(self):
        print("get dataloader:", self.params['loader'])
        if self.params['model'] == 'dgnn':
            if self.params['loader'] == 'LinkNeighborLoader':
                loader = LinkNeighborLoader(data=self.train,
                                            edge_label_index=self.train.edge_index,
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
                                            edge_label_time=self.train.edge_time,
                                            time_attr='edge_time')

            elif self.params['loader'] == 'LinkLoader':
                loader = LinkLoader(data=self.train,
                                    shuffle=self.params['shuffle'],
                                    neg_sampling=NegativeSampling(
                                        mode='triplet'),
                                    batch_size=self.params['batch_size'],
                                    neg_sampling_ratio=self.params[
                                        'neg_sampling_ratio'],
                                    num_workers=self.params['num_workers'],
                                    edge_label_time=self.train.edge_time,
                                    time_attr='edge_time')
        return loader

    def get_test_loader(self):
        pass


class Trainer:
    def __init__(self, params: dict, data: Dataset, device=device):
        self.params = params
        # Initialize
        self.data = data
        self.graph = data.train
        self.model = self.init_model().to(device)
        self.optimizer = optim.Adam([{'params': self.model.parameters()}],
                                    lr=params['lr'],
                                    weight_decay=params['weight_decay'])
        self.criterion = self.model.criterion.to(device)
        self.model.train()

    def init_model(self):
        """Initialize the model"""
        print('init model:', self.params['model'])
        if self.params['model'] == 'dgnn':
            model = model = DGNN(self.params,
                                 self.graph.num_node_features,
                                 self.params['out_channels'])

        # if pytorch version >= 2.0.0 and cuda is available
        if (int(torch.__version__.split('.')[0]) >= 2) and torch.cuda.is_available():
            model = torch.compile(model)  # pytorch compile to accelerate
        return model

    def train(self, epoch_num=None, is_early_stopping=True):
        loader = self.data.get_train_loader()
        if epoch_num is None:
            epoch_num = self.params['epoch_num']
        if is_early_stopping:
            ckpt_path = (self.params['result_path'] / self.params[
                'dataset'] / self.params['model'] / 'checkpoint.pt')
            early_stopping = EarlyStopping(patience=self.params[
                'patience'], verbose=True, path=ckpt_path)

        if self.params['model'] == 'dgnn':
            # self.node_emb = torch.zeros_like((self.graph.num_nodes,
            #                                   self.params['out_channels']))
            self.node_emb_dict = {}
            # self.node_emb = self.node_emb.to(device)
        print("start training")
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
            node_emb = self.model(batch.x,
                                  batch.edge_index,
                                  batch.edge_time,
                                  batch.edge_time,
                                  batch.edge_weight)
            for sn, et, se in zip(batch.n_id.flatten(), batch.edge_time.flatten(), node_emb):
                self.node_emb_dict[(sn, et)] = se

            loss = self.criterion(self.node_emb[batch.src_index],
                                  self.node_emb[batch.dst_pos_index],
                                  self.node_emb[batch.dst_neg_index])
        return loss


class Tester:
    def __init__(self, params: dict, data: Dataset, device=device):
        self.params = params

        self.data = data
        self.graph = data.train
        self.model = self.init_model().to(device)
        self.optimizer = optim.Adam([{'params': self.model.parameters()}],
                                    lr=params['lr'],
                                    weight_decay=params['weight_decay'])
        self.criterion = self.model.criterion.to(device)
        self.model.eval()

    def test(self):
        with torch.no_grad():
            pass
