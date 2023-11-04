import torch
import logging
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from datasets.temporal_graph import TemporalGraph
from model.dgnn import DGNN
from utils.early_stopping import EarlyStopping
from modules.info_nce import BiInfoNCE

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True


class DGNNTrainer:
    def __init__(self, params, device=device):
        self.params = params
        # Load data
        origin_edge_path = (params['processed_data_path'] / params[
            'dataset'] / params['origin_edge_file'])
        x_path = (params['processed_data_path'] / params[
            'dataset'] / params['x_file'])
        self.graph = TemporalGraph(edge=origin_edge_path,
                                   x=x_path,
                                   directed=params['directed'])

        # Initialize
        self.model = self.init_model().to(device)
        self.optimizer = optim.Adam([{'params': self.model.parameters()}],
                                    lr=self.params['lr'],
                                    weight_decay=self.params['weight_decay'])
        self.node_emb_dict = {}
        self.writer = SummaryWriter(
            log_dir=self.params['result_path'] / self.params[
                'dataset'] / self.params['model'] / 'logs')
        self.criterion = BiInfoNCE(temperature=params['temperature'],
                                   reduction='mean',
                                   negative_mode='paired').to(device)

    def init_model(self):
        """Initialize the model"""
        logging.info('init model: DGNN')
        if self.params['model'] == 'dgnn':
            model = DGNN(self.params,
                         self.graph.num_node_features,
                         self.params['out_channels'])
        else:
            raise "Only train for DGNN"
        return model

    def get_train_loader(self):
        logging.info("getting LinkNeighborLoader")
        loader = LinkNeighborLoader(data=self.graph,
                                    edge_label_index=self.graph.edge_index,
                                    batch_size=self.params['batch_size'],
                                    num_neighbors=self.params['num_neighbors'],  # bug in PyG, must be -1 here if use temporal sampling
                                    neg_sampling=NegativeSampling(
                                        mode='triplet', amount=self.params[
                                            'neg_sampling_ratio']),
                                    shuffle=True,
                                    # replace=True,
                                    # temporal_strategy='uniform',
                                    # edge_label_time=self.graph.edge_time,
                                    # time_attr='edge_time',
                                    )
        return loader

    def train(self, epoch_num=None, is_early_stopping=True):
        if epoch_num is None:
            epoch_num = self.params['epoch_num']
        if is_early_stopping:
            ckpt_path = (self.params['result_path'] / self.params[
                'dataset'] / self.params['model'] / 'checkpoint.pt')
            early_stopping = EarlyStopping(patience=self.params[
                'patience'], verbose=True, path=ckpt_path)

        logging.info("---------start training----------")
        train_loader = self.get_train_loader()
        for epoch_idx in range(epoch_num):
            self.model.train()
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                train_loss = self.model_forward(batch)
                self.optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=5, norm_type=2.0)
                self.optimizer.step()
                if epoch_idx == 0:
                    if batch_idx % 10 == 0:
                        logging.info('batch_{0} train loss: {1}'.format(
                            batch_idx, train_loss.item()))
                del batch
            else:
                # Epoch finished and evaluate the model
                logging.info('batch_{0} train loss: {1}'.format(
                    batch_idx, train_loss.item()))
                self.writer.add_scalar('Loss/{0}'.format(
                    'train'), train_loss.item(), epoch_idx + 1)
                if is_early_stopping:
                    early_stopping(train_loss, self.model)
                    if early_stopping.early_stop:
                        logging.info("Early stopping")
                        break
            torch.cuda.empty_cache()
        else:
            logging.info("---------all epochs finished----------")
        self.writer.close()
        logging.info("---------finish training----------")

    def model_forward(self, batch):
        if self.params['model'] == 'dgnn':
            node_time = self.graph.edge_time[batch.input_id.cpu().detach()]
            node_time = node_time.to(device)
            node_emb = self.model(x=batch.x,
                                  edge_index=batch.edge_index,
                                  edge_time=batch.edge_time,
                                  node_time=node_time,
                                  edge_weight=batch.edge_weight)

            src_emb_list, dst_pos_emb_list, dst_neg_emb_list = [], [], []
            for ts_index, ts in enumerate(node_time):
                # Get the node id of the target node
                node_id = batch.n_id[batch.dst_pos_index[ts_index]]

                # Get the embedding of the target node
                emb = node_emb[ts_index, batch.dst_pos_index[ts_index]]

                # Get the embedding of the source node and negative nodes
                src_emb_list.append(
                    node_emb[ts_index, batch.src_index[ts_index]])
                dst_neg_emb_list.append(
                    node_emb[ts_index, batch.dst_neg_index[ts_index]])
                dst_pos_emb_list.append(emb)

                # Save the embedding of the target node
                self.node_emb_dict[(node_id.item(), ts.item())] = emb
            else:
                src_emb = torch.stack(src_emb_list, dim=0)
                dst_pos_emb = torch.stack(dst_pos_emb_list, dim=0)
                dst_neg_emb = torch.stack(dst_neg_emb_list, dim=0)

            loss = self.criterion(src_emb, dst_pos_emb, dst_neg_emb)

        return loss
