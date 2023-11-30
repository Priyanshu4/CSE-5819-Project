import torch
from torch import nn
from dataclasses import dataclass
import numpy as np
from typing import Optional
import logging

from src.dataloader import BasicDataset
import src.utils as utils

@dataclass
class LightGCNTrainingConfig:
    epochs: int = 10
    dropout: bool = True
    learning_rate: float = 0.001
    batch_size: int = 2048
    weight_decay: float = 1e-4
    n_pos_samples: int = 10
    n_neg_samples: int = 10

    # If pretrained, set True and pass pretrained embeddings
    pretrained: bool = False
    user_emb: Optional[np.ndarray] = None
    item_emb: Optional[np.ndarray] = None

@dataclass
class LightGCNConfig:
    latent_dim: int
    n_layers: int
    keep_prob: float
    A_split: int
    device: torch.device
    train_config: LightGCNTrainingConfig


class LightGCN(nn.Module):
    def __init__(self, config: LightGCNConfig, dataset: BasicDataset, logger: logging.Logger):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight(logger)
        self.to(self.config.device)

    def __init_weight(self, logger: logging.Logger):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config.latent_dim
        self.n_layers = self.config.n_layers
        self.keep_prob = self.config.keep_prob
        self.A_split = self.config.A_split
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        if self.config.train_config.pretrained == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            logger.info("Initializing weights with normal distribution")
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config.train_config.user_emb)
            )
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config.train_config.item_emb)
            )
            logger.info("Using pretrained weights")
        self.f = nn.Sigmoid()

        if self.config.A_split:
            # Get an adjacency matrix split into a list of folds and convert to tensors
            self.Graph = self.dataset.get_adj_mat_split(self.A_split)
            for i, fold in enumerate(self.Graph):
                self.Graph[i] = utils.sparse_matrix_to_tensor(fold).to(self.config.device)
            logger.info(f"Split adjacency matrix in {self.A_split} folds.")
        else:
            self.Graph = utils.sparse_matrix_to_tensor(self.dataset.graph_adj_mat).to(self.config.device)

        logger.info(f"LightGCN is ready to go! (dropout:{self.config.train_config.dropout})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config.train_config.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbeddingsForBPR(self, users, pos_items, neg_items):
        """ For BPR Loss computation
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def forward(self, users = None, items = None):
        # compute embedding
        all_users, all_items = self.computer()
        if users is None:
            users_emb = all_users
        else:
            users_emb = all_users[users] 
        if items is None:
            items_emb = all_items
        else:
            items_emb = all_items[items]
        return users_emb, items_emb