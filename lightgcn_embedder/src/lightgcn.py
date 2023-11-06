import torch
from torch import nn
from dataclasses import dataclass
from dataloader import BasicDataset
import numpy as np
import copy

@dataclass
class LightGCNConfig:
    latent_dim: int
    n_layers: int
    keep_prob: float
    A_split: int
    device: torch.device
    train_config: LightGCNTrainConfig

@dataclass
class LightGCNTrainingConfig:
    dropout: bool = True
    lr: float = 0.001
    batch_size: int

    pretrained: bool = False
    user_emb = 0
    item_emb = 0



class LightGCN(nn.Module):
    def __init__(self, config: LightGCNConfig, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
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
        if self.config.pretrain == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print("use NORMAL distribution initializer")
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config.user_emb)
            )
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config.item_emb)
            )
            print("use pretrained data")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config.dropout})")

        # print("save_txt")

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
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config.dropout:
            if self.training:
                print("droping")
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
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
            )
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        return users_emb, items_emb


class Loss_DeepFD:
    """A DeepFD style loss will be applied to LightGCN for embeddings."""

    def __init__(self, config: LightGCNConfig, dataset: BasicDataset):
        self.config = config
        self.dataset = dataset
        pass

    def __init__(self, features, graph_simi, device, alpha, beta, gamma):
        self.features = features
        self.graph_simi = graph_simi
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.node_pairs = {}
        self.original_nodes_batch = None
        self.extended_nodes_batch = None

    def extend_nodes(self, nodes_batch, training_cps):
        self.original_nodes_batch = copy.deepcopy(nodes_batch)
        self.node_pairs = {}
        self.extended_nodes_batch = set(nodes_batch)

        for node in nodes_batch:
            cps = training_cps[node]
            self.node_pairs[node] = cps
            for cp in cps:
                self.extended_nodes_batch.add(cp[1])
        self.extended_nodes_batch = list(self.extended_nodes_batch)
        return self.extended_nodes_batch

    def get_loss(self, nodes_batch, embs_batch, recon_batch):
        # calculate loss_simi and loss+recon,
        # loss_reg is included in SGD optimizer as weight_decay
        loss_recon = self.get_loss_recon(nodes_batch, recon_batch)
        loss_simi = self.get_loss_simi(embs_batch)
        loss = loss_recon + self.alpha * loss_simi
        return loss

    def get_loss_simi(self, embs_batch):
        node2index = {n: i for i, n in enumerate(self.extended_nodes_batch)}
        simi_feat = []
        simi_embs = []
        for node, cps in self.node_pairs.items():
            for i, j in cps:
                simi_feat.append(torch.FloatTensor([self.graph_simi[i, j]]))
                dis_ij = (embs_batch[node2index[i]] - embs_batch[node2index[j]]) ** 2
                dis_ij = torch.exp(-dis_ij.sum())
                simi_embs.append(dis_ij.view(1))
        simi_feat = torch.cat(simi_feat, 0).to(self.device)
        simi_embs = torch.cat(simi_embs, 0)
        L = simi_feat * ((simi_embs - simi_feat) ** 2)
        return L.mean()

    def get_loss_recon(self, nodes_batch, recon_batch):
        feats_batch = self.features[nodes_batch]
        H_batch = (feats_batch * (self.beta - 1)) + 1
        assert feats_batch.size() == recon_batch.size() == H_batch.size()
        L = ((recon_batch - feats_batch) * H_batch) ** 2
        return L.mean()
