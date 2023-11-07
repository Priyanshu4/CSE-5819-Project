from lightgcn import LightGCN, LightGCNConfig
from numpy import np
from dataloader import BasicDataset
import torch
from similarity import GraphSimilarity

class ModelLoss:


    def __init__(self, dataset: BasicDataset):
        self.dataset = dataset
        pass

    def get_loss(self, user_nodes, item_nodes, user_embs, item_embs):
        pass

class SimilarityLoss(ModelLoss):
    """The similarity loss from DeepFD will be applied to LightGCN for embeddings.
       This class also contains the code for sampling the positive and negative users for each user node.
       Similarities are only computed between each node and its samples to avoid computing all pairwise similarities.
       Before each epoch, the new_epoch function must be called which calls the sampling functions.
    """

    def __init__(self, dataset: BasicDataset, graph_simi: GraphSimilarity, n_pos: int, n_neg: int, fast_sampling: bool = False):
        super().__init__(dataset)
        self.graph_simi = graph_simi
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.fast_sampling = fast_sampling
        self.samples = None

    def new_epoch(self):
        """ Generates positive and negative samples at the start of each epoch.
            See SimilarityLoss.sample_train_set for more information.
        """
        self.samples = SimilarityLoss.sample_train_set(self.dataset, self.n_pos, self.n_neg, fast=self.fast_sampling)

    def get_loss(self, user_nodes, item_nodes, user_embs, item_embs):
        if not self.samples:
            raise RuntimeError("new_epoch() must be called before using get_loss()!")

        node2index = {n: i for i, n in enumerate(self.extended_nodes_batch)}
        simi_feat = []
        simi_embs = []
        for node_i in user_nodes:
            for sample_index in self.samples.shape[1]:
                node_j = self.samples[node_i, sample_index]
                simi_feat.append(torch.FloatTensor([self.graph_simi[node_i, node_j]]))
                dis_ij = (user_embs[node2index[node_i]] - user_embs[node2index[node_j]]) ** 2
                dis_ij = torch.exp(-dis_ij.sum())
                simi_embs.append(dis_ij.view(1))
        simi_feat = torch.cat(simi_feat, 0).to(self.device)
        simi_embs = torch.cat(simi_embs, 0)
        L = simi_feat * ((simi_embs - simi_feat) ** 2)
        return L.mean()

    @staticmethod
    def sample_train_set(dataset: BasicDataset, n_pos: int, n_neg: int, fast: bool = False):
        """ For each user node in the dataset, this samples n_pos positive nodes and n_neg negative nodes.
            A positive node shares an item and a negative node does not share an item.
            This returns a 2D numpy array of shape (n_users, n_pos+n_neg) with the indices of samples for each user node.
            samples[i, :] gives the positive and negative samples for the ith node in the dataset,
            If there are not enough negative or positive samples, empty spots are filled with completely random nodes.

            When fast is True, we assume the dataset has high number of negative nodes (sparse).
            Therefore, we assume that the a sample of all nodes is a sample of mostly negative nodes.
        """
        if fast:
            return SimilarityLoss._sample_train_set_fast(dataset, n_pos, n_neg)
        return SimilarityLoss._sample_train_set_normal(dataset, n_pos, n_neg)

    @staticmethod
    def _sample_train_set_normal(dataset, n_pos, n_neg):
        g_u2u = dataset.graph_u2u 
        samples = np.zeros((dataset.n_users, n_pos + n_neg), dtype=int)
        all_indices = np.arange(dataset.n_users)
        
        for i in range(dataset.n_users):
            pos_pool = g_u2u[i].nonzero()[1]                # indices of all positive nodes for user i
            neg_pool = np.setdiff1d(all_indices, pos_pool)  # indices of all negative nodes for user i

            if len(pos_pool) >= n_pos:
                samples[i, :n_pos] = np.random.choice(pos_pool, n_pos, replace=False)
            else:
                samples[i, :len(pos_pool)] = pos_pool
                samples[i, len(pos_pool):n_pos] = np.random.choice(all_indices, n_pos - len(pos_pool), replace=False)
            
            if len(neg_pool) >= n_neg:
                samples[i, n_pos:] = np.random.choice(neg_pool, n_neg, replace=False)
            else:
                samples[i, n_pos:n_pos + len(neg_pool)] = neg_pool
                samples[i, n_pos+len(neg_pool):] = np.random.choice(all_indices, n_neg - len(neg_pool), replace=False)

        return samples

    @staticmethod
    def _sample_train_set_fast(dataset, n_pos, n_neg):
        g_u2u = dataset.graph_u2u 
        samples = np.zeros((dataset.n_users, n_pos + n_neg), dtype=int)
        all_indices = np.arange(dataset.n_users)
        
        for i in range(dataset.n_users):
            pos_pool = g_u2u[i].nonzero()[1]                # indices of all positive nodes for user i

            if len(pos_pool) >= n_pos:
                samples[i, :n_pos] = np.random.choice(pos_pool, n_pos, replace=False)
            else:
                samples[i, :len(pos_pool)] = pos_pool
                samples[i, len(pos_pool):n_pos] = np.random.choice(all_indices, n_pos - len(pos_pool), replace=False)
            
            # Here we make the simplifying assumption that the majority of nodes with be negatives.
            # Therefore, sampling from all nodes wil give us mostly negatives and is good enough.
            samples[i, n_pos:] = np.random.choice(all_indices, n_neg, replace=False)

        return samples


class BPRLoss(ModelLoss):

    def get_loss(self):
        loss, reg_loss = self.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        return loss

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
