import numpy as np
import torch

from src.dataloader import BasicDataset
from src.similarity import UserSimilarity
from . import sampling

class ModelLoss:


    def __init__(self, device: torch.device, dataset: BasicDataset):
        self.device = device
        self.dataset = dataset
        pass

    def get_loss(self):
        pass

class SimilarityLoss(ModelLoss):
    """The similarity loss from DeepFD will be applied to LightGCN for embeddings.
       Similarities are only computed between each node and its samples to avoid computing all pairwise similarities.


        When fast_sampling is True, we assume the dataset has high number of negative nodes (sparse).
        Therefore, we assume that the a sample of all nodes is a sample of mostly negative nodes.
        We also allow duplicate negative nodes. 
        This means that some negative nodes will actually be positive nodes, so it may be good to increase n_neg.
    """

    def __init__(self, device: torch.device, dataset: BasicDataset, n_pos: int, n_neg: int, fast_sampling: bool = False):
        super().__init__(device, dataset)
        self.user_simi = UserSimilarity(dataset.graph_u2i)
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.fast_sampling = fast_sampling

    def get_loss(self, user_nodes, user_embs, samples):
        """ 
        Arguments:
            user_nodes: list of user node indices in this batch
            user_embs: tensor of all user embeddings
            samples: 2D numpy array of pos/neg samples of each node
                samples should be generated once per epoch by sample_train_set_pos_neg_users
        """
        # Get user embeddings
        user_embs_selected = user_embs[user_nodes]

        # Get sample embeddings for each user in this batch
        selected_samples = samples[user_nodes]
        sample_embs = user_embs[selected_samples.flatten()].view(*selected_samples.shape, -1)
        
        # Compute similarities between each user embedding and its samples embeddings
        dis_ij = (user_embs_selected[:, None, :] - sample_embs) ** 2
        simi_embs = torch.exp(-dis_ij.sum(dim=-1))

        # Calculate similarity from graph
        simi_feat_list = []
        for node_i in user_nodes:
            for sample_index in range(samples.shape[1]):
                node_j = samples[node_i, sample_index]
                simi_feat_list.append(self.user_simi.get_smoothed_jaccard_similarity(node_i, node_j))
        simi_feat = torch.FloatTensor(simi_feat_list).view(len(user_nodes), -1).to(self.device)

        # Compute loss
        L = simi_feat * ((simi_embs - simi_feat) ** 2)
        return L.mean()

    def sample_train_set_pos_neg_users(self):
        return sampling.sample_train_set_pos_neg_users(self.dataset, self.n_pos, self.n_neg, self.fast_sampling)

    @staticmethod
    def extend_user_node_batch(user_nodes, samples):
        """ Extends the batch of user nodes with their samples.
        """
        extended_nodes_batch = set(user_nodes)
        for node_i in user_nodes:
            extended_nodes_batch |= set(samples[node_i])
        
        return np.array(list(extended_nodes_batch))
    
class BPRLoss(ModelLoss):

    def __init__(self, device: torch.device, dataset: BasicDataset, weight_decay: float):
        super().__init__(device, dataset)
        self.weight_decay = weight_decay

    def get_loss(self, users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego):
        loss, reg_loss = self.bpr_loss(users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        return loss

    def bpr_loss(self, users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego):
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) + 
                         pos_emb_ego.norm(2).pow(2)  +
                         neg_emb_ego.norm(2).pow(2))/float(len(users_emb))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
