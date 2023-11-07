from lightgcn import LightGCN, LightGCNConfig
from dataloader import BasicDataset
import torch
from similarity import GraphSimilarity

class SimilarityLoss:
    """The similarity loss from DeepFD will be applied to LightGCN for embeddings.
    """

    def __init__(self, dataset: BasicDataset, graph_simi: GraphSimilarity):
        self.dataset = dataset
        self.graph_simi = graph_simi

    def get_loss(self, nodes_batch, embs_batch, samples):
        """ Computes similarity loss
            loss_reg is included in optimizer as weight decay
            nodes_batch is the indices of all nodes (and their pos/neg samples) included in this batch
            embs_batch is the corresponding embeddings of these nodes
            samples is the numpy array of samples for all nodes in the training set
        """
        node2index = {n: i for i, n in enumerate(self.extended_nodes_batch)}
        simi_feat = []
        simi_embs = []
        for node_i in nodes_batch:
            for sample_index in samples.shape[1]:
                node_j = samples[node_i, sample_index]
                simi_feat.append(torch.FloatTensor([self.graph_simi[node_i, node_j]]))
                dis_ij = (embs_batch[node2index[i]] - embs_batch[node2index[node_j]]) ** 2
                dis_ij = torch.exp(-dis_ij.sum())
                simi_embs.append(dis_ij.view(1))
        simi_feat = torch.cat(simi_feat, 0).to(self.device)
        simi_embs = torch.cat(simi_embs, 0)
        L = simi_feat * ((simi_embs - simi_feat) ** 2)
        return L.mean()
