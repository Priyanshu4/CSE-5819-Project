from lightgcn import LightGCN, LightGCNConfig
from dataloader import BasicDataset
import torch
from similarity import GraphSimilarity

class Loss_DeepFD:
    """The loss from DeepFD will be applied to LightGCN for embeddings."""

    def __init__(self, dataset: BasicDataset, graph_simi: GraphSimilarity, device, alpha, beta, gamma):
        self.dataset = dataset
        self.graph_simi = graph_simi
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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
        feats_batch = self.dataset.graph_u2i[nodes_batch]
        H_batch = (feats_batch * (self.beta - 1)) + 1
        assert feats_batch.size() == recon_batch.size() == H_batch.size()
        L = ((recon_batch - feats_batch) * H_batch) ** 2
        return L.mean()
