import torch
import numpy as np
import math
from loss import ModelLoss, SimilarityLoss, BPRLoss
from dataloader import BasicDataset
from lightgcn import LightGCN

def lightgcn_training_loop(dataset: BasicDataset, model: LightGCN, model_loss: ModelLoss, optimizer: torch.optim.Optimizer, epochs: int, logger):


    for epoch in range(epochs):

        if type(model_loss) == SimilarityLoss:
            train_lightgcn_simi_loss(dataset, model, model_loss, optimizer, epoch, logger)
        elif type(model_loss) == BPRLoss:
            train_lightgcn_bpr_loss(dataset, model, model_loss, optimizer, epoch, logger)
        else:
            raise TypeError(f"Loss of type {type(model_loss)} is not supported.")

def train_lightgcn_simi_loss(dataset: BasicDataset, model: LightGCN, model_loss: SimilarityLoss, optimizer: torch.optim.Optimizer, epoch: int, logger):

    # Put the model in training mode
    model.train()

    # Get configurations
    model_config = model.config
    config = model_config.train_config
    batch_size = config.batch_size

    # Generates positive and negative samples at the start of each epoch
    model_loss.new_epoch()

    # Get indices of all nodes in random order
    user_nodes = np.random.permutation(dataset.n_users)
    item_nodes = np.random.arange(dataset.m_items)
    n_batches = math.ceil(len(user_nodes) // batch_size)

    optimizer.zero_grad()
    model.zero_grad()

    loss_sum = 0

    for i in range(n_batches):
        nodes_batch = user_nodes[i * batch_size: (i + 1) * batch_size]
        samples = model_loss.samples[nodes_batch]

        # Add the nodes in this batch and the sampled nodes for this batch to visited set
        visited_user_nodes |= set(nodes_batch)
        visited_user_nodes |= set(samples.flatten())

        user_embs, item_embs = model(nodes_batch)
        loss = model_loss.get_loss(nodes_batch, item_nodes)
        loss_sum += loss.item()

        model_loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        model.zero_grad()

        logger.info(
            f"EP[{epoch}], Batch [{i+1}/{n_batches}], Loss: {loss.item():.4f}, Dealed Nodes [{len(visited_user_nodes)}/{len(user_nodes)}]"
        )

        # Stop when all nodes are trained, this may be before all batches are used    
        if len(visited_user_nodes) == len(user_nodes):
            logger.info(f"Epoch {epoch} complete! All nodes dealed after {i} batches. Average Loss: {loss_sum / (i + 1)} ")            
            break 


def train_lightgcn_bpr_loss(dataset: BasicDataset, model: LightGCN, model_loss: SimilarityLoss, optimizer: torch.optim.Optimizer, epoch: int, logger):


    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    pass

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


