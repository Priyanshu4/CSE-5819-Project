import torch
import numpy as np
import math
from dataloader import BasicDataset
from lightgcn import LightGCN
from dataclasses import dataclass


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
        return _sample_train_set_fast(dataset, n_pos, n_neg)
    return _sample_train_set_normal(dataset, n_pos, n_neg)

def _sample_train_set_normal(dataset, n_pos, n_neg):
    g_u2u = dataset.graph_u2u 
    samples = np.zeros((dataset.n_users, n_pos + n_neg), dtype=int)
    all_indices = np.arange(dataset.n_users)
    
    for i in range(n_users):
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

    return pos_samples, neg_samples

def _sample_train_set_fast(dataset, n_pos, n_neg):
    g_u2u = dataset.graph_u2u 
    samples = np.zeros((dataset.n_users, n_pos + n_neg), dtype=int)
    all_indices = np.arange(dataset.n_users)
    
    for i in range(n_users):
        pos_pool = g_u2u[i].nonzero()[1]                # indices of all positive nodes for user i

        if len(pos_pool) >= n_pos:
            samples[i, :n_pos] = np.random.choice(pos_pool, n_pos, replace=False)
        else:
            samples[i, :len(pos_pool)] = pos_pool
            samples[i, len(pos_pool):n_pos] = np.random.choice(all_indices, n_pos - len(pos_pool), replace=False)
        
        samples[i, n_pos:] = np.random.choice(all_indices, n_neg, replace=False)

    return pos_samples, neg_samples

def train_lightgcn(dataset: BasicDataset, model: LightGCN, loss):

    model.train() # Put the model in training mode
    device = model.config.device

    pos_and_neg_samples = sample_train_set(dataset, pos_samples, neg_samples)
    train_nodes = np.random.permutation(dataset.n_users)





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
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
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

def train_model(Dl, args, logger, deepFD, model_loss, device, epoch):
    train_nodes = getattr(Dl, Dl.ds + "_train")
    np.random.shuffle(train_nodes)

    params = []
    for param in deepFD.parameters():
        if param.requires_grad:
            params.append(param)
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.gamma)
    optimizer.zero_grad()
    deepFD.zero_grad()

    batches = math.ceil(len(train_nodes) / args.b_sz)
    visited_nodes = set()
    training_cps = Dl.get_train()
    logger.info("sampled pos and neg nodes for each node in this epoch.")
    for index in range(batches):
        nodes_batch = train_nodes[index * args.b_sz : (index + 1) * args.b_sz]
        nodes_batch = np.asarray(model_loss.extend_nodes(nodes_batch, training_cps))
        visited_nodes |= set(nodes_batch)

        embs_batch, recon_batch = deepFD(nodes_batch)
        loss = model_loss.get_loss(nodes_batch, embs_batch, recon_batch)

        logger.info(
            f"EP[{epoch}], Batch [{index+1}/{batches}], Loss: {loss.item():.4f}, Dealed Nodes [{len(visited_nodes)}/{len(train_nodes)}]"
        )
        loss.backward()

        nn.utils.clip_grad_norm_(deepFD.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        deepFD.zero_grad()

        # stop when all nodes are trained
        if len(visited_nodes) == len(train_nodes):
            break

    return deepFD
