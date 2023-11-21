import torch
import numpy as np
import math
from loss import ModelLoss, SimilarityLoss, BPRLoss
from dataloader import BasicDataset
from lightgcn import LightGCN
import utils
import sampling

def train_lightgcn_simi_loss(dataset: BasicDataset, model: LightGCN, model_loss: SimilarityLoss, optimizer: torch.optim.Optimizer, epoch: int, logger):

    # Put the model in training mode
    model.train()

    # Get configurations
    model_config = model.config
    config = model_config.train_config
    batch_size = config.batch_size

    # Generates positive and negative samples at the start of each epoch
    logger.debug(f"EP[{epoch}]: Sampling positive and negative user nodes...")
    with utils.timer(name="Sampling"):
        samples = model_loss.sample_train_set_pos_neg_users()
    logger.debug(f"EP[{epoch}]: Positive and negative user nodes sampled.")

    # Get indices of all nodes in random order
    user_nodes = np.random.permutation(dataset.n_users)
    item_nodes = np.arange(dataset.m_items)
    n_batches = math.ceil(len(user_nodes) // batch_size)

    optimizer.zero_grad()
    model.zero_grad()

    loss_sum = 0
    avg_loss = 0
    visited_user_nodes = set()

    with utils.timer(name="Training"):
        for i in range(n_batches):
            nodes_batch = user_nodes[i * batch_size: (i + 1) * batch_size]
            extended_nodes_batch = model_loss.extend_user_node_batch(nodes_batch, samples)

            # Add the nodes in this batch and the sampled nodes for this batch to visited set
            visited_user_nodes |= set(extended_nodes_batch)

            user_embs, item_embs = model(extended_nodes_batch, item_nodes)
            loss = model_loss.get_loss(nodes_batch, user_embs=user_embs, extended_user_nodes_batch=extended_nodes_batch, samples=samples)
            loss_sum += loss.item()

            model_loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            model.zero_grad()

            logger.debug(
                f"EP[{epoch}], Batch [{i+1}/{n_batches}], Loss: {loss.item():.4f}, Dealed Nodes [{len(visited_user_nodes)}/{len(user_nodes)}]"
            )

            # Stop when all nodes are trained, this may be before all batches are used    
            if len(visited_user_nodes) == len(user_nodes):
                avg_loss = loss_sum / (i + 1)
                break
    
    time_info = utils.timer.formatted_tape_str(select_keys=["Sampling", "Training"])
    utils.timer.zero()
    logger.info(f"EPOCH {epoch} complete. Average Loss: {avg_loss:.4f}, Time: {time_info}")            

    return avg_loss

def minibatch(batch_size, *tensors):

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def train_lightgcn_bpr_loss(dataset: BasicDataset, model: LightGCN, bpr_loss: BPRLoss, optimizer: torch.optim.Optimizer, epoch: int, logger):

    # Put model in training mode
    model.train() 

    # Get configurations
    model_config = model.config
    config = model_config.train_config
    batch_size = config.batch_size
    device = model_config.device

    with utils.timer(name="BPR_Sampling"):
        S = sampling.BPR_UniformSample_original(dataset)
        
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()

        users = users.to(device)
        posItems = posItems.to(device)
        negItems = negItems.to(device)
        users, posItems, negItems = shuffle(users, posItems, negItems)

    with utils.timer(name="BPR_Training"):
        n_batches = len(users) // batch_size + 1
        loss_sum = 0.
        for (batch_i,
            (batch_users,
            batch_pos,
            batch_neg)) in enumerate(minibatch(batch_size, 
                                                    users,
                                                    posItems,
                                                    negItems)):

            loss = bpr_loss.get_loss(*model.getEmbeddingsForBPR(batch_users, batch_pos, batch_neg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

    avg_loss = loss_sum / n_batches
    time_info = utils.timer.formatted_tape_str(select_keys=["BPR_Sampling", "BPR_Training"])
    utils.timer.zero()

    logger.info(f"EPOCH {epoch} complete. Average Loss: {avg_loss:.4f}, Time: {time_info}")            

    return avg_loss