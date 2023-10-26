import torch
import numpy as np
import math

def minibatch(batch_size, *tensors):

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


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
