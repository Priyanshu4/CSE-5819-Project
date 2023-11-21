import logging
import numpy as np
from dataloader import BasicDataset
from time import time

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    _sampling = imp_from_filepath(path)
    _sample_ext = True
except:
    logging.getLogger("Logger").info("cpp sampling extension not loaded")
    _sample_ext = False


def set_sampling_seed(seed):
    np.random.seed(seed)
    if _sample_ext:
        _sampling.set_seed(seed)


def sample_train_set_pos_neg_users(dataset: BasicDataset, n_pos: int, n_neg: int, fast: bool = False):
    """ For each user node in the dataset, this samples n_pos positive nodes and n_neg negative nodes.
        A positive node shares an item and a negative node does not share an item.
        This returns a 2D numpy array of shape (n_users, n_pos+n_neg) with the indices of samples for each user node.
        samples[i, :] gives the positive and negative samples for the ith node in the dataset,
        If there are not enough negative or positive samples, empty spots are filled with completely random nodes.

        When fast is True, we assume the dataset has high number of negative nodes (sparse).
        Therefore, we assume that the a sample of all nodes is a sample of mostly negative nodes.
    """
    if fast:
        return _sample_train_set_pos_neg_users_fast(dataset, n_pos, n_neg)
    return _sample_train_set_pos_neg_users_normal(dataset, n_pos, n_neg)

def _sample_train_set_pos_neg_users_normal(dataset, n_pos, n_neg):
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

def _sample_train_set_pos_neg_users_fast(dataset, n_pos, n_neg):
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

def BPR_UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    start = time()
    if _sample_ext:
        S = _sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, _get_positive_items(dataset), neg_ratio)
    else:
        S = _BPR_UniformSample_original_python(dataset)
    return S

def _get_positive_items(dataset):
    allPos = []
    for i in range(dataset.graph_u2i.shape[0]):
        row_slice = dataset.graph_u2i[i] 
        indices = row_slice.nonzero()[1]
        allPos.append(indices)
    return allPos

def _BPR_UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    #user_num = dataset.trainDataSize
    user_num = dataset.n_users
    users = np.random.randint(0, dataset.n_users, user_num)
    
    # List of positive item indices in each row
    allPos = _get_positive_items(dataset)

    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)