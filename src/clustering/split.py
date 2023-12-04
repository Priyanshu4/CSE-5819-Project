import numpy as np
from sklearn.cluster import KMeans
import math

def split_matrix_random(matrix, max_group_size = 0, num_groups = 0):
    """
    Splits a matrix row-wise into a given number of random groups.

    Arguments:
        matrix (np.ndarray) - matrix to split
        max_group_size (int) - max size of a group
            if passed, num_groups is ignored, num_groups is set to ceil(matrix.shape[0] / approx_group_sizes)
        num_groups (int) - number of groups to split the matrix into

    Returns:
        groups (list) - list of groups 
        group_indices (list) - list of indices of the rows in the original matrix that are in each group 
    """
    rows, columns = matrix.shape

    shuffled_indices = np.random.permutation(rows)

    if max_group_size != 0:

        if num_groups != 0:
            raise ValueError("approx_group_sizes and num_groups cannot both be non-zero")
     
        num_groups = math.ceil(rows / max_group_size)
   
    elif num_groups == 0:
        raise ValueError("approx_group_sizes and num_groups cannot both be zero")
    
    group_size = matrix.shape[0] // num_groups
    
    # Split the matrix into groups using shuffled indics
    groups = []
    group_indices = []
    for i in range(num_groups):
        if i == num_groups - 1:
            group_indices.append(shuffled_indices[i * group_size:])
        else:
            group_indices.append(shuffled_indices[i * group_size:(i + 1) * group_size])
        group = matrix[group_indices[i]]
        groups.append(group)

    return groups, group_indices


def split_matrix_kmeans(matrix, num_groups, max_group_size = 60000, trials = 10):
    """
    Splits an embeddings matrix row-wise into a given number of groups using k-means clustering.

    Arguments:
        matrix (np.ndarray) - matrix to split
        num_groups (int) - number of groups to split the matrix into
        max_group_size (int) - maximum size of each group
        trials (int) - number of trials to run k-means clustering to get groups < max_group_size.
                       If the groups are still too large after trials, we use split_matrix_random to split the groups.
    Returns:
        groups (list) - list of groups 
        group_indices (list) - list of indices of the rows in the original matrix that are in each group 
    """
    def _split_matrix_kmeans(matrix, num_groups):
        kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(matrix)
        labels = kmeans.labels_

        groups = []
        group_indices = []
        for i in range(num_groups):
            group_indices.append(np.where(labels == i)[0])
            group = matrix[group_indices[i]]
            groups.append(group)

        return groups, group_indices

    for i in range(trials):
        groups, group_indices = _split_matrix_kmeans(matrix, num_groups)
        if max([len(group) for group in groups]) <= max_group_size:
            break
    else:
        groups, group_indices = split_matrix_random(matrix, approx_group_sizes = max_group_size)

    return groups, group_indices


def merge_hierarchical_splits(splits):
    """
    Merges the results of multiple splits of the hierarchical clustering algorithm.
    
    Arguments:
        splits (list): A list of splits.
                       Each split should be a list or tuple containing (clusters (list), children (list of pairs), anomaly_scores (list)).

    Returns:
        clusters (list): A list of lists of users (indices) in each cluster.
                         The list should be ordered such that the last cluster is the root cluster and the first cluster is the first in the linkage matrix.
                         In format outputted by src.clustering.anomaly.hierarchical_anomaly_scores
        children (list): A list of tuples (pairs) of children clusters for each cluster.
        anomaly_scores (np.ndarray): An array of anomaly scores for each cluster.
    """
    all_clusters = []
    all_children = []
    all_anomaly_scores = []
    counter = 0
    for split in splits:
        clusters, childrens, anomaly_scores = split[0], split[1], split[2]
        all_clusters.extend(clusters)
        all_anomaly_scores.extend(anomaly_scores)
        for children in childrens:
            if children[0] is not None and children[1] is not None:
                children = (children[0] + counter, children[1] + counter)
            all_children.append(children)
        counter += len(clusters)
    return all_clusters, all_children, np.array(all_anomaly_scores)