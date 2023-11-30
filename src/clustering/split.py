import numpy as np
import math

def split_matrix_random(matrix, approx_group_sizes = 0, num_groups = 0):
    """
    Splits a matrix row-wise into a given number of random groups.

    Arguments:
        matrix (np.ndarray) - matrix to split
        approx_group_sizes (int) - approximate size of each group
            if passed, num_groups is ignored, num_groups is set to ceil(matrix.shape[0] / approx_group_sizes)
        num_groups (int) - number of groups to split the matrix into

    Returns:
        groups (list) - list of groups 
        group_indices (list) - list of indices of the rows in the original matrix that are in each group 
    """
    rows, columns = matrix.shape

    shuffled_indices = np.random.permutation(rows)

    if approx_group_sizes != 0:

        if num_groups != 0:
            raise ValueError("approx_group_sizes and num_groups cannot both be non-zero")
     
        num_groups = math.ceil(rows / approx_group_sizes)
   
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
