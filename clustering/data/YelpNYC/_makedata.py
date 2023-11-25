
import numpy as np
import pickle
import torch
import os


def write_pickle(file_path, content):
    try:
        with open(file_path+".pkl", 'wb') as file:
            pickle.dump(content, file, protocol=4)  
            print('yes')
    except Exception as e:
        print(f'{e}')


def read_pickle(file_name):
    try:
        with open(file_name, "rb") as file:
            loaded_object = pickle.load(file)
            return loaded_object
    except Exception as e:
        print(f"An error occurred while loading the pickled object: {e}")



def unpickle(file_name):
    try:
        with open(file_name, "rb") as file:
            loaded_object = pickle.load(file)
            print("Successfully loaded the pickled object:")
            return loaded_object
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred while loading the pickled object: {e}")

# Example usage:
file = "../../../data/yelpnyc/embedded/deepfd/embs_ep10.pkl"
obj = unpickle(file)
large_matrix = obj.numpy()


# Get the current working directory
cwd = os.getcwd()


num_rows, num_cols = large_matrix.shape
print(large_matrix.shape)

"""# Shuffle the row indices
shuffled_indices = np.random.permutation(num_rows)

# Calculate the number of rows for each group
group_size = num_rows // 3

# Split the shuffled indices into three equal-sized groups
group1_indices = shuffled_indices[:group_size]
group2_indices = shuffled_indices[group_size:2 * group_size]
group3_indices = shuffled_indices[2 * group_size:]

# Use the indices to split the matrix into three groups
group1 = large_matrix[group1_indices]
group2 = large_matrix[group2_indices]
group3 = large_matrix[group3_indices]


write_pickle("g1", group1)
write_pickle("g2", group2)
write_pickle("g3", group3)

write_pickle("g1_map", group1_indices)
write_pickle("g2_map", group2_indices)
write_pickle("g3_map", group3_indices)
"""