
import pickle
from sklearn.datasets import make_blobs, make_biclusters
import numpy as np


def write_pickle(file_path, content):
    try:
        with open(file_path+".pkl", 'wb') as file:
            pickle.dump(content, file, protocol=4)  
            print('yes')
    except Exception as e:
        print('nono')


def read_pickle(file_name):
    try:
        with open(file_name, "rb") as file:
            loaded_object = pickle.load(file)
            return loaded_object
    except Exception as e:
        print(f"An error occurred while loading the pickled object: {e}")

data = np.array([
    [1, 2, 3, 4],
    [2, 2, 3, 3],
    [2, 3, 4, 4],
    [8, 8, 7, 8],
    [9, 8, 8, 9],
    [9, 9, 8, 10],
    [15, 15, 14, 16],
    [16, 15, 14, 17],
    [16, 16, 15, 17]
])  

blobs, labels = make_blobs(n_samples=165000, n_features=2)
#biclusters, rows, cols = make_biclusters(shape=(65000,10), n_clusters=4)

write_pickle("/Users/niteeshsaravanan/Documents/GitHub/CSE-5819-Project/hclust/data/largeblobs", blobs)