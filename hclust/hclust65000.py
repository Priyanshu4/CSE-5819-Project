from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt
import time



def find_all_leaves(Z):
    leaves = list()

    for row in Z:
        branch = []
        find_leaves(Z,row, branch)
        leaves.append(branch)
    
    return leaves

def find_leaves(Z, row, leaves=None):
    if leaves is None:
        leaves = []

    maxlen = int(Z[-1, -1]) - 1
    lefti = int(row[0])
    righti = int(row[1])

    if lefti <= maxlen:
        leaves.append(lefti)
    else:
        newrow = Z[lefti - maxlen-1]
        find_leaves(Z, newrow, leaves)

    if righti <= maxlen:
        leaves.append(righti)
    else:
        newrow = Z[righti - maxlen-1]
        find_leaves(Z, newrow,leaves)
         
    return leaves


def write_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"File '{file_path}' has been successfully written.")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")


file_path = "results/"

blobs, labels = make_blobs(n_samples=65000, n_features=10)

start = time.time()
linkage_matrix = linkage(blobs, method='average')
endlinkage = time.time()

s = f"{endlinkage-start} seconds to create linkage matrix"
write_file(file_path+"linkage.txt", s)


start = time.time()
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
dendrogram_info = dendrogram(linkage_matrix, ax=ax)
enddendrogram = time.time()

fig.savefig(file_path+'dendrogram.png', dpi=300)
s = f"{enddendrogram-start} seconds to create dendrogram"
write_file(file_path+"dendrogram.txt", s)


start = time.time()
leaves_list = find_all_leaves(linkage_matrix)
endleaves = time.time()

s = f"{endleaves-start} seconds to create leaves list\n{leaves_list}"
write_file(file_path+"leaves.txt", s)