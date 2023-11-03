from hclust import HClust
import numpy as np
from sklearn.datasets import make_blobs, make_biclusters
import time, datetime, os, pickle, sys

results_folder = 'results'
calling_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(calling_directory)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

run_folder = os.path.join(results_folder, timestamp)
os.makedirs(run_folder)

def write_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"File '{file_path}' has been successfully written.")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")


if __name__ == "__main__":

    small = np.array([
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
    blobs, labels = make_blobs(n_samples=65000, n_features=10)
    biclusters, rows, cols = make_biclusters(shape=(65000,10), n_clusters=4)

    arg_dict = {"small":small, "blobs":blobs, "bicluster":biclusters}

    if len(sys.argv) < 2:
        print("Usage: python hclusttest.py <small/blobs/bicluster>")
        sys.exit(0)
        
    arg = sys.argv[-1]

    data = arg_dict[arg]
    
    clusterer = HClust(data)

    path = os.path.join(run_folder, 'linkage.txt')
    start = time.time()
    clusterer.generate_linkage_matrix()
    end = time.time()
    s = f"{end-start} seconds to create linkage matrix"
    write_file(path, s)

    path = os.path.join(run_folder, 'dendrogram.txt')
    start = time.time()
    clusterer.generate_dendrogram(run_folder)
    end = time.time()
    s = f"{end-start} seconds to create dendrogram"
    write_file(path, s)

    path = os.path.join(run_folder, 'leaves.')
    start = time.time()
    leaves = clusterer.find_all_leaves()
    end = time.time()
    s = f"{end-start} seconds to find all leaves"
    write_file(path+'txt', s)
    with open(path+'pkl', 'wb') as file:
       pickle.dump(leaves, file)
    
