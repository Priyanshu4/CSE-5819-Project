from hclust import HClust
import numpy as np
from sklearn.datasets import make_blobs, make_biclusters
import time, datetime, os, pickle, sys, json, logging


# Create the results folder if it doesn't exist
results_folder = 'results'
calling_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(calling_directory)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Create a timestamp for the run
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Create the run folder
run_folder = os.path.join(results_folder, timestamp)
os.makedirs(run_folder)

# Create a log file in the run folder
log_file = os.path.join(run_folder, 'run.log')

# Configure the logging module to write to the log file
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def write_pickle(file_path, content):
    try:
        with open(file_path+".pkl", 'wb') as file:
            pickle.dump(content, file)
        logging.info(f"File '{file_path}' has been successfully written.")
    except Exception as e:
        logging.error(f"An error occurred while writing the file: {e}")


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

    arg_dict = {"small":small, "blobs":blobs, "biclusters":biclusters}

    if len(sys.argv) < 2:
        print("Usage: python hclusttest.py <small/blobs/bicluster>")
        sys.exit(0)
        
    arg = sys.argv[-1]
    logging.info(f"Running {arg}")

    data = arg_dict[arg]
    
    clusterer = HClust(data)

    path = os.path.join(run_folder, 'linkage')
    logging.info("generating linkage matrix")
    start = time.time()
    clusterer.generate_linkage_matrix()
    end = time.time()
    logging.info(f"{end-start} seconds to create linkage matrix")
    write_pickle(path, clusterer.Z)


    start = time.time()
    logging.info("generating dendrogram")
    try:
        clusterer.generate_dendrogram(run_folder)
        end = time.time()
        logging.info(f"{end-start} seconds to create dendrogram")
    except Exception as e:
        logging.error("Failed to generate dendrogram: {e}")


    path = os.path.join(run_folder, 'leaves')
    logging.info("starting leaves search")
    start = time.time()
    leaves = clusterer.find_all_leaves()
    end = time.time()
    logging.info(f"{end-start} seconds to find all leaves")
    write_pickle(path, leaves)
    
