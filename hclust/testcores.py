import multiprocessing, os, pickle, time
from multiprocessing import Pool, TimeoutError

def _find_leaves_helper(row, leaves = None):
    # Your original _find_leaves_helper logic here
    if leaves == None:
        leaves = []

    for i, element in enumerate(row):
        leaves.append((i, element))

def process_row(row):
    leaves = _find_leaves_helper(row)
    return leaves

def find_all_leaves(d):
    """
    Finds all leaves under every branch with the generated linkage matrix

    Outputs:
    leaves (list) - list of lists of all leaves under each branch
    """
    leaves = list()

    num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores
    pool = multiprocessing.Pool(processes=num_cores)

    for row in d:
        leaves.append(pool.apply(process_row, args=(row,)))

    pool.close()
    pool.join()

    return leaves

os.chdir("/Users/niteeshsaravanan/Documents/GitHub/CSE-5819-Project/hclust/results/2023-11-03_14-21-56")

with open("linkage.pkl", "rb") as file:
    d = pickle.load(file)
print(d)
leaavestest = find_all_leaves(d)

