from hclust import HClust
from anomaly import AnomalyScore
from dataloader import DataLoader
import os, pickle
from pathlib import Path

from multiprocessing import Pool
import time, datetime, os, sys, json, logging, pickle, argparse


def write_pickle(file_path, content):
    try:
        with open(file_path+".pkl", 'wb') as file:
            pickle.dump(content, file, protocol=4)
        logging.info(f"File '{file_path}' has been successfully written.")
    except Exception as e:
        logging.error(f"An error occurred while writing {file_path}: {e}")


def read_pickle(file_name):
    try:
        with open(file_name, "rb") as file:
            loaded_object = pickle.load(file)
            return loaded_object
    except Exception as e:
        logging.error(f"An error occurred while writing {file_name}: {e}")


def time_method(obj, method_name, *args):
    if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
        method = getattr(obj, method_name)
    else:
        raise ValueError(f"{obj} does not have a callable method named {method_name}")

    start_time = time.time()
    result = method(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def find_leaves_wrapper(args):
    clusterer,row = args
    return clusterer._find_leaves_iterative(row)

def get_args():
    parser = argparse.ArgumentParser(
    usage='%(prog)s --data=<DATANAME>',
    description="This program requires a data argument (--data)."
)

    # Create a `--data` argument with an optional string value
    parser.add_argument("--data", type=str, help="Data argument", required=True)
    parser.add_argument("--dend", action='store_true', help="Use if you want to get the dendrogram visualization")
    parser.add_argument("--tau", type=int, help="Data argument", required=True)


    args = parser.parse_args()
    dataname = args.data


    with open("config.json", "r") as json_file:
        arg_dict = json.load(json_file)

    if dataname not in arg_dict['data'].keys():
        print(f"{dataname} not setup in config.json")
        sys.exit(1)
    return args, arg_dict, dataname

def create_dirs(dataname):

    results_folder = 'results'
    calling_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(calling_directory)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_folder = os.path.join(results_folder, timestamp)
    os.makedirs(run_folder)
    log_file = os.path.join(run_folder, 'run.log')

    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Running with data: {dataname}")

    return run_folder

def main_hclust(args, arg_dict, dataname):

    run_folder = create_dirs(dataname)
    datapath = arg_dict['data'][dataname]
    mappath = arg_dict['data'][dataname]
    data = read_pickle(datapath)
    clusterer = HClust(data)

    path = os.path.join(run_folder, 'linkage')
    logging.info("generating linkage matrix")
    result, exec_time = time_method(clusterer, "generate_linkage_matrix", *())
    logging.info(f"{exec_time} seconds to create linkage matrix")
    write_pickle(path, clusterer.Z)

    if args.dend:
        logging.info("generating dendrogram")
        try:
            result, exec_time = time_method(clusterer, "generate_dendrogram", run_folder)
            logging.info(f"{exec_time} seconds to create dendrogram")
        except Exception as e:
            logging.error(f"Failed to generate dendrogram: {e}")

    path = os.path.join(run_folder, 'leaves')
    num_cores = os.cpu_count()
    with Pool(num_cores) as p:
        start = time.time()
        result = p.map(find_leaves_wrapper, [(clusterer, row) for row in clusterer.Z])
        end = time.time()
        logging.info(f"{end - start} seconds to find all leaves")
        write_pickle(path, result)

    print('Completed :D')
    return result, mappath

def get_dataset():
    pwd = os.getcwd()
    CONFIGS_PATH = Path(pwd).parent / "lightgcn_embedder" / "configs"
    DATASET_CONFIG = CONFIGS_PATH / "datasets.json"

    loader = DataLoader(DATASET_CONFIG)
    dataset = loader.load_dataset("yelpnyc")
    return dataset

def get_mapping(mappath):
    with open(mappath, "rb") as mf:
        mapping = pickle.load(mf)
    return mapping

def main_anomaly(leaves, mappath, args):
    dataset = get_dataset()
    mapping = get_mapping(mappath)
    adj = dataset.graph_u2i.toarray()
    avg_ratings = dataset.metadata_df.groupby(dataset.METADATA_ITEM_ID)[dataset.METADATA_STAR_RATING].mean()
    average_ratings = avg_ratings.values
    rating_matrix = dataset.rated_graph_u2i.toarray()
    first_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].min()
    last_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].max()
    review_times = (last_date-first_date).astype('timedelta64[D]')

    AS = AnomalyScore(leaves,mapping,adj,average_ratings,rating_matrix,review_times, args.tau)
    scores = AS.generate_anomaly_scores()
    return scores, AS.mapped_leaves

if __name__ == "__main__":
    args, args_dict, dataname = get_args()
    leaves, mappath = main_hclust(args, args_dict, dataname)
    main_anomaly(leaves, mappath, args)