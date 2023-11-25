from dataloader import DataLoader
import pandas as pd
import os, pickle
from pathlib import Path
from anomaly import AnomalyScore

def get_dataset():
    pwd = os.getcwd()
    CONFIGS_PATH = Path(pwd) / "lightgcn_embedder" / "configs"
    DATASET_CONFIG = CONFIGS_PATH / "datasets.json"

    loader = DataLoader(DATASET_CONFIG)
    dataset = loader.load_dataset("yelpnyc")
    return dataset

def get_leaves_mapping(leavespath, mappath):
    with open(leavespath, "rb") as lf:
        leaves = pickle.load(lf)
    with open(mappath, "rb") as mf:
        mapping = pickle.load(mf)
    return leaves,mapping

def main():
    dataset = get_dataset()
    leaves,mapping = get_leaves_mapping( ... , ... )
    adj = dataset.graph_u2i.toarray()
    avg_ratings = dataset.metadata_df.groupby(dataset.METADATA_ITEM_ID)[dataset.METADATA_STAR_RATING].mean()
    average_ratings = avg_ratings.values
    rating_matrix = dataset.rated_graph_u2i.toarray()
    first_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].min()
    last_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].max()
    review_times = (last_date-first_date).astype('timedelta64[D]')

    AS = AnomalyScore(leaves,mapping,adj,average_ratings,rating_matrix,review_times)
    scores = AS.generate_anomaly_scores()
    #AS.mapped_leaves

if __name__ == main():
    main()