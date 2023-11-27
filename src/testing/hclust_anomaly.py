import os
import multiprocessing
import numpy as np

import src.utils as utils
from src.clustering.hclust import HClust
from src.clustering.anomaly import AnomalyScore
import src.clustering.split as split
from src.dataloader import YelpNycDataset

def test_hclust_anomaly_main(dataset, user_embs, tau, logger):
    """ Called by main function in src/main.py to run hcluster and anomaly score and log results.
    """

    n_users, n_features = user_embs.shape

    if n_users > 60000:
        logger.warning(f"Splitting {n_users} into groups with max size 60000.")
        groups, group_indices = split.split_matrix_random(user_embs, approx_group_sizes=60000)
    else:
        groups, group_indices = split.split_matrix_random(user_embs, num_groups=1)

    all_clusters = []
    num_cores = os.cpu_count()
    for i, group in enumerate(groups):
        hclust = HClust(group)
        with utils.timer(name="linkages"):
            linkage = hclust.generate_linkage_matrix()

        with utils.timer(name="leaves"):
            with multiprocessing.Pool(num_cores) as p:
                # Find the leaves under every branch of the hierarchical clustering
                leaves = p.map(hclust._find_leaves_iterative, [row for row in linkage])
                group_user_indices = np.array(group_indices[i])
                mapped_leaves = [group_user_indices[group] for group in leaves]
                all_clusters.extend(mapped_leaves)
   
    hclust_time_info = utils.timer.formatted_tape_str(select_keys=["linkages", "leaves"])
    utils.timer.zero(select_keys=["linkages", "leaves"])
    logger.info(f"Hierarchical Clustering Time: {hclust_time_info}")

    # Generate anomaly scores
    with utils.timer(name="anomaly_scores"):
        use_metadata = (type(dataset) == YelpNycDataset)
        anomaly_scorer = AnomalyScore(dataset, use_metadata=use_metadata, burstness_threshold=tau)
        anomaly_scores = anomaly_scorer.generate_anomaly_scores(clusters=all_clusters)
    anomaly_score_time_info = utils.timer.formatted_tape_str(select_keys=["anomaly_scores"])
    utils.timer.zero(select_keys=["anomaly_scores"])
    logger.info(f"Anomaly Score Time: {anomaly_score_time_info}")

    # TODO: Make a testing script to test different anomaly score thresholds and compute metrics
    