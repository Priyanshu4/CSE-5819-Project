""" Functions for validating/testing fraud detection on embeddings with DBSCAN and OPTICS.
"""

import numpy as np
import logging
import argparse
from pathlib import Path
import os
import sys
from sklearn.cluster import OPTICS, DBSCAN, cluster_optics_dbscan

from .metrics import evaluate_predictions
import src.utils as utils


def dbscan_fraud_detection(data: np.ndarray, epsilon: float, min_samples: int) -> np.ndarray:
    """
    Applies DBSCAN clustering algorithm on the input data and classifies the dataset into noise points and non-noise points.
    Non-noise points are considered to fraudulent because they are part of a high-density group with highly similar (spam) behavior.
    
    Args:
    data: A 2D numpy array of  shape (datapoints, features).
    epsilon: A float representing the maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples: An integer representing the number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
    A 1D numpy array where noise points are 0 and non-noise points are 1.
    """
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', leaf_size=500, metric='euclidean')
    dbscan.fit(data)
    labels = dbscan.labels_
    return np.where(labels == -1, 0, 1)

def optics_fraud_detection(data: np.ndarray, min_samples: int, xi: float) -> np.ndarray:
    """
    Applies OPTICS clustering algorithm on the input data and classifies the dataset into noise points and non-noise points.
    Non-noise points are considered to fraudulent because they are part of a high-density group with highly similar (spam) behavior.
    
    Args:
    data: A 2D numpy array of  shape (datapoints, features).
    min_samples: An integer representing the number of samples in a neighborhood for a point to be considered as a core point.
    xi: A float representing the minimum steepness on the reachability plot that constitutes a cluster boundary.
        
    Returns:
    A 1D numpy array where noise points are 0 and non-noise points are 1.
    """
    optics = OPTICS(min_samples=min_samples, xi=xi, metric='minkowski', p=2)
    optics.fit(data)
    labels = optics.labels_
    return np.where(labels == -1, 0, 1)

def test_dbscan_fraud_detection(data: np.ndarray, epsilon_values: list, min_samples_values: list, true_labels: np.ndarray, logger: logging.Logger):
    """
    Tests the dbscan_fraud_detection function over a series of epsilon and min_samples values.
    
    Args:
    data: A 2D numpy array of  shape (users, features).
    epsilon_values: A list of float values representing the maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples_values: A list of integer values representing the number of samples in a neighborhood for a point to be considered as a core point.
    true_labels: A 1D numpy array of true labels for the data.
    logger: A logging.Logger object to log the results.
        
    Returns:
    A dictionary containing the evaluation metrics for each combination of epsilon and min_samples values.
    """
    results = {}
    for epsilon in epsilon_values:
        for min_samples in min_samples_values:
            predicted_labels = dbscan_fraud_detection(data, epsilon, min_samples)
            results[(epsilon, min_samples)] = evaluate_predictions(true_labels, predicted_labels)
            logger.debug(f"epsilon={epsilon}, min_samples={min_samples}, results={results[(epsilon, min_samples)]}")
    return results

def test_optics_dbscan_fraud_detection(data: np.ndarray, xi: float, epsilon_values: list, min_samples_values: list, true_labels: np.ndarray, logger: logging.Logger):
    """
    Test function to evaluate the performance of the OPTICS and DBSCAN clustering algorithms for fraud detection.
    This function works using OPTICS, and then extracts DBSCAN clusters from the OPTICS data.
    Therefore, the results may be slightly different than the test_dbscan_fraud_detection function.

    Args:
    - data (np.ndarray): The dataset to be used for clustering.
    - xi (float): The minimum separation between samples required to form a cluster in OPTICS.
    - epsilon_values (list): A list of values for the maximum distance between two samples for one to be considered as in the neighborhood of the other in DBSCAN.
    - min_samples_values (list): A list of values for the number of samples in a neighborhood for a point to be considered as a core point in DBSCAN.
    - true_labels (np.ndarray): The true labels of the dataset.
    - logger (logging.Logger): A logger object to log debug and info messages.

    Returns:
    - results (dict): A dictionary containing the evaluation results for each combination of hyperparameters.
    """
    results = {}
    for min_samples in min_samples_values:
        optics = OPTICS(min_samples=min_samples, xi=xi, metric='minkowski', p=2)
        optics.fit(data)
        
        optics_labels = np.where(optics.labels_ == -1, 0, 1)
        results[("OPTICS", min_samples)] = evaluate_predictions(true_labels, optics_labels)
        logger.debug(f"OPTICS: min_samples={min_samples}, results={results[('OPTICS', min_samples)]}")
 
        for epsilon in epsilon_values:

            clusters = cluster_optics_dbscan(
                reachability=optics.reachability_,
                core_distances=optics.core_distances_,
                ordering=optics.ordering_,
                eps=epsilon,
            )
        
            predicted_labels = np.where(clusters == -1, 0, 1)
            results[(epsilon, min_samples)] = evaluate_predictions(true_labels, predicted_labels)
            logger.debug(f"epsilon={epsilon}, min_samples={min_samples}, results={results[(epsilon, min_samples)]}")
    return results

def log_dbscan_results(results: dict, logger: logging.Logger):
    headers = ["Epsilon", "Min Samples", "Accuracy", "Precision", "Recall", "F1 Score"]
    results_data = []
    for key, result in results.items():
        epsilon = key[0]
        min_samples = key[1]
        results_data.append({
            "Epsilon": epsilon,
            "Min Samples": min_samples,
            "Accuracy": f"{result['accuracy']:.3f}",
            "Precision": f"{result['precision']:.3f}",
            "Recall": f"{result['recall']:.3f}",
            "F1 Score": f"{result['f1_score']:.3f}",
        })
    utils.print_table(headers, results_data, logger.info)    
    return results    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Clustering Technique')
    parser.add_argument('--embeddings', type=str, required=True, help='File path of the embeddings file')
    parser.add_argument('--labels', type=str, required=True, help='File path of the true labels file')
    parser.add_argument('--algorithm', type=str, default="OPTICS", help='Algorithm to use: either OPTICS or DBSCAN')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    labels_path = Path(args.labels)
    if not labels_path.exists():
        logger.error(f"Labels file {labels_path} does not exist.")
        exit(1)

    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        logger.error(f"Embeddings file {embeddings_path} does not exist.")
        exit(1)

    embeddings_file_size = os.path.getsize(args.embeddings)
    embeddings_file_size_mb = embeddings_file_size / (1024 * 1024)
    if embeddings_file_size_mb > 128:
        logger.info(f"Embeddings file size is {embeddings_file_size_mb} MiB.")

    true_labels = np.load(args.labels, allow_pickle=True)
    embeddings = utils.load_user_embeddings(embeddings_path)

    embeddings_size = embeddings.nbytes
    embeddings_size_gb = embeddings_size / (1024 * 1024 * 1024)

    if embeddings_size_gb > 4:
        logger.warning(f"Embedding array size is {embeddings_size_gb} GiB.")
    else:
        logger.debug(f"Embedding array size is {embeddings_size} bytes.")

    logger.debug(f"Embedding has {embeddings.shape[0]} users and {embeddings.shape[1]} features.")

    results = []
    if args.algorithm == "OPTICS":
        epsilon_values = [0.01, 0.1, 1, 5, 10]
        min_samples_values = [5, 10, 20, 50]
        results = test_optics_dbscan_fraud_detection(embeddings, 0.05, epsilon_values, min_samples_values, true_labels, logger)
    elif args.algorithm == "DBSCAN":
        epsilon_values = [0.01, 0.1, 1, 5, 10]
        min_samples_values = [5, 10, 20, 50]
        results = test_dbscan_fraud_detection(embeddings, epsilon_values, min_samples_values, true_labels, logger)
    log_dbscan_results(results, logger)


