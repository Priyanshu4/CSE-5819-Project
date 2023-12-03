""" Functions for validating/testing fraud detection on embeddings using candidate clusters and anomaly scores.
""" 

import numpy as np
import logging

from .metrics import evaluate_predictions
import src.utils as utils

def clust_anomaly_fraud_detection(clusters, anomaly_scores, threshold: float, true_labels: np.array):
    """
    Detects fraud users using a list of candidate clusters and their anomaly scores.

    Arguments:
        clusters (list): A list of lists of users (indices) in each cluster.
        anomaly_scores (np.ndarray): An array of anomaly scores for each cluster.
        threshold (float): The threshold for the anomaly score to be considered as fraud.
        true_labels (np.ndarray): The true labels of the dataset, where 1 is fraud and 0 is not fraud.

    Returns:
        predicted_labels (np.ndarray): An array of predicted fraudulent user labels.
    """
    predicted_labels = np.zeros(len(true_labels))
    for i, score in enumerate(anomaly_scores):
        if score > threshold:
            predicted_labels[clusters[i]] = 1
    return predicted_labels

def test_clust_anomaly_fraud_detection(clusters, anomaly_scores, threshold_values: list, true_labels: np.ndarray):
    """
    Tests the clust_anomaly_fraud_detection function over a series of threshold values.

    Arguments:
        clusters (list): A list of lists of users (indices) in each cluster.
        anomaly_scores (np.ndarray): An array of anomaly scores for each cluster.
        threshold_values (list): A list of threshold values to test.
        true_labels (np.ndarray): The true labels of the dataset, where 1 is fraud and 0 is not fraud.

    Returns:
        results (list): A list of dictionary results from evaluate predictions for each threshold value.
        best (int): The index of the threshold value with the best F1 score.
    """
    results = []
    best_threshold = -1
    best_f1 = -1
    for i, threshold in enumerate(threshold_values):
        predicted_labels = clust_anomaly_fraud_detection(clusters, anomaly_scores, threshold, true_labels)
        result = evaluate_predictions(true_labels, predicted_labels)
        if result['f1_score'] > best_f1:
            best_f1 = result['f1_score']
            best_threshold = i
        results.append(result)
    return results, best_threshold

def log_clust_anomaly_results(threshold_values, results, best_threshold, logger: logging.Logger):
    """
    Logs the results of the test_clust_anomaly_fraud_detection function.

    Arguments:
        threshold_values (list): A list of threshold values to test. Input you passed test_clust_anomaly_fraud_detection.
        results (list): Output from test_clust_anomaly_fraud_detection.        
        best_threshold (int): The index of the threshold value with the best F1 score. Output from test_clust_anomaly_fraud_detection.
    """
    headers = ["Threshold", "Accuracy", "Precision", "Recall", "F1 Score"]
    results_table = []
    for i, result in enumerate(results):
        results_table.append({
            "Threshold": f"{threshold_values[i]:.3f}",
            "Accuracy": f"{result['accuracy']:.3f}",
            "Precision": f"{result['precision']:.3f}",
            "Recall": f"{result['recall']:.3f}",
            "F1 Score": f"{result['f1_score']:.3f}",
        })
    results_table.append({
        "Threshold": f"Best ({threshold_values[best_threshold]:.3f})",
        "Accuracy": f"{results[best_threshold]['accuracy']:.3f}",
        "Precision": f"{results[best_threshold]['precision']:.3f}",
        "Recall": f"{results[best_threshold]['recall']:.3f}",
        "F1 Score": f"{results[best_threshold]['f1_score']:.3f}",
    })
    utils.print_table(headers, results_table, logger.info)
    return results_table

def hierarchical_clust_anomaly_fraud_detection(clusters, children, anomaly_scores, 
                                               threshold: float, min_size: int, max_allowed_drop: float, 
                                               true_labels: np.array, predicted_labels: np.array = None):
    """
    Detects fraud users using a list of candidate clusters, their 2 children, and their anomaly scores.
    This algorithm is designed to be used without a penalty function for smaller groups.
    In the hierarchical clustering dendrogram, this starts from the bottom and finds the highest clusters with anomaly scores greater than the threshold.
    This allows has a parameter max_allowed_drop. If a parent of 2 children has a score less than max_allowed_drop * min(scores(children)) that it cannot be fraud.
    If a parent node has children that are not fraudulent, then it cannot be fraud.

    Arguments:
        clusters (list): A list of lists of users (indices) in each cluster.
                         The list should be ordered such that the last cluster is the root cluster and the first cluster is the first in the linkage matrix.
                         In format outputted by src.clustering.anomaly.hierarchical_anomaly_scores
        children (list): A list of tuples (pairs) of children clusters for each cluster.
        anomaly_scores (np.ndarray): An array of anomaly scores for each cluster.
        threshold (float): The threshold for the anomaly score to be considered as fraud.
        min_size (int): The minimum size of a cluster to be considered as fraud. Even if set to 0, no clusters of size 1 will be considered as fraud. 
        true_labels (np.ndarray): The true labels of the dataset, where 1 is fraud and 0 is not fraud.
        predicted_labels (np.ndarray): A set of users that have already been predicted as fraud.    

    Returns:
        predicted_labels (np.ndarray): An array of predicted fraudulent user labels.
    """
    fraud_clusters = np.zeros(len(clusters), dtype=np.int8) # 1 if cluster is fraud, 0 if not fraud (this ignores min_size)
    if not predicted_labels:
        predicted_labels = np.zeros(len(true_labels))
    for i, score in enumerate(anomaly_scores):
        if score > threshold:
            child1 = children[i][0]
            child2 = children[i][1]
            if child1 is None or child2 is None:
                # Mark as fraud, but do add users to the predicted labels
                # We use fraud_clusters to say that parents can be fraud if and only if their children are fraud
                fraud_clusters[i] = 1
            elif (fraud_clusters[child1] == 1) and (fraud_clusters[child2] == 1):
                min_child_score = min(anomaly_scores[child1], anomaly_scores[child2])
                #print(score, max_allowed_drop * min_child_score, len(clusters[i]), min_size)
                if score > max_allowed_drop * min_child_score:
                    fraud_clusters[i] = 1
                    if len(clusters[i]) >= min_size:
                        predicted_labels[clusters[i]] = 1
    return predicted_labels

def test_hierarchical_clust_anomaly_fraud_detection(clusters, children, anomaly_scores, 
                                                     threshold_values: list, min_size: int, max_allowed_drops: list, 
                                                     true_labels: np.ndarray):
    """
    Tests the hierarchical_clust_anomaly_fraud_detection function over a series of threshold values.

    Arguments:
        clusters (list): A list of lists of users (indices) in each cluster.
                         The list should be ordered such that the last cluster is the root cluster and the first cluster is the first in the linkage matrix.
                         In format outputted by src.clustering.anomaly.hierarchical_anomaly_scores
        children (list): A list of tuples (pairs) of children clusters for each cluster.
        anomaly_scores (np.ndarray): An array of anomaly scores for each cluster.
        threshold_values (list): A list of threshold values to test.
        min_size (int): The minimum size of a cluster to be considered as fraud. Even if set to 0, no clusters of size 1 will be considered as fraud. 
        max_allowed_drops (list): A list of max_allowed_drop values to test.
        true_labels (np.ndarray): The true labels of the dataset, where 1 is fraud and 0 is not fraud.

    Returns:
        results (list): A list of dictionary results from evaluate predictions for each threshold value.
                        The dictionaries also contain the threshold and max_allowed_drop values.
    """
    results = []
    for i, threshold in enumerate(threshold_values):
        for j, max_allowed_drop in enumerate(max_allowed_drops):
            predicted_labels = hierarchical_clust_anomaly_fraud_detection(clusters, children, anomaly_scores, 
                                                                           threshold, min_size, max_allowed_drop, 
                                                                           true_labels)
            result = evaluate_predictions(true_labels, predicted_labels)
            result['threshold'] = threshold
            result['max_allowed_drop'] = max_allowed_drop
            results.append(result)
    return results
        
def log_results(results, headers, dict_keys, format_strings, logger: logging.Logger):
    """
    Logs a results table.

    Arguments:
        results (list): A list of dictionaries with results for diffeerent parameters. 
        headers (list): A list of the headers for the table.
        dict_keys (list): A list of keys for the dictionary results corresponding to the headers.
        format_strings (list): A list of format strings for each value in the dictionary.
    """
    results_table = []
    for result in results:
        result_dict = dict()
        for i, key in enumerate(dict_keys):
            result_dict[key] = format_strings[i].format(result[key])
        results_table.append(result_dict)
        logger.info(result_dict)
    utils.print_table(headers, results_table, logger.info)
    return results_table

def log_hierarchical_clust_anomaly_results(results, logger: logging.Logger):
    """
    Logs the results of the test_hierarchical_clust_anomaly_fraud_detection function.

    Arguments:
        results (list): Output from test_hierarchical_clust_anomaly_fraud_detection.        
    """
    return log_results(results, 
                ["Threshold", "Max_Drop", "Accuracy", "Precision", "Recall", "F1 Score"],
                ["threshold", "max_allowed_drop", "accuracy", "precision", "recall", "f1_score"],
                ["{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}"],
                logger)
    






