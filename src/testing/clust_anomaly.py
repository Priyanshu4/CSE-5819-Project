""" Functions for validating/testing fraud detection on embeddings using candidate clusters and anomaly scores.
""" 

import numpy as np
import logging

from .metrics import evaluate_predictions
import src.utils as utils

def userwise_anomaly_scores(clusters, anomaly_scores, n_users):
    """
    Given a set a clusters and anomaly scores for each clusters, this computes anomaly scores for each user.
    The anomaly score for a user is the lowest anomaly score of any cluster it is in.
    This indicates the threshold at which a user may be classified as fraud.

    Arguments:
        clusters (list): A list of lists of users (indices) in each cluster.
        anomaly_scores (np.ndarray): An array of anomaly scores for each cluster.
        n_users (int): number of users

    Returns:
        user_anomaly_scores (np.ndarray): An array of anomaly scores for each user
    """
    user_anomaly_scores = np.zeros(n_users)
    for i, score in enumerate(anomaly_scores):
        for user in clusters[i]:
            if user_anomaly_scores[user] < score:
                user_anomaly_scores[user] = score
    return user_anomaly_scores

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
        cluster_labels (np.ndarray): An array of predicted labels for each group
    """
    cluster_labels = np.zeros(len(clusters), dtype=np.int8)
    predicted_labels = np.zeros(len(true_labels), dtype=np.int8)
    for i, score in enumerate(anomaly_scores):
        if score > threshold:
            cluster_labels[i] = 1
            predicted_labels[clusters[i]] = 1
    return predicted_labels, cluster_labels

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
        predicted_labels, cluster_labels = clust_anomaly_fraud_detection(clusters, anomaly_scores, threshold, true_labels)
        largest_fraud_group = max(range(len(cluster_labels)), 
                                        key=lambda i: len(clusters[i]) if cluster_labels[i] == 1 else -1)
        largest_fraud_group_size = len(clusters[largest_fraud_group])
        result = evaluate_predictions(true_labels, predicted_labels)
        result["largest_fraud_group_size"] = largest_fraud_group_size
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
    headers = ["Threshold", "Accuracy", "Precision", "Recall", "F1 Score", "Largest Group"]
    results_table = []
    for i, result in enumerate(results):
        results_table.append({
            "Threshold": f"{threshold_values[i]:.4f}",
            "Accuracy": f"{result['accuracy']:.3f}",
            "Precision": f"{result['precision']:.3f}",
            "Recall": f"{result['recall']:.3f}",
            "F1 Score": f"{result['f1_score']:.3f}",
            "Largest Group": f"{result['largest_fraud_group_size']}"
        })
    results_table.append({
        "Threshold": f"Best ({threshold_values[best_threshold]:.4f})",
        "Accuracy": f"{results[best_threshold]['accuracy']:.3f}",
        "Precision": f"{results[best_threshold]['precision']:.3f}",
        "Recall": f"{results[best_threshold]['recall']:.3f}",
        "F1 Score": f"{results[best_threshold]['f1_score']:.3f}",
        "Largest Group": f"{results[best_threshold]['largest_fraud_group_size']}"

    })
    utils.print_table(headers, results_table, logger.info)
    return results_table

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








