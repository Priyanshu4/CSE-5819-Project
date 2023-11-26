""" Functions for validating/testing fraud detection on embeddings using candidate clusters and anomaly scores.
""" 

import numpy as np
import logging

from .metrics import evaluate_predictions
import src.utils as utils

def clust_anomaly_fraud_detection(all_clusters, anomaly_scores, threshold: float, true_labels: np.array):
    """
    Detects fraud users using a list of candidate clusters and their anomaly scores.

    Arguments:
        all_clusters (list): A list of lists of users (indices) in each cluster.
        anomaly_scores (np.ndarray): An array of anomaly scores for each cluster.
        threshold (float): The threshold for the anomaly score to be considered as fraud.
        true_labels (np.ndarray): The true labels of the dataset, where 1 is fraud and 0 is not fraud.

    Returns:
        predicted_labels (np.ndarray): An array of predicted fraudulent user labels.
    """
    predicted_labels = np.zeros(len(true_labels))
    for i, score in enumerate(anomaly_scores):
        if score > threshold:
            predicted_labels[all_clusters[i]] = 1
    return predicted_labels

def test_clust_anomaly_fraud_detection(all_clusters, anomaly_scores, threshold_values: list, true_labels: np.ndarray):
    """
    Tests the clust_anomaly_fraud_detection function over a series of threshold values.

    Arguments:
        all_clusters (list): A list of lists of users (indices) in each cluster.
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
        predicted_labels = clust_anomaly_fraud_detection(all_clusters, anomaly_scores, threshold, true_labels)
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
            "Threshold": f"{threshold_values[i]:.1f}",
            "Accuracy": f"{result['accuracy']:.3f}",
            "Precision": f"{result['precision']:.3f}",
            "Recall": f"{result['recall']:.3f}",
            "F1 Score": f"{result['f1_score']:.3f}",
        })
    results_table.append({
        "Threshold": f"Best ({threshold_values[best_threshold]:.1f})",
        "Accuracy": f"{results[best_threshold]['accuracy']:.3f}",
        "Precision": f"{results[best_threshold]['precision']:.3f}",
        "Recall": f"{results[best_threshold]['recall']:.3f}",
        "F1 Score": f"{results[best_threshold]['f1_score']:.3f}",
    })
    utils.print_table(headers, results_table, logger.info)
    return results_table








