import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """
    Tests a binary classifier given the labels and predictions as np arrays.

    Args:
        labels (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Predicted labels.

    Returns:
        dict: A dictionary containing the accuracy, precision, recall, and F1 score.
    """
    assert labels.shape == predictions.shape, "Labels and predictions must have the same shape."
    assert len(labels.shape) == 1, "Labels and predictions must be 1D arrays."
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # Store results in a dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    
    return results

