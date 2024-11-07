
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score


def metrics_report_regression(test_labels: np.ndarray, predictions: np.ndarray) -> dict:
    """
    Get some metric scores from predictions
    :param predictions: Output from classifiers (D instances)
    :param test_labels: True values from test sample (D instances)
    :return: Dictionary with various scores
    """
    metrics = { "mae": mean_absolute_error(test_labels, predictions),
                "mse": mean_squared_error(test_labels, predictions),
                "max_error": max_error(test_labels, predictions),
                "f1-score": r2_score(test_labels, predictions) }
    return metrics