from typing import Union, Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score


def metrics_regression(test_labels: np.ndarray, predictions: np.ndarray) -> tuple [
    Union [float, Any], Union [float, Any], float, dict [str, Union [float, Any]]]:
    """
    Get some metric scores from predictions
    :param predictions: Output from classifiers (D instances)
    :param test_labels: True values from test sample (D instances)
    :return: Metrics
    """
    mae = mean_absolute_error(test_labels, predictions)
    mse = mean_squared_error(test_labels, predictions)
    max_err = max_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)
    return mae, mse, max_err, r2


def regression_report(test_labels: np.ndarray, predictions: np.ndarray) -> Dict:
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



def save_results_to_csv(results, file_path):
    """
    Save the past_results to a CSV file.

    :param results: List of dictionaries containing the past_results.
    :param file_path: Path to the CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
