
from pathlib import Path
import time
import os

import pandas as pd
from pandas import DataFrame

import numpy as np


from models.ANN_derivative_keras import ANN_Keras
from models.utils_metrics import metrics_regression


# Test implementation using three datasets
DATA_SET_PATH_1 = Path(__file__).parent.parent.joinpath("datasets/base_dataset").resolve()
DATA_SET_PATH_2 = Path(__file__).parent.parent.joinpath("datasets/larger_dataset").resolve()
DATA_SET_PATH_3 = Path(__file__).parent.parent.joinpath("datasets/noisy_dataset").resolve()

# Save path
RESULTS_DIRECTORY = Path(__file__).parent.parent.joinpath("results/ANN_derivative_keras").resolve()


def ann_keras_parameter_test(verbose: bool = False, test: bool = False) -> DataFrame:

    # DATASETS
    datasets = []
    for dataset_paths in [DATA_SET_PATH_1, DATA_SET_PATH_2, DATA_SET_PATH_3]:
        x_train = pd.read_csv(dataset_paths.joinpath("X_train.csv"))
        y_train = pd.read_csv(dataset_paths.joinpath("y_train.csv"))
        x_test = pd.read_csv(dataset_paths.joinpath("X_test.csv"))
        y_test = pd.read_csv(dataset_paths.joinpath("y_test.csv"))
        x_val = pd.read_csv(dataset_paths.joinpath("X_val.csv"))
        y_val = pd.read_csv(dataset_paths.joinpath("y_val.csv"))
        datasets.append([dataset_paths, x_train, y_train, x_test, y_test, x_val, y_val])


    learning_rate_range = np.arange(0.01, 0.05, 0.01)
    activation_functions_hidden = ["relu", "sigmoid"]

    # EMPTY RESULTS DSs
    column_names = ["dataset", "learning_rate", "activation_function",
                    "mae", "mse", "max_error", "r2-score", "training_time", "test_time", "total_time"]
    results_df = pd.DataFrame(columns=column_names)

    # ITERATE
    max_iterations = (len(datasets) * len(learning_rate_range) * len(activation_functions_hidden))
    iteration = 1

    for path, x_train, y_train, x_test, y_test, x_val, y_val in datasets:
        for learning_rate in learning_rate_range:
            for activation in activation_functions_hidden:
                print(f"{iteration} of {max_iterations} iterations")

                ann_keras = ANN_Keras(x_train, y_train, x_val, y_val,
                                      learning_rate=learning_rate, activation_hidden=activation)

                # Training
                training_time_start = time.time()
                ann_keras.fit()
                training_time_end = time.time()
                training_time = training_time_end - training_time_start
    
                # Testing
                test_time_start = time.time()
                predictions = ann_keras.predict(x_test)
                test_time_end = time.time()
                test_time = test_time_end - test_time_start
    
                # Get metrics
                mae, mse, max_error, r2_score = metrics_regression(y_test.to_numpy(), predictions)
    
                # Save
                new_row_as_df = pd.Series({
                    "dataset": os.path.basename(path),
                    "learning_rate": learning_rate,
                    "activation_function": activation,
                    "mae": mae,
                    "mse": mse,
                    "max_error": max_error,
                    "r2-score": r2_score,
                    "training_time": training_time,
                    "test_time": test_time,
                    "total_time": training_time + test_time,
                })
                results_df.loc[len(results_df)] = new_row_as_df
                iteration += 1
    return results_df


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    result_df = ann_keras_parameter_test(True, True)
    output_file_name = "ann_derivative_keras_test.csv"

    # Modification for Windows users
    output_file_name = output_file_name.replace(":", "-")

    output_file_path = RESULTS_DIRECTORY.joinpath(output_file_name)
    result_df.to_csv(output_file_path, index=False)
