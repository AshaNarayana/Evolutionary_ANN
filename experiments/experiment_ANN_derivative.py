
import time

from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

import numpy as np

from models.utils_metrics import metrics_regression
from models.ANN_derivative import train_network, predict_outputs


# Test implementation only using base dataset
DATA_SET_PATH = Path(__file__).parent.parent.joinpath("datasets/base_dataset").resolve()
RESULTS_DIRECTORY = Path(__file__).parent.parent.joinpath("results/ANN_derivative").resolve()


def ann_parameter_test(verbose: bool = False, test: bool = False) -> DataFrame:

    # BASE DATASET
    x_train = pd.read_csv(DATA_SET_PATH.joinpath("X_train.csv"))
    y_train = pd.read_csv(DATA_SET_PATH.joinpath("y_train.csv"))

    x_test = pd.read_csv(DATA_SET_PATH.joinpath("X_test.csv"))
    y_test = pd.read_csv(DATA_SET_PATH.joinpath("y_test.csv"))

    x_val = pd.read_csv(DATA_SET_PATH.joinpath("X_val.csv"))
    y_val = pd.read_csv(DATA_SET_PATH.joinpath("y_val.csv"))

    # ANN PARAMETERS
    learning_rate_range = np.arange(0.01, 0.05, 0.01)
    activation_functions = ["relu", "sigmoid"]

    # INITIALIZE FIXED PARAMETERS
    HL_neurons = 150
    output_neurons = 1
    input_HL_weights = np.random.uniform(low=-0.1, high=0.1, size=(x_train.shape[1], HL_neurons))  # Input to hidden
    HL_output_weights = np.random.uniform(low=-0.1, high=0.1, size=(HL_neurons, output_neurons))  # Hidden to output
    weights = [input_HL_weights, HL_output_weights]
    weights_info = []
    number_iterations_NN = 75

    # EMPTY RESULTS DSs
    column_names = ["learning_rate", "activation_function",
                    "mae", "mse", "max_error", "r2-score", "training_time", "test_time", "total_time"]
    results_df = pd.DataFrame(columns=column_names)

    # ITERATE
    max_iterations = len(learning_rate_range) * len(activation_functions)
    iteration = 1

    for learning_rate in learning_rate_range:
        for activation in activation_functions:
            print(f"{iteration} of {max_iterations} iterations")

            # Training
            training_time_start = time.time()
            trained_weights = train_network(number_iterations_NN, weights=weights,
                                            data_inputs=x_train, data_outputs=y_train,
                                            learning_rate=learning_rate, activation=activation)
            training_time_end = time.time()
            training_time = training_time_end - training_time_start

            # Testing
            test_time_start = time.time()
            predictions = predict_outputs(trained_weights, x_test, activation)
            test_time_end = time.time()
            test_time = test_time_end - test_time_start

            # Get metrics
            mae, mse, max_error, r2_score = metrics_regression(y_test.to_numpy(), predictions)

            # Save
            new_row_as_df = pd.Series({
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

    result_df = ann_parameter_test(True, True)
    output_file_name = f"{datetime.now()}.csv"

    # Modification for Windows users
    output_file_name = output_file_name.replace(":", "-")

    output_file_path = RESULTS_DIRECTORY.joinpath(output_file_name)
    result_df.to_csv(output_file_path, index=False)
