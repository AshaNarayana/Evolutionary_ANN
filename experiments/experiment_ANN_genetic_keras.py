
import time
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from models.ANN_genetic_keras import ANN_Genetic_keras
from models.utils_metrics import metrics_regression
from models.utils_models import default_parameters_ga_keras

# Test implementation using three datasets
DATA_SET_PATH_1 = Path(__file__).parent.parent.joinpath("datasets/base_dataset").resolve()
DATA_SET_PATH_2 = Path(__file__).parent.parent.joinpath("datasets/larger_dataset").resolve()
DATA_SET_PATH_3 = Path(__file__).parent.parent.joinpath("datasets/noisy_dataset").resolve()

# Save path
RESULTS_DIRECTORY = Path(__file__).parent.parent.joinpath("results/ANN_genetic_keras").resolve()


def gann_keras_parameter_test(verbose: bool = False, test: bool = False) -> DataFrame:

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

    # GA PARAMETERS
    if test:
        mutation_percentages = ["default"]
        parent_selection = ["sss", "sus"]
        crossover = ["single_point"]
        mutation = ["random"]
        keep_parents = [-1]
    else:
        mutation_percentages = ["default", 0.25]
        parent_selection = ["sss", "sus", "rank", "tournament"]
        crossover = ["single_point", "uniform"]
        mutation = ["random", "scramble"]
        keep_parents = [-1, 0] # All or none


    # EMPTY RESULTS DSs
    column_names = ["dataset", "mutation_percentages", "parent_selection", "crossover", "mutation", "keep_parents",
                    "mae", "mse", "max_error", "f1-score", "training_time", "test_time", "total_time"]
    results_df = pd.DataFrame(columns=column_names)

    # ITERATE
    max_iterations = (len(datasets) * len(mutation_percentages) * len(parent_selection) *
                      len(crossover) * len(mutation) * len(keep_parents))
    iteration = 1

    for path, x_train, y_train, x_test, y_test, x_val, y_val in datasets:
        for p_mutation in mutation_percentages:
            for select in parent_selection:
                for cross in crossover:
                    for mutate in mutation:
                        for keep in keep_parents:
                            if verbose:
                                print(f"Iteration {iteration} of {max_iterations} "
                                      f"[{round(100 * iteration / max_iterations, 2)}]")

                            # Prepare model
                            parameters_GA = default_parameters_ga_keras()
                            parameters_GA["mutation_percent_genes"] = p_mutation
                            parameters_GA["parent_selection_type"] = select
                            parameters_GA["crossover_type"] = cross
                            parameters_GA["mutation_type"] = mutate
                            parameters_GA["keep_parents"] = keep

                            gann_keras = ANN_Genetic_keras(x_train, y_train, x_val, y_val, parameters_GA)

                            # Training
                            training_time_start = time.time()
                            gann_keras.fit()
                            training_time_end = time.time()
                            training_time = training_time_end - training_time_start

                            # Testing
                            test_time_start = time.time()
                            prediction = gann_keras.predict(x_test)
                            test_time_end = time.time()
                            test_time = test_time_end - test_time_start

                            # Get metrics
                            mae, mse, max_error, r2_score = metrics_regression(y_test, prediction)

                            # Save
                            new_row_as_df = pd.Series({
                                "dataset": os.path.basename(path),
                                "mutation_percentages": p_mutation,
                                "parent_selection": select,
                                "crossover": cross,
                                "mutation": mutate,
                                "keep_parents": keep,
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
    result_df = gann_keras_parameter_test(True, True)
    output_file_name = f"{datetime.now()}.csv"

    # Modification for Windows users
    output_file_name = output_file_name.replace(":", "-")

    output_file_path = RESULTS_DIRECTORY.joinpath(output_file_name)
    result_df.to_csv(output_file_path, index=False)
