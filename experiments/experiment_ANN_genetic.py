
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from models.ANN_genetic_keras import ANN_Genetic_keras
from models.utils_metrics import metrics_regression
from models.utils_models import default_parameters_ga

# Test implementation using ONLY base dataset
DATA_SET_PATH = Path(__file__).parent.parent.joinpath("datasets/base_dataset").resolve()

# Save path
RESULTS_DIRECTORY = Path(__file__).parent.parent.joinpath("results/ANN_genetic").resolve()


def gann_parameter_test(verbose: bool = False, test: bool = False) -> DataFrame:

    # BASE DATASET
    x_train = pd.read_csv(DATA_SET_PATH.joinpath("X_train.csv"))
    y_train = pd.read_csv(DATA_SET_PATH.joinpath("y_train.csv"))

    x_test = pd.read_csv(DATA_SET_PATH.joinpath("X_test.csv"))
    y_test = pd.read_csv(DATA_SET_PATH.joinpath("y_test.csv"))

    x_val = pd.read_csv(DATA_SET_PATH.joinpath("X_val.csv"))
    y_val = pd.read_csv(DATA_SET_PATH.joinpath("y_val.csv"))

    # GA PARAMETERS: TO SPECIFY
    if test:
        parents_mating = []
        mutation_percentages = ["default"]
        parent_selection = []
        crossover = []
        mutation = []
        keep_parents = []
    else:
        parents_mating = []
        mutation_percentages = ["default"]
        parent_selection = []
        crossover = []
        mutation = []
        keep_parents = []


    # EMPTY RESULTS DSs
    column_names = ["dataset", "learning_rate", "mutation_percentages", "parent_selection", "crossover", "mutation", "keep_parents",
                    "mae", "mse", "max_error", "f1-score", "training_time", "test_time", "total_time"]
    results_df = pd.DataFrame(columns=column_names)

    # ITERATE
    max_iterations = (len(parents_mating) * len(mutation_percentages) * len(parent_selection) *
                      len(crossover) * len(mutation) * len(keep_parents))
    iteration = 1

    for parents in parents_mating:
        for p_mutation in mutation_percentages:
            for select in parent_selection:
                for cross in crossover:
                    for mutate in mutation:
                        for keep in keep_parents:
                            if verbose:
                                print(f"Iteration {iteration} of {max_iterations} "
                                      f"[{round(100 * iteration / max_iterations, 2)}]")

                            # Prepare model
                            parameters_GA = default_parameters_ga()
                            parameters_GA["num_parents_mating"] = parents
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
                                "parents_mating": parents,
                                "mutation_percentages": p_mutation,
                                "parent_selection": select,
                                "crossover": cross,
                                "mutation": mutate,
                                "keep_parents": keep,
                                "mae": mae,
                                "mse": mse,
                                "max_error": max_error,
                                "r2-score": f1_score,
                                "training_time": training_time,
                                "test_time": test_time,
                                "total_time": training_time + test_time,
                            })
                            results_df.loc[len(results_df)] = new_row_as_df
                            iteration += 1
    return results_df


if __name__ == "__main__":
    result_df = gann_parameter_test(True, True)
    output_file_name = f"{datetime.now()}.csv"

    # Modification for Windows users
    output_file_name = output_file_name.replace(":", "-")

    output_file_path = RESULTS_DIRECTORY.joinpath(output_file_name)
    result_df.to_csv(output_file_path, index=False)
