from pathlib import Path
import pandas as pd


RESULTS_DIRECTORY = Path(__file__).parent.parent.joinpath("results").resolve()


if __name__ == "__main__":
    file_path = RESULTS_DIRECTORY.joinpath('ANN_genetic_keras/ann_genetic_keras_test.csv')

    df = pd.read_csv(file_path, delimiter=',')
    best_params_df = df.sort_values(by=["mae","f1-score", "max_error"], ascending = [False, True, False])
    worst_params_df = df.sort_values(by=["mae","f1-score", "max_error"], ascending = [True, False, True])

    print("All results:")
    print(df)

    print("Best Parameters:")
    print(best_params_df)

    best_params_output_file_path = RESULTS_DIRECTORY.joinpath('ANN_genetic_keras/best_ann_genetic_keras_analyse.csv')
    best_params_df.to_csv(best_params_output_file_path, index=False)

    print("Worst Parameters:")
    print(best_params_df)

    worst_params_output_file_path = RESULTS_DIRECTORY.joinpath('ANN_genetic_keras/worst_ann_genetic_keras_analyse.csv')
    worst_params_df.to_csv(worst_params_output_file_path, index=False)

