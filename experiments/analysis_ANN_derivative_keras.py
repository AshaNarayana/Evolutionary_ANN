
from pathlib import Path
import pandas as pd


RESULTS_DIRECTORY = Path(__file__).parent.parent.joinpath("results").resolve()


if __name__ == "__main__":
    file_path = RESULTS_DIRECTORY.joinpath('ANN_derivative_keras/ann_derivative_keras_test.csv')
    df = pd.read_csv(file_path, delimiter=',')
    print(df)
    best_params = df.loc[df['mae'].idxmin()] if 'mae' in df.columns else None
    worst_params = df.loc[df['mae'].idxmax()] if 'mae' in df.columns else None

    print("Best Parameters based on MAE:")
    print(best_params)
    dataset = best_params['dataset']


