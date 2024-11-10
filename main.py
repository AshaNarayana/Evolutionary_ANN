
from pathlib import Path
import pandas as pd

from models.ANN_genetic_keras import ANN_Genetic_keras
from models.utils_models import default_parameters_ga_keras

# Test implementation using ONLY base dataset
DATA_SET_PATH = Path(__file__).parent.joinpath("datasets/base_dataset").resolve()


if __name__ == '__main__':

    # Dataset BASE
    x_train = pd.read_csv(DATA_SET_PATH.joinpath("X_train.csv"))
    y_train = pd.read_csv(DATA_SET_PATH.joinpath("y_train.csv"))

    x_test = pd.read_csv(DATA_SET_PATH.joinpath("X_test.csv"))
    y_test = pd.read_csv(DATA_SET_PATH.joinpath("y_test.csv"))

    x_val = pd.read_csv(DATA_SET_PATH.joinpath("X_val.csv"))
    y_val = pd.read_csv(DATA_SET_PATH.joinpath("y_val.csv"))


    print("BEST GANN implementation")

    parameters_GA = default_parameters_ga_keras()
    parameters_GA["parent_selection_type"] = "sus"
    parameters_GA["crossover_type"] = "single_point"
    parameters_GA["mutation_type"] = "random"
    parameters_GA["keep_parents"] = -1

    # Genetic ANN
    gann_keras = ANN_Genetic_keras(x_train, y_train, x_val, y_val, parameters_GA)

    # Training
    gann_keras.fit()
    gann_keras.ga_plots()

    # Testing
    predictions = gann_keras.predict(x_test)

    # Results
    print(y_test.shape)
    print(predictions.shape)