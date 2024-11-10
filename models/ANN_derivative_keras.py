import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from models.utils_metrics import regression_report, save_results_to_csv


class ANN_Keras:
    def __init__(self, train_data, train_outputs, valid_data, valid_outputs):
        """

        :param train_data:
        :param train_outputs:
        :param valid_data:
        :param valid_outputs:
        """
        self.train_data = train_data
        self.train_outputs = train_outputs
        self.valid_data = valid_data
        self.valid_outputs = valid_outputs

        # Architecture
        self.model = Sequential()
        self.model.add(Dense(150, input_dim=self.train_data.shape[1], activation='relu'))  # Hidden layer with 150 neurons
        self.model.add(Dense(1))  # Output layer for regression

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='mean_squared_error',
            metrics=[MeanSquaredError(), MeanAbsoluteError()]
        )

        # Other parameters
        self.history = None
        self.results = None

    def fit(self, epochs=50, batch_size=32):
        """

        :param epochs:
        :param batch_size:
        :return:
        """
        self.history = self.model.fit( self.train_data, self.train_outputs,
                                       validation_data=(self.valid_data, self.valid_outputs),
                                       epochs=epochs, batch_size=batch_size, verbose=1 )


    def predict(self, test_data):
        """

        :param test_data:
        :return:
        """
        return self.model.predict(test_data)

    def evaluate(self, test, predictions):
        """

        :param test:
        :param predictions:
        :return:
        """
        self.results = regression_report(test.to_numpy(), predictions)
        return self.results

    def save_results(self, results_file_path):
        """

        """
        save_results_to_csv(self.results, results_file_path)

    def print_results(self):
        """

        """
        print("MSE:", self.results["mse"])
        print("MAE:", self.results["mae"])
        print("Max Error:", self.results["max_error"])
        print("R2:", self.results["r2"])



# Usage example
if __name__ == "__main__":

    # Read example data
    DATA_SET_PATH = Path(__file__).parent.parent.joinpath("datasets/base_dataset").resolve()

    train_data_file_path = DATA_SET_PATH.joinpath("X_train.csv")
    output_labels_file_path = DATA_SET_PATH.joinpath("y_train.csv")
    test_data_file_path = DATA_SET_PATH.joinpath("X_test.csv")
    test_label_file_path = DATA_SET_PATH.joinpath("y_test.csv")
    valid_data_file_path = DATA_SET_PATH.joinpath("X_val.csv")
    valid_label_file_path = DATA_SET_PATH.joinpath("y_val.csv")

    train_data = pd.read_csv(train_data_file_path)
    data_outputs = pd.read_csv(output_labels_file_path)
    test_data = pd.read_csv(test_data_file_path)
    test_label = pd.read_csv(test_label_file_path)
    valid_data = pd.read_csv(valid_data_file_path)
    valid_label = pd.read_csv(valid_label_file_path)

    # Create instance
    ann_keras = ANN_Keras(train_data, data_outputs, valid_data, valid_label)

    # Train
    ann_keras.fit()

    # Test
    predictions = ann_keras.predict(test_data)
    print (predictions)

    # Evaluate metrics
    ann_keras.evaluate(test_label, predictions)
    ann_keras.print_results()