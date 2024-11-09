import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from models.utils_metrics import metrics_report_regression, save_results_to_csv

class ANN_Keras:
    def __init__(self, train_data, data_outputs, test_data, test_label):
        self.train_data = train_data
        self.data_outputs = data_outputs
        self.test_data = test_data
        self.test_label = test_label

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.train_data, self.data_outputs, test_size=0.2, random_state=1
        )

        self.model = Sequential()
        self.model.add(Dense(150, input_dim=self.x_train.shape[1], activation='relu'))  # Hidden layer with 150 neurons
        self.model.add(Dense(1))  # Output layer for regression

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='mean_squared_error',
            metrics=[MeanSquaredError(), MeanAbsoluteError()]
        )

    def train(self, epochs=50, batch_size=32):
        start_train_time = time.time()
        self.history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=epochs, batch_size=batch_size, verbose=1
        )
        end_train_time = time.time()
        self.training_time = end_train_time - start_train_time

    def evaluate(self):
        start_test_time = time.time()
        self.predictions = self.model.predict(self.test_data)
        end_test_time = time.time()
        self.testing_time = end_test_time - start_test_time

        # Calculate metrics
        self.mae, self.mse, self.max_err, self.r2 = metrics_report_regression(
            self.test_label.to_numpy(), self.predictions
        )
        self.results = [{
            "test_time": self.testing_time,
            "training_time": self.training_time,
            "mse": self.mse,
            "mae": self.mae,
            "max_error": self.max_err,
            "r2": self.r2
        }]

    def save_results(self, results_file_path):
        save_results_to_csv(self.results, results_file_path)

    def print_results(self):
        print(f"Training time: {self.training_time}")
        print(f"Testing time: {self.testing_time}")
        print(f"MSE: {self.mse}")
        print(f"MAE: {self.mae}")
        print(f"Max Error: {self.max_err}")
        print(f"R2: {self.r2}")

# Usage example
if __name__ == "__main__":
    DATA_SET_PATH = Path(__file__).parent.parent.joinpath("datasets").resolve()
    train_data_file_path = DATA_SET_PATH.joinpath("X_train.csv")
    output_labels_file_path = DATA_SET_PATH.joinpath("y_train.csv")
    test_data_file_path = DATA_SET_PATH.joinpath("X_test.csv")
    test_label_file_path = DATA_SET_PATH.joinpath("y_test.csv")

    train_data = pd.read_csv(train_data_file_path)
    data_outputs = pd.read_csv(output_labels_file_path)
    test_data = pd.read_csv(test_data_file_path)
    test_label = pd.read_csv(test_label_file_path)

    ann_keras = ANN_Keras(train_data, data_outputs, test_data, test_label)
    ann_keras.train()
    ann_keras.evaluate()
    results_file_path = DATA_SET_PATH.joinpath("Ann_keras_results.csv")
    ann_keras.save_results(results_file_path)
    ann_keras.print_results()