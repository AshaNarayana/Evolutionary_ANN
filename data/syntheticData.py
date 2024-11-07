from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class SyntheticDataGenerator:
    def __init__(self, num_samples: int = 2000, num_features: int = 4, noise_level: float = 2.0, datasets_dir: str = 'datasets'):
        self.num_samples = num_samples
        self.num_features = num_features
        self.noise_level = noise_level
        self.datasets_directory = Path(__file__).parent.joinpath(datasets_dir).resolve()
        self.datasets_directory.mkdir(exist_ok=True)

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic regression data with specified parameters.

        :return: Tuple of feature matrix X and target vector y.
        """
        X, y = make_regression(
            n_samples=self.num_samples,
            n_features=self.num_features,
            noise=self.noise_level,
            random_state=42
        )

        feature_range = (-1, 1)
        target_range = (0, 3)

        # Scale features to the desired range
        scaler_X = MinMaxScaler (feature_range=feature_range)
        X = scaler_X.fit_transform (X)

        # Scale target labels to the desired range
        scaler_y = MinMaxScaler (feature_range=target_range)
        y = scaler_y.fit_transform (y.reshape (-1, 1)).flatten ()
        X = np.round (X, 1)
        y = np.round (y, 1)
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, training_size: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation, and testing sets.

        :param X: Feature matrix.
        :param y: Target vector.
        :param training_size: Proportion of data to use for training.
        :return: Split data arrays for training, validation, and testing.
        """
        # Initial split to get a large test set (e.g., 30% of total data)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Split the temp set into training and validation with variable training set size
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, train_size=training_size, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_to_csv(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """
        Save datasets to CSV files.

        :param X_train: Training feature matrix.
        :param X_val: Validation feature matrix.
        :param X_test: Testing feature matrix.
        :param y_train: Training target vector.
        :param y_val: Validation target vector.
        :param y_test: Testing target vector.
        """
        # Save each dataset split
        pd.DataFrame(X_train).to_csv(self.datasets_directory / 'X_train.csv', index=False)
        pd.DataFrame(y_train).to_csv(self.datasets_directory / 'y_train.csv', index=False)
        pd.DataFrame(X_val).to_csv(self.datasets_directory / 'X_val.csv', index=False)
        pd.DataFrame(y_val).to_csv(self.datasets_directory / 'y_val.csv', index=False)
        pd.DataFrame(X_test).to_csv(self.datasets_directory / 'X_test.csv', index=False)
        pd.DataFrame(y_test).to_csv(self.datasets_directory / 'y_test.csv', index=False)

# Example usage
if __name__ == "__main__":
    generator = SyntheticDataGenerator(num_samples=2000, num_features=4, noise_level=10)
    X, y = generator.generate_data()
    X_train, X_val, X_test, y_train, y_val, y_test = generator.split_data(X, y, training_size=0.5)
    generator.save_to_csv(X_train, X_val, X_test, y_train, y_val, y_test)
