from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Define datasets directory
DATASETS_DIRECTORY = Path(__file__).parent.joinpath('datasets').resolve()

class SyntheticData(Enum):
    NUMBER_OF_SAMPLES = 1000
    NUMBER_OF_FEATURES = 4
    NOISE_LEVEL = 10

    @staticmethod
    def generate_synthetic_data():
        """
        Generate synthetic regression data, split into training and testing sets, and save to a CSV file.
        """
        # Generate synthetic regression data
        X_FEATURES, y = make_regression(
            n_samples=SyntheticData.NUMBER_OF_SAMPLES.value,
            n_features=SyntheticData.NUMBER_OF_FEATURES.value,
            noise=SyntheticData.NOISE_LEVEL.value
        )

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_FEATURES, y, test_size=0.3, random_state=42
        )

        # Create DataFrame for saving
        train_df = pd.DataFrame (X_train, columns=[f'feature_{i}' for i in
                                                   range (1, SyntheticData.NUMBER_OF_FEATURES.value + 1)])

        train_df['label'] = y_train
        test_df = pd.DataFrame (X_test, columns=[f'feature_{i}' for i in
                                                   range (1, SyntheticData.NUMBER_OF_FEATURES.value + 1)])

        test_df['label'] = y_test

        # Save training and testing sets to CSV files
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not DATASETS_DIRECTORY.exists ():
            DATASETS_DIRECTORY.mkdir ()
        train_df.to_csv(DATASETS_DIRECTORY.joinpath(f"train_data_{timestamp}.csv"), index=False)
        test_df.to_csv(DATASETS_DIRECTORY.joinpath(f"test_data_{timestamp}.csv"), index=False)


# Main function to run the code
if __name__ == "__main__":
    SyntheticData.generate_synthetic_data()
