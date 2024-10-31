import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 1000  # Total number of samples
n_features = 1    # Number of features
noise_level = 10  # Amount of noise (hyperparameter)

# Generate synthetic regression data
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise_level)

# Split the data into training, validation, and testing sets
# We can remove validation if not needed
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% val, 15% test

# Checking the sizes of the datasets
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Visualize the synthetic data
plt.scatter(X, y, color='blue', label='Synthetic Data')
plt.title('Synthetic Regression Data')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()