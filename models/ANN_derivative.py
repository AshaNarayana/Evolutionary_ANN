import time
from pathlib import Path

import numpy as np

import pandas as pd

from models.utils_metrics import metrics_report_regression, save_results_to_csv

np.set_printoptions(suppress=True)
DATA_SET_PATH = Path(__file__).parent.parent.joinpath("datasets").resolve()
learning_rate_range = np.arange(0.01, 0.05, 0.01)
activation_functions = ["relu", "sigmoid"]
number_iterations = 75
def sigmoid(inpt):
    """Sigmoid activation function."""
    return 1.0 / (1 + np.exp(-np.clip(inpt, -500, 500)))


def relu(inpt):
    """ReLU activation function."""
    result = np.copy (inpt)
    result [inpt < 0] = 0
    return result

def update_weights(weights, gradients, learning_rate):
    """Update weights using gradient descent."""
    new_weights = weights - learning_rate * gradients
    return new_weights

max_iterations = len(learning_rate_range) * len(activation_functions)
def train_network(iteration, weights, data_inputs, data_outputs, learning_rate, activation="relu"):
    print (f"{iteration} of {max_iterations} iterations")
    for _ in range (number_iterations):


        for sample_idx in range (data_inputs.shape [0]):
            r1 = data_inputs.iloc[sample_idx, :]
            # Pass through the hidden layer (single layer)
            hidden_input = np.matmul(r1, weights[0])  # First layer weights
            if activation == "relu":
                hidden_output = relu(hidden_input)
            elif activation == "sigmoid":
                hidden_output = sigmoid(hidden_input)
            # Output layer
            output_input = np.matmul(hidden_output, weights[1])  # Second layer weights (output)
            predicted_label = output_input  # Predicted value (regression)

            # Desired label
            desired_label = data_outputs.iloc[sample_idx, 0]


            if predicted_label != desired_label:
                # Calculate error at output
                output_error = output_input - desired_label

                # Calculate gradient for weights[1]
                if activation == "relu":
                    hidden_output_derivative = hidden_output > 0
                elif activation == "sigmoid":
                    hidden_output_derivative = hidden_output * (1 - hidden_output)
                gradient_weights_1 = np.outer(hidden_output, output_error)

                # Calculate error at hidden layer
                hidden_error = np.dot (output_error, weights [1].T) * hidden_output_derivative
                hidden_error = np.nan_to_num (hidden_error)  # Ensure no invalid values

                # Calculate gradient for weights[0]
                gradient_weights_0 = np.outer(r1, hidden_error)

                # Update weights
                weights[0] = update_weights(weights[0], gradient_weights_0, learning_rate)
                weights[1] = update_weights(weights[1], gradient_weights_1, learning_rate)


    return weights


def predict_outputs(weights, test_df, activation="relu"):
    """Make predictions using the trained network."""
    predictions = np.zeros (shape=(test_df.shape [0]))
    print(test_df)
    for sample_idx in range (test_df.shape [0]):
        r1 = test_df.iloc[sample_idx, :]
        r1 = np.matmul (r1, weights [0])
        if activation == "relu":
            r1 = relu (r1)
        elif activation == "sigmoid":
            r1 = sigmoid (r1)

        r1 = np.matmul (r1, weights [1])
        predicted_label = r1
        predictions[sample_idx] = predicted_label

    return predictions


train_data_file_path = DATA_SET_PATH.joinpath("X_train.csv")

train_data = pd.read_csv(train_data_file_path)

features_STDs = np.std (train_data, axis=0) # Feature selection based on std deviation

output_labels_file_path = DATA_SET_PATH.joinpath("y_train.csv")

# Load dataset outputs (true labels)
data_outputs = pd.read_csv(output_labels_file_path)

# Define the architecture: single hidden layer with neurons
HL_neurons = 150  # Number of neurons in the hidden layer
output_neurons = 1  # Number of output neurons (for regression)

# Initialize weights (input to hidden, hidden to output)
input_HL_weights = np.random.uniform (low=-0.1, high=0.1, size=(train_data.shape [1], HL_neurons))  # Input to hidden
HL_output_weights = np.random.uniform (low=-0.1, high=0.1, size=(HL_neurons, output_neurons))  # Hidden to output

# Store weights in a list
weights = [input_HL_weights, HL_output_weights]
weights_info = []
# Train the network
start_train_time = time.time ()
iteration = 1
for learning_rate in learning_rate_range:
    for activation in activation_functions:

        trained_weights = train_network(iteration, weights=weights, data_inputs=train_data, data_outputs=data_outputs,
                                        learning_rate=learning_rate, activation=activation)
        weights_info.append((trained_weights, learning_rate, activation))
        iteration += 1

end_train_time = time.time ()
training_time = end_train_time - start_train_time

# Make predictions
test_data_file_path = DATA_SET_PATH.joinpath("X_test.csv")

test_data = pd.read_csv(test_data_file_path)

test_label_file_path = DATA_SET_PATH.joinpath("y_test.csv")

test_label = pd.read_csv(test_label_file_path)

test_label_array = test_label.to_numpy()
results = []
for trained_weights, learning_rate, activation in weights_info:
    start_test_time = time.time()
    predictions = predict_outputs(trained_weights, test_data, activation)

    end_test_time = time.time()

    testing_time = end_test_time - start_test_time

    mae, mse, max_err, r2 = metrics_report_regression (test_label_array, predictions)
    results.append ({
        "learning_rate": learning_rate,
        "activation_function": activation,
        "test_time": testing_time,
        "training_time": training_time,
        "mse": mse,
        "mae": mae,
        "max_error": max_err,
        "r2": r2
    })
# Calculate averages
avg_mse = np.mean([result["mse"] for result in results])
avg_mae = np.mean([result["mae"] for result in results])
avg_max_error = np.mean([result["max_error"] for result in results])
avg_r2 = np.mean([result["r2"] for result in results])

print(f"Average MSE: {avg_mse}")
print(f"Average MAE: {avg_mae}")
print(f"Average Max Error: {avg_max_error}")
print(f"Average R2: {avg_r2}")

# Save results to CSV
results_file_path = DATA_SET_PATH.joinpath("Ann_derivative_results.csv")
save_results_to_csv(results, results_file_path)

