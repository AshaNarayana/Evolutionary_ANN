from pathlib import Path

import numpy as np

import pandas as pd
np.set_printoptions(suppress=True)
DATA_SET_PATH = Path(__file__).parent.parent.joinpath("data").joinpath("datasets").resolve()
def sigmoid(inpt):
    """Sigmoid activation function."""
    return 1.0 / (1 + np.exp (-inpt))


"""
______________________________________________________________
TODO : learning rate should be in range of 0.01 to 0.1 to find best value
2. test for both activation functions
3. maybe we could reduce training data to save the time
4. HL_neurons value could be between 150 to 200. Unless required or else we can leave it
To be discussed
______________________________________________________________
"""



def relu(inpt):
    """ReLU activation function."""
    result = np.copy (inpt)
    result [inpt < 0] = 0
    return result


def update_weights(weights, learning_rate):
    """Update weights using simple gradient descent."""
    new_weights = weights - learning_rate * weights
    return new_weights


def train_network(num_iterations, weights, data_inputs, data_outputs, learning_rate, activation="relu"):
    for iteration in range (num_iterations):
        print ("Iteration", iteration)
        for sample_idx in range (data_inputs.shape [0]):


            # Get the input features for the current sample
            r1 = data_inputs.iloc[sample_idx, :]
            # Pass through the hidden layer (single layer)
            r1 = np.matmul (r1, weights [0])  # First layer weights
            if activation == "relu":
                r1 = relu (r1)
            elif activation == "sigmoid":
                r1 = sigmoid (r1)

            # Output layer
            r1 = np.matmul (r1, weights [1])  # Second layer weights (output)
            predicted_label = np.where (r1 == np.max (r1)) [0] [0]  # Predicted label (classification)

            # Desired label
            desired_label = data_outputs.iloc[sample_idx, 0]

            if predicted_label != desired_label:
                # Update weights if the prediction is incorrect
                weights [0] = update_weights (weights [0], learning_rate)
                weights [1] = update_weights (weights [1], learning_rate)

    return weights


def predict_outputs(weights, test_df, activation="relu"):
    """Make predictions using the trained network."""
    predictions = np.zeros (shape=(test_df.shape [0]))
    for sample_idx in range (test_df.shape [0]):
        r1 = test_df.iloc[sample_idx, :]
        r1 = np.matmul (r1, weights [0])
        if activation == "relu":
            r1 = relu (r1)
        elif activation == "sigmoid":
            r1 = sigmoid (r1)

        r1 = np.matmul (r1, weights [1])
        predicted_label = np.where (r1 == np.max (r1)) [0] [0]
        predictions [sample_idx] = predicted_label

    return predictions

train_data_file_path = DATA_SET_PATH.joinpath("X_train.csv")

train_data = pd.read_csv(train_data_file_path)

features_STDs = np.std (train_data, axis=0) # Feature selection based on std deviation

output_labels_file_path = DATA_SET_PATH.joinpath("y_train.csv")

# Load dataset outputs (true labels)
data_outputs = pd.read_csv(output_labels_file_path)

# Define the architecture: single hidden layer with neurons
HL_neurons = 150  # Number of neurons in the hidden layer
output_neurons = 4  # Number of output neurons (for classification)

# Initialize weights (input to hidden, hidden to output)
input_HL_weights = np.random.uniform (low=-0.1, high=0.1, size=(train_data.shape [1], HL_neurons))  # Input to hidden
HL_output_weights = np.random.uniform (low=-0.1, high=0.1, size=(HL_neurons, output_neurons))  # Hidden to output

# Store weights in a list
weights = [input_HL_weights, HL_output_weights]

# Train the network
weights = train_network (num_iterations=100, weights=weights, data_inputs=train_data, data_outputs=data_outputs,
                         learning_rate=0.02, activation="relu")


# Make predictions
test_data_file_path = DATA_SET_PATH.joinpath("X_test.csv")

test_data = pd.read_csv(test_data_file_path)

test_label_file_path = DATA_SET_PATH.joinpath("y_test.csv")

test_label = pd.read_csv(test_label_file_path)
predictions = predict_outputs (weights, test_data, "sigmoid")
# Evaluate performance


test_label_array = test_label.to_numpy()
# Calculate the number of correct and incorrect predictions
num_false = np.sum(predictions != test_label_array)
num_true = np.sum(predictions == test_label_array)

print("Number of incorrect predictions:", num_false)
print("Number of correct predictions:", num_true)

