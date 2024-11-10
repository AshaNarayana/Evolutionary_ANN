
import numpy as np


"""
Activation functions
"""

def sigmoid(inpt):
    """
    Sigmoid activation function.
    :param inpt:
    :return:
    """
    return 1.0 / (1 + np.exp(-np.clip(inpt, -500, 500)))

def relu(inpt):
    """
    ReLU activation function.
    :param inpt:
    :return:
    """
    result = np.copy (inpt)
    result [inpt < 0] = 0
    return result



"""
Derivative ANN methods
"""

def update_weights(weights, gradients, learning_rate):
    """
    Update weights using gradient descent.
    :param weights:
    :param gradients:
    :param learning_rate:
    :return:
    """
    new_weights = weights - learning_rate * gradients
    return new_weights


def train_network(number_iterations, weights, data_inputs, data_outputs, learning_rate, activation="relu"):
    """

    :param number_iterations:
    :param weights:
    :param data_inputs:
    :param data_outputs:
    :param learning_rate:
    :param activation:
    :return:
    """
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
    """
    Make predictions using the trained network.
    :param weights:
    :param test_df:
    :param activation:
    :return:
    """
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