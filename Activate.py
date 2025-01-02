import numpy as np

import pandas as pd

import ActivationFunctions as AF
# Define the main function to select and apply activation
def apply_activation(x, activation_id):
    """
    Applies the specified activation function to the input based on an ID.

    Parameters:
    x (numpy.ndarray): Input array or matrix.
    activation_id (int or str): The ID or name of the activation function.
        Supported values:
            0: ReLU
            1: Sigmoid
            2: Tanh
            3: Softmax
            4: Leaky ReLU
            5: ELU
            6: Swish
            7: GELU

    Returns:
    numpy.ndarray: Result of applying the activation function.
    """
    if activation_id == 0 or activation_id == "relu":
        return AF.relu(x)
    elif activation_id == 1 or activation_id == "sigmoid":
        return AF.sigmoid(x)
    elif activation_id == 2 or activation_id == "tanh":
        return AF.tanh(x)
    elif activation_id == 3 or activation_id == "softmax":
        return AF.softmax(x)
    elif activation_id == 4 or activation_id == "leaky_relu":
        return AF.leaky_relu(x, alpha=0.01)
    elif activation_id == 5 or activation_id == "elu":
        return AF.elu(x, alpha=1.0)
    elif activation_id == 6 or activation_id == "swish":
        return AF.swish(x)
    elif activation_id == 7 or activation_id == "gelu":
        return AF.gelu(x)
    else:
        raise ValueError(f"Unsupported activation ID: {activation_id}")


def apply_activation_derivative(x, activation_id):
    """
    Applies the derivative of the specified activation function to the input based on an ID.

    Parameters:
    x (numpy.ndarray): Input array or matrix.
    activation_id (int or str): The ID or name of the activation function.
        Supported values:
            0: ReLU
            1: Sigmoid
            2: Tanh
            3: Softmax (commonly computed during loss calculation)
            4: Leaky ReLU
            5: ELU
            6: Swish
            7: GELU

    Returns:
    numpy.ndarray: Result of applying the derivative of the activation function.
    """
    if activation_id == 0 or activation_id == "relu":
        return AF.relu_derivative(x)
    elif activation_id == 1 or activation_id == "sigmoid":
        return AF.sigmoid_derivative(x)
    elif activation_id == 2 or activation_id == "tanh":
        return AF.tanh_derivative(x)
    elif activation_id == 3 or activation_id == "softmax":
        raise NotImplementedError(
            "Softmax derivative is typically handled during loss calculation. Ensure proper implementation."
        )
    elif activation_id == 4 or activation_id == "leaky_relu":
        return AF.leaky_relu_derivative(x, alpha=0.01)
    elif activation_id == 5 or activation_id == "elu":
        return AF.elu_derivative(x, alpha=1.0)
    elif activation_id == 6 or activation_id == "swish":
        return AF.swish_derivative(x)
    elif activation_id == 7 or activation_id == "gelu":
        return AF.gelu_derivative(x)
    else:
        raise ValueError(f"Unsupported activation ID: {activation_id}")