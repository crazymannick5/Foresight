import pandas as pd
import numpy as np


import numpy as np

def relu(x):
    """
    Applies the ReLU activation function element-wise.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Output with ReLU applied.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of the ReLU function.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Derivative of ReLU, where 1 for x > 0, else 0.
    """
    return (x > 0).astype(float)


def sigmoid(x):
    """
    Applies the sigmoid activation function element-wise.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Output with sigmoid applied.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Derivative of sigmoid.
    """
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """
    Applies the tanh activation function element-wise.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Output with tanh applied.
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of the tanh function.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Derivative of tanh.
    """
    return 1 - np.tanh(x)**2
def softmax(x):
    """
    Applies the softmax activation function to a matrix or vector.

    Parameters:
    x (numpy.ndarray): Input array or matrix. Should be 2D for batch operations.

    Returns:
    numpy.ndarray: Output with softmax applied along the last axis.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Softmax derivative is often computed directly during backward propagation
def leaky_relu(x, alpha=0.01):
    """
    Applies the Leaky ReLU activation function element-wise.

    Parameters:
    x (numpy.ndarray): Input array or matrix.
    alpha (float): Negative slope coefficient.

    Returns:
    numpy.ndarray: Output with Leaky ReLU applied.
    """
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of the Leaky ReLU function.

    Parameters:
    x (numpy.ndarray): Input array or matrix.
    alpha (float): Negative slope coefficient.

    Returns:
    numpy.ndarray: Derivative of Leaky ReLU.
    """
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    """
    Applies the ELU activation function element-wise.

    Parameters:
    x (numpy.ndarray): Input array or matrix.
    alpha (float): Slope for x <= 0.

    Returns:
    numpy.ndarray: Output with ELU applied.
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """
    Derivative of the ELU activation function.

    Parameters:
    x (numpy.ndarray): Input array or matrix.
    alpha (float): Slope for x <= 0.

    Returns:
    numpy.ndarray: Derivative of ELU.
    """
    return np.where(x > 0, 1, alpha * np.exp(x))


def swish(x):
    """
    Applies the Swish activation function element-wise.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Output with Swish applied.
    """
    return x * sigmoid(x)

def swish_derivative(x):
    """
    Derivative of the Swish activation function.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Derivative of Swish.
    """
    s = sigmoid(x)
    return s + x * s * (1 - s)

def gelu(x):
    """
    Applies the GELU activation function element-wise.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Output with GELU applied.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    """
    Derivative of the GELU activation function (approximation).

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Derivative of GELU.
    """
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    pdf = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2) * np.exp(-0.5 * (np.sqrt(2 / np.pi) * x)**2)
    return cdf + x * pdf

def maxout(x, weights, biases):
    """
    Applies the Maxout activation function.

    Parameters:
    x (numpy.ndarray): Input array or matrix.
    weights (list of numpy.ndarray): List of weight matrices for Maxout units.
    biases (list of numpy.ndarray): List of bias vectors for Maxout units.

    Returns:
    numpy.ndarray: Output with Maxout applied.
    """
    z = [np.dot(w, x) + b for w, b in zip(weights, biases)]
    return np.max(z, axis=0)


def hard_sigmoid(x):
    """
    Applies the Hard Sigmoid activation function element-wise.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Output with Hard Sigmoid applied.
    """
    return np.clip(0.2 * x + 0.5, 0, 1)

def hard_sigmoid_derivative(x):
    """
    Derivative of the Hard Sigmoid activation function.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Derivative of Hard Sigmoid.
    """
    return np.where((x > -2.5) & (x < 2.5), 0.2, 0)

