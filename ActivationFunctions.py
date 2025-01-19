import numpy as np

def relu(x):
    """
    Applies the ReLU activation function robustly to scalars, 1D arrays, or 2D arrays.
    """
    x = np.asarray(x)  # Ensure x is an array
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of the ReLU function, applied robustly to scalars, 1D arrays, or 2D arrays.
    """
    x = np.asarray(x)
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    """
    Applies the sigmoid activation function robustly to scalars, 1D arrays, or 2D arrays.
    """
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function, applied robustly to scalars, 1D arrays, or 2D arrays.
    """
    x = np.asarray(x)
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    """
    Applies the tanh activation function robustly to scalars, 1D arrays, or 2D arrays.
    """
    x = np.asarray(x)
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of the tanh function, applied robustly to scalars, 1D arrays, or 2D arrays.
    """
    x = np.asarray(x)
    return 1 - np.tanh(x)**2

def softmax(x):
    """
    Applies the softmax activation function to 1D or 2D arrays.
    """
    x = np.asarray(x)
    if x.ndim == 1:  # Handle vector input
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Row-wise numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def leaky_relu(x, alpha=0.01):
    """
    Applies the Leaky ReLU activation function robustly to scalars, 1D arrays, or 2D arrays.
    """
    x = np.asarray(x)
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of the Leaky ReLU function, applied robustly to scalars, 1D arrays, or 2D arrays.
    """
    x = np.asarray(x)
    return np.where(x > 0, 1, alpha)
