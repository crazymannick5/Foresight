import numpy as np

def initialize_matrix(iID, shape):
    """
    Initializes a matrix based on the specified initialization ID.

    Args:
        iID (str): Initialization ID determining how the matrix is initialized.
        shape (tuple): Shape of the matrix to initialize.

    Returns:
        numpy.ndarray: Initialized matrix.
    """
    if iID == "he":
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    elif iID == "xavier":
        return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])
    elif iID == "random":
        return np.random.randn(*shape) * 0.01
    elif iID == "zeros":
        return np.zeros(shape)
    else:
        raise ValueError(f"Unknown initialization ID: {iID}")

def PostInputInit(hSize, X, yTrue, iID):
    """
    Initializes and returns all required matrices and vectors for an RNN,
    including a Z matrix to store Z_t vectors as columns.

    Args:
        hSize (int): Size of the hidden layer (uniform for all time steps).
        X (numpy.ndarray): Input data of shape (n_samples, input_size).
        yTrue (numpy.ndarray): Ground truth output data of shape (n_samples, output_size).
        iID (str): Initialization ID determining how matrices are initialized.

    Returns:
        dict: A dictionary containing all initialized matrices and vectors.
    """
    input_size = X.shape[1]  # Number of features in input data
    output_size = yTrue.shape[1]  # Number of features in output data
    n_samples = X.shape[0]  # Number of samples
    T = n_samples  # Assume each sample corresponds to a time step

    W_xh = initialize_matrix(iID, (hSize, input_size))  # Input-to-hidden weights
    W_hh = initialize_matrix(iID, (hSize, hSize))  # Hidden-to-hidden weights
    b_h = initialize_matrix(iID, (hSize, 1))  # Hidden bias
    W_hy = initialize_matrix(iID, (output_size, hSize))  # Hidden-to-output weights
    b_y = initialize_matrix(iID, (output_size, 1))  # Output bias
    Z = initialize_matrix("zeros", (hSize, T))  # Z matrix to store Z_t vectors

    return {
        "W_xh": W_xh,
        "W_hh": W_hh,
        "b_h": b_h,
        "W_hy": W_hy,
        "b_y": b_y,
        "Z": Z
    }




"""# Example input and output data
X = np.random.randn(100, 10)  # 100 samples, 10 features
yTrue = np.random.randn(100, 5)  # 100 samples, 5 output features

# Hidden layer size and initialization method
hSize = 20
iID = "he"

# Call the function
initialized_matrices = PostInputInit(hSize, X, yTrue, iID)

# Access the matrices
W_xh = initialized_matrices["W_xh"]  # Input-to-hidden weights
W_hh = initialized_matrices["W_hh"]  # Hidden-to-hidden weights
b_h = initialized_matrices["b_h"]    # Hidden bias
W_hy = initialized_matrices["W_hy"]  # Hidden-to-output weights
b_y = initialized_matrices["b_y"]    # Output bias

# Check the shapes
print("W_xh shape:", W_xh.shape)
print("W_hh shape:", W_hh.shape)
print("b_h shape:", b_h.shape)
print("W_hy shape:", W_hy.shape)
print("b_y shape:", b_y.shape)"""