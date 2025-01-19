import numpy as np
import pandas as pd 
import Activate as af
import dyHatT as dy
from printArray import print_array
def column_to_matrix(column_vector, m):
    """
    Converts a column vector into a matrix with m identical columns.

    Args:
        column_vector (numpy.ndarray): A column vector of shape (n, 1).
        m (int): The number of columns in the resulting matrix.

    Returns:
        numpy.ndarray: A matrix of shape (n, m) with m identical columns.
    """
    if column_vector.ndim != 2 or column_vector.shape[1] != 1:
        raise ValueError("Input must be a column vector of shape (n, 1)")
    
    # Repeat the column vector along the column axis
    return np.repeat(column_vector, m, axis=1)
def h0MatrixHandle(H):
    H[:,1] = 0

def h0MatrixInit(H):
    H[:,1] = 0

def hMatrixTUpdate(H, Z, Wx, Wh, Wy, bH, by, X, aID):
    for i in range(0, H.shape[1]):
        if(i == 0):
            H[:,0] = af.apply_activation(( np.dot(Wx,X[:,0]) + bH ),aID)
            Z[:,0] = np.dot(Wx,X[:,0]) + bH
        else:
            H[:,(i)] = af.apply_activation((np.dot(Wh,H[:,(i-1)]) + np.dot(Wx,X[:,i]) + bH ),aID)
            print("row " + str(i) + ":")
            print_array(H[i])
            Z[:,(i)] = (np.dot(Wh,H[:,(i-1)]) + np.dot(Wx,X[:,i]) + bH )

        print("row " + str(i) + ":")
        print_array(H[i])
    return(H, Z)

def yOutputUpdate(H, yHat, by, Wy, aID):
    print_array(by)
    #yHat = af.apply_activation((np.dot(Wy,H)+by),aID)
    #print("tiled:")
    #print_array(np.tile(by,(1,2)))
    #yHat = af.apply_activation((np.dot(Wy,H)+column_to_matrix(by,2)),aID)
    for t in range(2) :
        yHat[:, t] = af.apply_activation((np.dot(Wy, H[:, t]) + by ), aID)
    
    return(yHat)


def c_dyHatT(dyHatT, yHat, yTrue, LID, delta=1.0):
    """
    Computes the gradient of the loss with respect to yHat (dyHatT) for various loss functions.

    Parameters:
    yHat (numpy.ndarray): Predicted outputs (output_size x time_steps).
    yTrue (numpy.ndarray): True labels (output_size x time_steps).
    LID (int): Loss function ID.
        Supported values:
            0: Mean Squared Error (MSE)
            1: Cross-Entropy Loss
            2: Huber Loss
    delta (float): Delta parameter for Huber Loss. Default is 1.0.

    Returns:
    numpy.ndarray: Gradient of the loss with respect to yHat (dyHatT).
    """
    if LID == 0:  # Mean Squared Error
        dyHatT = dy.mse_loss(yHat, yTrue)  # Gradient of MSE

    elif LID == 1:  # Cross-Entropy Loss
        dyHatT = dy.cross_entropy_loss(yHat, yTrue)  # Cross-entropy gradient (assumes softmax output)

    elif LID == 2:  # Huber Loss
        dyHatT = dy.huber_loss(yHat, yTrue)
    elif LID == 3:
        dyHatT = dy.mse_loss_simple(yHat, yTrue)

    else:
        raise ValueError(f"Unsupported Loss Function ID: {LID}")

    return dyHatT

def c_dWy(dWy, dyHatT, H):
    dWy = np.dot(dyHatT,H.T)
    return(dWy)

def c_dby(dby, dyHatT):
    dby = np.sum(dyHatT, axis=1, keepdims=True)
    return(dby)

def c_tdH(dyHatT, Wy, H, by, aID):
    # Step 1: Compute Z_y = Wy * H + by
    Z_y = np.dot(Wy, H) + by  # Shape: (output_size, time_steps)
    
    # Step 2: Compute activation derivative f'(Z_y)
    f_prime = af.apply_activation_derivative(Z_y, aID)  # Shape: (output_size, time_steps)
    
    # Step 3: Element-wise multiply dyHatT with f'(Z_y)
    weighted_dyHatT = dyHatT * f_prime  # Shape: (output_size, time_steps)
    
    # Step 4: Compute tdH = Wy^T * (dyHatT * f'(Z_y))
    tdH = np.dot(Wy.T, weighted_dyHatT)  # Shape: (hidden_size, time_steps)
    
    return tdH


def dhtMatrixUpdate(dH, tdH, aID, Wh, dyHatT, Wy, Z):
    dH = np.zeros_like(tdH)
    dH[:,tdH.shape[1]-1] = tdH[:,tdH.shape[1]-1]
    #dH[:,0] = np.dot(dyHatT[:,(dyHatT.shape[1]-1)],Wy.T)
    for j in range(1, dH.shape[1]):
        i = dH.shape[1]-1-j
        dH[:, i] = tdH[:, i] + np.dot(Wh.T, dH[:, i + 1]) * af.apply_activation_derivative(Z[:, i + 1], aID)
    return(dH)

def dWxMatrixUpdate(dWx, X, dH, Z, aID):
    """dWx = np.dot((af.apply_activation_derivative(Z,aID)*dH),X.T)
    return(dWx)"""
    
    """f_prime_dH = (af.apply_activation_derivative(Z, aID) * dH).T  # Transpose to match independent implementation
    dWx = np.dot(f_prime_dH.T, X.T) 
    return dWx"""

    time_steps = X.shape[1]
    hidden_size = dH.shape[0]
    input_size = X.shape[0]

    # Initialize dWx
    dWx = np.zeros((hidden_size, input_size))

    # Compute dWx step by step to ensure correctness
    for t in range(time_steps):
        # Compute f'(z_t) * dH[:, t]
        f_prime_dH = af.apply_activation_derivative(Z[:, t], aID) * dH[:, t]
        print(f"PROG CHECK CHECK PROGTime Step {t}, postACT:\n", af.apply_activation_derivative(Z[:, t], aID))
        print(f"PROG CHECK CHECK PROGTime Step {t}, fprime:\n", f_prime_dH)
        # Compute the outer product with x_t
        dWx += np.dot(f_prime_dH[:, None], X[:, t][None, :])
        contribution = np.outer(f_prime_dH, X[:, t])
        print(f"PROG CHECK CHECK PROGTime Step {t}, Contribution:\n", contribution)
    return dWx
    
    """
    Compute dWx using the most mathematically correct implementation.

    Args:
        dWx: Placeholder for gradient w.r.t. Wx (not used in computation, for consistency in header)
        X: Input matrix (input_size x time_steps)
        dH: Gradient of the loss w.r.t. hidden states (hidden_size x time_steps)
        Z: Pre-activation values (hidden_size x time_steps)
        aID: Activation function identifier

    Returns:
        dWx: Gradient w.r.t. input-to-hidden weights (hidden_size x input_size)
    """
    """assert X.shape[1] == Z.shape[1] == dH.shape[1], "Time steps must match across X, Z, and dH"
    assert Z.shape[0] == dH.shape[0], "Hidden size must match across Z and dH"

    # Step 1: Compute the element-wise product f'(Z) * dH
    weighted_dH = af.apply_activation_derivative(Z, aID) * dH

    # Step 2: Compute dWx via matrix multiplication
    dWx = np.dot(weighted_dH, X.T)
    contributions = np.dot(weighted_dH, X.T)
    print("Matrix Multiplication Contributions:\n", contributions)
    return dWx"""


def dWhMatrixUpdate(dWh, H, dH, Z, aID):
    sum = 0
    for t in range(1, H.shape[1]):
        sum = sum + (np.outer(((af.apply_activation_derivative(Z[:,t],aID))*dH[:,t]), H[:,t-1]))
    return(sum)

def dbh(dH, Z, aID):
    sum = 0
    for t in range(1, Z.shape[1]):
        sum = sum + (af.apply_activation_derivative(Z[:,t], aID) * dH[:,t])
    return(sum)