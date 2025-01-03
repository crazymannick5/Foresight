import numpy as np
import pandas as pd 
import Activate as af
import dyHatT as dy

def h0MatrixHandle(H):
    H[:,1] = 0

def h0MatrixInit(H):
    H[:,1] = 0

def hMatrixTUpdate(H, Z, Wx, Wh, Wy, bH, by, X, aID):
    h0MatrixHandle(H)
    for i in range(1, H.shape[1]):
        H[:,(i)] = af.apply_activation((np.dot(Wh,H[:,(i-1)]) + np.dot(Wx,X[:,i]) + bH ),aID)
        Z[:,(i)] = (np.dot(Wh,H[:,(i-1)]) + np.dot(Wx,X[:,i]) + bH )
    return(H, Z)

def yOutputUpdate(H, yHat, by, Wy, aID):
    Yhat = af.apply_activation((np.dot(Wy,H)+by),aID)
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


