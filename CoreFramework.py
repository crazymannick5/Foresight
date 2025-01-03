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
    dH[:,0] = np.dot(dyHatT[:,(dyHatT.shape[1]-1)],Wy.T)
    for j in range(1, dH.shape[1]):
        i = dH.shape[1]-1-j
        dH[:,(i)] = tdH[:,i]+np.dot(dH[:,(i+1)],(af.apply_activation_derivative((Z[:,i+1]),aID)))
    return(dH)

def dWxMatrixUpdate(dWx, X, dH, Z, aID):
    dWx = np.dot((af.apply_activation_derivative(Z,aID)*dH),X.T)
    return(dWx)

def dWhMatrixUpdate(dWh, H, dH, Z, aID):
    sum = 0
    for t in range(1, H.shape[1]):
        sum = sum + (np.outer(((af.apply_activation_derivative(Z[:,t],aID))*dH[:,t]), H[:,t-1]))
    return(sum)

def dbh(dH, Z, aID):
    sum = 0
    for t in range(1, Z.shape[1]):
        sum = sum + (af.apply_activation_derivative(Z[:,t]) * dH[:,t])
    return(sum)