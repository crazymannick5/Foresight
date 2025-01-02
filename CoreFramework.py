import numpy as np
import pandas as pd 
import Activate as af


def h0MatrixHandle(H):
    H[:,1] = 0

def h0MatrixInit(H):
    H[:,1] = 0

def hMatrixTUpdate(H, Wx, Wh, Wy, bH, by, X, aID):
    h0MatrixHandle(H)
    for i in range(1, H.shape[1]):
        H[:,(i)] = af.apply_activation((np.dot(Wh,H[:,(i-1)]) + np.dot(Wx,X[:,i]) + bH ),aID)
    return(H)

def yOutputUpdate(H, Yhat, by, Wy, aID):
    Yhat = af.apply_activation((np.dot(Wy,H)+by),aID)
    return(Yhat)


        
