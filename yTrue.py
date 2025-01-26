import numpy as np
import pandas as pd
import af

def getYtrue(X, XLast, aID, yTID=1):
    if(yTID == 1) :
        yTrue = YtrueDirect(X,XLast,aID)


def YtrueDirect(X, XLast, aID):
    """
    Computes Ytrue, where the true outputs are the prices of the stock at the next time step
    with the specified activation function applied.

    Args:
        X (np.ndarray): Input matrix (rows = tickers, columns = time steps).
        XLast (np.ndarray): The last column of X (final prices for each ticker, shape: (n_tickers,)).
        aID (str): The identifier for the activation function.

    Returns:
        np.ndarray: Ytrue matrix (rows = tickers, columns = time steps), where each value
                    is the activated price at the next time step.
    """
    # Number of rows (tickers) and columns (time steps) in X
    num_tickers, num_time_steps = X.shape

    # Initialize Ytrue with the same shape as X
    Ytrue = np.zeros_like(X)

    # For each time step, take the value from the next time step and apply the activation function
    for t in range(num_time_steps - 1):  # Exclude the last column
        next_time_step = X[:, t + 1]  # Values at the next time step
        Ytrue[:, t] = af.apply_activation(next_time_step, aID)  # Apply activation directly

    # Handle the last column using XLast (final prices)
    Ytrue[:, -1] = af.apply_activation(XLast, aID)  # Apply activation directly

    return Ytrue

    
    