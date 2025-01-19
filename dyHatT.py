import numpy as np
def mse_loss_simple(yHat, yTrue):
    """
    Mean Squared Error (MSE) Loss and Gradient.
    
    Gradient: dL/dyHat = 2 * (yHat - yTrue)
    """
    return 2 * (yHat - yTrue)

def mse_loss(yHat, yTrue):
    """
    Mean Squared Error (MSE) Loss and Gradient.
    
    Loss: L = (1/2) * sum((yHat - yTrue)^2)
    Gradient: dL/dyHat = yHat - yTrue
    
    Parameters:
    yHat (numpy.ndarray): Predicted outputs.
    yTrue (numpy.ndarray): True labels.

    Returns:
    numpy.ndarray: Gradient of MSE loss with respect to yHat.
    """
    return yHat - yTrue

def cross_entropy_loss(yHat, yTrue):
    """
    Cross-Entropy Loss and Gradient.
    
    Loss: L = -sum(yTrue * log(yHat))
    Gradient: dL/dyHat = yHat - yTrue
    
    Parameters:
    yHat (numpy.ndarray): Predicted outputs (softmax probabilities).
    yTrue (numpy.ndarray): True labels (one-hot encoded).

    Returns:
    numpy.ndarray: Gradient of Cross-Entropy loss with respect to yHat.
    """
    return yHat - yTrue

def huber_loss(yHat, yTrue, delta=1.0):
    """
    Huber Loss and Gradient.
    
    Loss:
    L = { (1/2)*(yHat - yTrue)^2 if |yHat - yTrue| <= delta
          delta * |yHat - yTrue| - (1/2) * delta^2 otherwise }
    Gradient:
    dL/dyHat = { yHat - yTrue if |yHat - yTrue| <= delta
                 delta * sign(yHat - yTrue) otherwise }

    Parameters:
    yHat (numpy.ndarray): Predicted outputs.
    yTrue (numpy.ndarray): True labels.
    delta (float): Threshold for switching between MSE and Absolute Error.

    Returns:
    numpy.ndarray: Gradient of Huber loss with respect to yHat.
    """
    error = yHat - yTrue
    return np.where(np.abs(error) <= delta, error, delta * np.sign(error))

