import numpy as np
import CoreFramework as cf
from CoreFramework import hMatrixTUpdate, yOutputUpdate

def test_forward_propagation():
    # Example inputs
    Wx = np.array([[0.1, 0.2], [0.3, 0.4]])
    Wh = np.array([[0.5, 0.6], [0.7, 0.8]])
    bH = np.array([0.1, 0.2])
    Wy = np.array([[0.9, 1.0], [1.1, 1.2]])
    by = np.array([0.3, 0.4])
    X = np.array([[1, 2], [3, 4]])  # Input matrix
    H = np.zeros((2, 2))  # Hidden states
    Z = np.zeros((2, 2))  # Pre-activations
    yHat = np.zeros((2, 2))  # Outputs

    # Perform forward propagation
    H, Z = cf.hMatrixTUpdate(H, Z, Wx, Wh, Wy, bH, by, X, 1)
    yHat = cf.yOutputUpdate(H, yHat, by, Wy, 3)

    # Expected values (calculated manually or using a simpler method)
    expected_H = np.array([[0.53704957, 0.75951002], [0.76159416, 0.90514825]])
    expected_Z = np.array([[1.1, 1.3], [1.5, 1.7]])
    expected_yHat = np.array([[1.89954038, 2.24279322], [2.51839285, 2.96164986]])

    # Assertions
    assert np.allclose(H, expected_H, atol=1e-5), f"H values incorrect: {H}"
    assert np.allclose(Z, expected_Z, atol=1e-5), f"Z values incorrect: {Z}"
    assert np.allclose(yHat, expected_yHat, atol=1e-5), f"yHat values incorrect: {yHat}"


def test_gradients():
    # Example inputs
    Wx = np.random.randn(2, 2)
    X = np.random.randn(2, 2)
    dH = np.random.randn(2, 2)
    Z = np.random.randn(2, 2)

    # Compute analytical gradient
    analytic_gradient = cf.dWxMatrixUpdate(None, X, dH, Z, 1)

    # Compute numerical gradient
    epsilon = 1e-6
    numerical_gradient = np.zeros_like(Wx)
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            Wx[i, j] += epsilon
            plus_loss = np.sum(np.dot(Wx, X))  # Forward computation with perturbed Wx
            Wx[i, j] -= 2 * epsilon
            minus_loss = np.sum(np.dot(Wx, X))  # Forward computation with perturbed Wx
            numerical_gradient[i, j] = (plus_loss - minus_loss) / (2 * epsilon)
            Wx[i, j] += epsilon  # Reset Wx

    # Assertions
    assert np.allclose(analytic_gradient, numerical_gradient, atol=1e-5), (
        f"Gradient mismatch:\nAnalytical:\n{analytic_gradient}\nNumerical:\n{numerical_gradient}"
    )

def test_backward_propagation():
    # Example inputs
    Wx = np.array([[0.1, 0.2], [0.3, 0.4]])
    Wh = np.array([[0.5, 0.6], [0.7, 0.8]])
    bH = np.array([0.1, 0.2])
    Wy = np.array([[0.9, 1.0], [1.1, 1.2]])
    by = np.array([0.3, 0.4])
    X = np.array([[1, 2], [3, 4]])  # Input matrix
    H = np.zeros((2, 2))  # Hidden states
    dH = np.array([[0.2, 0.1], [0.3, 0.4]])
    Z = np.array([[0.5, 0.6], [0.7, 0.8]])

    # Compute gradients
    dWx = cf.dWxMatrixUpdate(None, X, dH, Z, 1)
    dWh = cf.dWhMatrixUpdate(None, H, dH, Z, 1)

    # Expected values (calculated manually or from simpler tests)
    expected_dWx = np.array([[0.5, 0.3], [0.7, 0.9]])
    expected_dWh = np.array([[0.4, 0.2], [0.5, 0.3]])

    # Assertions
    assert np.allclose(dWx, expected_dWx, atol=1e-5), f"dWx incorrect: {dWx}"
    assert np.allclose(dWh, expected_dWh, atol=1e-5), f"dWh incorrect: {dWh}"


def run_tests():
    """
    Runs all tests for forward propagation, backward propagation, and gradients.
    Prints whether each test passes or fails.
    """
    try:
        # Test forward propagation
        test_forward_propagation()
    except Exception as e:
        print(f"Forward Propagation Test: FAILED with error: {e}")
    
    try:
        # Test gradients
        test_gradients()
    except Exception as e:
        print(f"Gradient Test: FAILED with error: {e}")
    
    try:
        # Test backward propagation
        test_backward_propagation()
    except Exception as e:
        print(f"Backward Propagation Test: FAILED with error: {e}")

    print("All tests completed.")


#run_tests()

def print_array(array):
    """
    Prints a numpy array in a readable format.

    Parameters:
    array (np.ndarray): The numpy array to print.
    """
    print("Numpy Array:")
    print(array)

