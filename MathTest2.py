import numpy as np
import CoreFramework as cf
from CoreFramework import hMatrixTUpdate, yOutputUpdate



def test_gradients():
    Wy = np.array([[0.5, 0.6], [0.7, 0.8]])
    dyHatT = np.array([[0.1, 0.2], [0.3, 0.4]])
    H = np.array([[0.1, 0.3], [0.5, 0.7]])
    dWy = c_dWy(None, dyHatT, H)

    expected_dWy = np.dot(dyHatT, H.T)
    assert np.allclose(dWy, expected_dWy, atol=1e-2), f"dWy mismatch: {dWy}"


def test_backward_propagation():
    Wx = np.array([[0.1, 0.2], [0.3, 0.4]])
    Wh = np.array([[0.5, 0.6], [0.7, 0.8]])
    bH = np.array([0.1, 0.2])
    Wy = np.array([[0.9, 1.0], [1.1, 1.2]])
    X = np.array([[1, 2], [3, 4]])
    H = np.zeros((2, 2))
    dH = np.array([[0.2, 0.1], [0.3, 0.4]])
    Z = np.array([[0.5, 0.6], [0.7, 0.8]])

    dWx = cf.dWxMatrixUpdate(None, X, dH, Z, 1)
    dWh = cf.dWhMatrixUpdate(None, H, dH, Z, 1)

    expected_dWx = np.array([[0.5, 0.3], [0.7, 0.9]])
    expected_dWh = np.array([[0.4, 0.2], [0.5, 0.3]])

    assert np.allclose(dWx, expected_dWx, atol=1e-2), f"dWx mismatch: {dWx}"
    assert np.allclose(dWh, expected_dWh, atol=1e-2), f"dWh mismatch: {dWh}"

def test_forward_propagation():
    Wx = np.array([[0.1, 0.2], [0.3, 0.4]])
    Wh = np.array([[0.5, 0.6], [0.7, 0.8]])
    bH = np.array([0.1, 0.2])
    Wy = np.array([[0.9, 1.0], [1.1, 1.2]])
    by = np.array([0.3, 0.4])
    X = np.array([[1, 2], [3, 4]])
    H = np.zeros((2, 2))
    Z = np.zeros((2, 2))
    yHat = np.zeros((2, 2))

    H, Z = cf.hMatrixTUpdate(H, Z, Wx, Wh, Wy, bH, by, X, 1)
    yHat = cf.yOutputUpdate(H, yHat, by, Wy, 3)

    expected_H = np.array([[0.537, 0.76], [0.762, 0.905]])
    expected_Z = np.array([[1.1, 1.3], [1.5, 1.7]])
    expected_yHat = np.array([[1.9, 2.24], [2.52, 2.96]])

    assert np.allclose(H, expected_H, atol=1e-2), f"H mismatch: {H}"
    assert np.allclose(Z, expected_Z, atol=1e-2), f"Z mismatch: {Z}"
    assert np.allclose(yHat, expected_yHat, atol=1e-2), f"yHat mismatch: {yHat}"



def run_tests():
    """
    Runs tests for forward propagation, backward propagation, and gradient calculations.
    Prints whether each test passes or fails.
    """
    try:
        # Forward Propagation Test
        print("Testing Forward Propagation...")
        test_forward_propagation()
        print("Forward Propagation Test: PASSED")

        # Backward Propagation Test
        print("Testing Backward Propagation...")
        test_backward_propagation()
        print("Backward Propagation Test: PASSED")

        # Gradient Calculation Test
        print("Testing Gradient Calculations...")
        test_gradients()
        print("Gradient Calculation Test: PASSED")

    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("All tests completed.")


run_tests()