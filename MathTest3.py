import numpy as np
from CoreFramework import hMatrixTUpdate, yOutputUpdate, dWxMatrixUpdate, dWhMatrixUpdate

def run_independent_tests():
    """
    Independently computes forward propagation, backward propagation, and gradients,
    compares these values with the outputs of the provided functions.
    """

    # Initialize parameters
    Wx = np.array([[0.1, 0.2], [0.3, 0.4]])
    Wh = np.array([[0.5, 0.6], [0.7, 0.8]])
    Wy = np.array([[0.9, 1.0], [1.1, 1.2]])
    bH = np.array([0.1, 0.2])
    by = np.array([0.3, 0.4])
    X = np.array([[1, 2], [3, 4]])  # Inputs: 2 features, 2 time steps
    y_true = np.array([[0.5, 0.7], [0.6, 0.8]])  # True outputs

    # Forward propagation (independent calculation)
    print("=== Forward Propagation ===")
    T = X.shape[1]
    hidden_size = Wx.shape[0]
    H_ind = np.zeros((hidden_size, T))
    Z_ind = np.zeros((hidden_size, T))
    yHat_ind = np.zeros_like(y_true)

    for t in range(T):
        z_t = np.dot(Wx, X[:, t]) + (np.dot(Wh, H_ind[:, t - 1]) if t > 0 else 0) + bH
        h_t = np.tanh(z_t)
        y_t = np.dot(Wy, h_t) + by
        yHat_ind[:, t] = y_t

        H_ind[:, t] = h_t
        Z_ind[:, t] = z_t

    print(f"Independent H: \n{H_ind}")
    print(f"Independent Z: \n{Z_ind}")
    print(f"Independent yHat: \n{yHat_ind}")

    # Forward propagation (using your functions)
    H_prog = np.zeros((hidden_size, T))
    Z_prog = np.zeros((hidden_size, T))
    yHat_prog = np.zeros_like(y_true)
    H_prog, Z_prog = hMatrixTUpdate(H_prog, Z_prog, Wx, Wh, Wy, bH, by, X, 1)
    yHat_prog = yOutputUpdate(H_prog, yHat_prog, by, Wy, 3)

    print(f"Program H: \n{H_prog}")
    print(f"Program Z: \n{Z_prog}")
    print(f"Program yHat: \n{yHat_prog}")

    assert np.allclose(H_prog, H_ind, atol=1e-5), "Mismatch in H"
    assert np.allclose(Z_prog, Z_ind, atol=1e-5), "Mismatch in Z"
    assert np.allclose(yHat_prog, yHat_ind, atol=1e-5), "Mismatch in yHat"
    print("Forward Propagation: PASSED")

    # Backward propagation
    print("\n=== Backward Propagation ===")
    dH_ind = np.zeros_like(H_ind)
    dWy_ind = np.zeros_like(Wy)
    dWh_ind = np.zeros_like(Wh)
    dWx_ind = np.zeros_like(Wx)
    dbH_ind = np.zeros_like(bH)
    dby_ind = np.zeros_like(by)

    dyHat = yHat_ind - y_true  # Loss gradient w.r.t. yHat
    for t in reversed(range(T)):
        dWy_ind += np.outer(dyHat[:, t], H_ind[:, t])
        dby_ind += dyHat[:, t]
        dH_ind[:, t] = np.dot(Wy.T, dyHat[:, t])

        if t < T - 1:
            dH_ind[:, t] += np.dot(Wh.T, dH_ind[:, t + 1]) * (1 - H_ind[:, t] ** 2)

        dWh_ind += np.outer(dH_ind[:, t] * (1 - H_ind[:, t] ** 2), H_ind[:, t - 1]) if t > 0 else 0
        dWx_ind += np.outer(dH_ind[:, t] * (1 - H_ind[:, t] ** 2), X[:, t])
        dbH_ind += dH_ind[:, t] * (1 - H_ind[:, t] ** 2)

    print(f"Independent dWy: \n{dWy_ind}")
    print(f"Independent dWh: \n{dWh_ind}")
    print(f"Independent dWx: \n{dWx_ind}")
    print(f"Independent dbH: \n{dbH_ind}")
    print(f"Independent dby: \n{dby_ind}")

    # Gradients using your functions
    dWx_prog = dWxMatrixUpdate(None, X, dH_ind, Z_ind, 1)
    dWh_prog = dWhMatrixUpdate(None, H_ind, dH_ind, Z_ind, 1)

    print(f"Program dWy: \n{dWy_ind}")  # Your code doesn't directly compute dWy
    print(f"Program dWh: \n{dWh_prog}")
    print(f"Program dWx: \n{dWx_prog}")

    assert np.allclose(dWx_prog, dWx_ind, atol=1e-5), "Mismatch in dWx"
    assert np.allclose(dWh_prog, dWh_ind, atol=1e-5), "Mismatch in dWh"
    print("Backward Propagation: PASSED")

run_independent_tests()
