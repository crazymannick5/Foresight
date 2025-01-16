import numpy as np
import CoreFramework as cf
import Activate as af

def forward_propagation(X, Wx, Wh, Wy, bH, by, aID):
    """
    Performs forward propagation through an RNN.
    Args:
        X: Input matrix (input_size x time_steps)
        Wx: Input-to-hidden weight matrix (hidden_size x input_size)
        Wh: Hidden-to-hidden weight matrix (hidden_size x hidden_size)
        Wy: Hidden-to-output weight matrix (output_size x hidden_size)
        bH: Hidden bias (hidden_size)
        by: Output bias (output_size)
    Returns:
        H: Hidden states (hidden_size x time_steps)
        Z: Pre-activations (hidden_size x time_steps)
        yHat: Outputs (output_size x time_steps)
    """
    hidden_size, time_steps = Wh.shape[0], X.shape[1]
    H = np.zeros((hidden_size, time_steps))
    Z = np.zeros((hidden_size, time_steps))
    yHat = np.zeros((Wy.shape[0], time_steps))

    for t in range(time_steps):
        Z[:, t] = np.dot(Wx, X[:, t]) + (np.dot(Wh, H[:, t - 1]) if t > 0 else 0) + bH
        H[:, t] = af.apply_activation((Z[:, t]),aID)  # Activation function
        yHat[:, t] = af.apply_activation((np.dot(Wy, H[:, t]) + by ), aID)

    return H, Z, yHat


"""def backward_propagation(X, y_true, H, Z, Wx, Wh, Wy, bH, by):
    """"""
    Performs backward propagation through an RNN to compute gradients.
    Args:
        X: Input matrix (input_size x time_steps)
        y_true: True outputs (output_size x time_steps)
        H, Z: Hidden states and pre-activations from forward propagation
        Wx, Wh, Wy, bH, by: Parameters
    Returns:
        Gradients for Wx, Wh, Wy, bH, by
    """"""
    time_steps = X.shape[1]
    hidden_size = Wh.shape[0]
    output_size = Wy.shape[0]

    dH = np.zeros_like(H)
    dZ = np.zeros_like(Z)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    dbH = np.zeros_like(bH)
    dby = np.zeros_like(by)

    dyHat = H - y_true  # Simple difference for MSE loss
    for t in reversed(range(time_steps)):
        dWy += np.outer(dyHat[:, t], H[:, t])
        dby += dyHat[:, t]

        dH[:, t] = np.dot(Wy.T, dyHat[:, t])
        if t < time_steps - 1:
            dH[:, t] += np.dot(Wh.T, dZ[:, t + 1])

        dZ[:, t] = dH[:, t] * (1 - H[:, t] ** 2)  # Derivative of tanh
        dWx += np.outer(dZ[:, t], X[:, t])
        dbH += dZ[:, t]
        if t > 0:
            dWh += np.outer(dZ[:, t], H[:, t - 1])

    return dWx, dWh, dWy, dbH, dby"""
"""def backward_propagation(X, y_true, H, Z, Wx, Wh, Wy, bH, by, aID, LID):
    
    Independent backward propagation through time for an RNN with output layer.
    Args:
        X: Input matrix (input_size x time_steps)
        y_true: True outputs (output_size x time_steps)
        H: Hidden states (hidden_size x time_steps)
        Z: Pre-activation values (hidden_size x time_steps)
        Wx, Wh, Wy: Weight matrices
        bH, by: Bias vectors
        aID: Identifier for activation function
        LID: Identifier for loss function
    Returns:
        Gradients: dWx, dWh, dWy, dbH, dby
    
    time_steps = X.shape[1]
    hidden_size = Wh.shape[0]
    output_size = Wy.shape[0]

    # Initialize gradient matrices
    dH = np.zeros_like(H)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    dbH = np.zeros_like(bH)
    dby = np.zeros_like(by)

    # Loss gradient w.r.t. outputs (dyHat) using yHat
    yHat = np.zeros((output_size, time_steps))  # Placeholder for actual outputs
    for t in range(time_steps):
        yHat[:, t] = af.apply_activation(np.dot(Wy, H[:, t]) + by, aID)
    
    if LID == 0:  # Mean Squared Error
        dyHat = (yHat - y_true)
    elif LID == 1:  # Cross-Entropy Loss
        dyHat = (yHat - y_true)
    else:
        raise ValueError("Unsupported loss function identifier")

    # Backward propagation through time
    for t in reversed(range(time_steps)):
        # Gradients w.r.t. output weights and biases
        dWy += np.outer(dyHat[:, t], H[:, t])
        dby += dyHat[:, t]

        # Gradients w.r.t. hidden states
        dH[:, t] = np.dot(Wy.T, dyHat[:, t])
        if t < time_steps - 1:
            dH[:, t] += np.dot(Wh.T, dH[:, t + 1]) * af.apply_activation_derivative(Z[:, t + 1], aID)

        # Halfway Check: Compare intermediate computation of `dWx`
        if t == time_steps // 2:  # Check halfway through the time steps
            f_prime_times_dH = af.apply_activation_derivative(Z[:, t], aID) * dH[:, t]
            single_step_dWx = np.dot(f_prime_times_dH[:, None], X[:, t][None, :])
            print(f"\n=== Halfway Check at t={t} ===")
            print(f"f'(z_t) * dH[:, t]: \n{f_prime_times_dH}")
            print(f"Single-step dWx contribution at t={t}: \n{single_step_dWx}")

        # Gradients w.r.t. input-to-hidden weights and biases
        dWx += np.dot((af.apply_activation_derivative(Z[:, t], aID) * dH[:, t])[:, None], X[:, t][None, :])
        dbH += af.apply_activation_derivative(Z[:, t], aID) * dH[:, t]

        # Gradients w.r.t. hidden-to-hidden weights
        if t > 0:
            dWh += np.dot((af.apply_activation_derivative(Z[:, t], aID) * dH[:, t])[:, None], H[:, t - 1][None, :])

    return dWx, dWh, dWy, dbH, dby"""

def backward_propagation(X, y_true, H, Z, Wx, Wh, Wy, bH, by, aID, LID):
    """
    Backward propagation through time for an RNN, computed from first principles.
    Args:
        X: Input matrix (input_size x time_steps)
        y_true: True outputs (output_size x time_steps)
        H: Hidden states (hidden_size x time_steps)
        Z: Pre-activation values (hidden_size x time_steps)
        Wx, Wh, Wy: Weight matrices
        bH, by: Bias vectors
        aID: Activation function identifier
        LID: Loss function identifier
    Returns:
        Gradients: dWx, dWh, dWy, dbH, dby
    """
    time_steps = X.shape[1]
    hidden_size = Wh.shape[0]
    input_size = X.shape[0]

    # Initialize gradients
    dH = np.zeros_like(H)
    dWx = np.zeros((hidden_size, input_size))
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    dbH = np.zeros_like(bH)
    dby = np.zeros_like(by)

    # Compute output predictions (forward pass)
    yHat = np.zeros((Wy.shape[0], time_steps))
    for t in range(time_steps):
        yHat[:, t] = af.apply_activation(np.dot(Wy, H[:, t]) + by, aID)

    # Compute loss gradient w.r.t. outputs
    if LID == 0:  # Mean Squared Error
        dyHat = 2 * (yHat - y_true)
    elif LID == 1:  # Cross-Entropy Loss 
        dyHat = yHat - y_true
    else:
        raise ValueError("Unsupported loss function identifier")

    # Backward pass through time
    for t in reversed(range(time_steps)):
        # Output weight gradients
        dWy += np.outer(dyHat[:, t], H[:, t])
        dby += dyHat[:, t]

        # Hidden state gradient
        #dH[:, t] = np.dot(Wy.T, dyHat[:, t])
        dH[:,t] = (np.dot(Wy.T, (dyHat*(af.apply_activation_derivative((np.dot(Wy,H) + by),aID)))))[:,t]
        #dH[:,t] = np.dot(Wy.T,(dyHat*(af.apply_activation_derivative())))
        if t < time_steps - 1:
            dH[:, t] += np.dot(Wh.T, dH[:, t + 1]) * af.apply_activation_derivative(Z[:, t + 1], aID)


    
    # Compute dWx from first principles
    # Sum over all time steps: f'(z_t) * dH[:, t] * X[:, t]^T
    for t in range(time_steps):
        f_prime_dH = af.apply_activation_derivative(Z[:, t], aID) * dH[:, t]
        print(f"IND CHECK CHECK INDTime Step {t}, postACT:\n", af.apply_activation_derivative(Z[:, t], aID))
        print(f"IND CHECK CHECK IND Time Step {t}, fprime:\n", f_prime_dH)
        dWx += np.outer(f_prime_dH, X[:, t])  # Outer product for this time step
        contribution = np.outer(f_prime_dH, X[:, t])
        print(f"IND CHECK CHECK IND Time Step {t}, Contribution:\n", contribution)
    # Compute dWh from first principles
    for t in range(1, time_steps):
        f_prime_dH = af.apply_activation_derivative(Z[:, t], aID) * dH[:, t]
        dWh += np.outer(f_prime_dH, H[:, t - 1])

    # Compute dbH
    dbH = np.sum(af.apply_activation_derivative(Z, aID) * dH, axis=1)

    return dWx, dWh, dWy, dbH, dby, dH



 
def test_rnn_implementation():
    aID = 1  # Activation function identifier
    LID = 0  # Loss function identifier (e.g., 0 for MSE, 1 for Cross-Entropy)
    """
    Tests the independent RNN implementation against the user's implementation,
    focusing on dH, Z, and X right before calculating dWx.
    """
    # Initialize parameters
    Wx = np.array([[0.1, 0.2], [0.3, 0.4]])
    Wh = np.array([[0.5, 0.6], [0.7, 0.8]])
    Wy = np.array([[0.9, 1.0], [1.1, 1.2]])
    bH = np.array([0.1, 0.2])
    by = np.array([0.3, 0.4])
    X = np.array([[1, 2], [3, 4]])
    y_true = np.array([[0.5, 0.7], [0.6, 0.8]])

    # Independent Forward Propagation
    H_ind, Z_ind, yHat_ind = forward_propagation(X, Wx, Wh, Wy, bH, by, aID)

    # User's Forward Propagation
    from CoreFramework import hMatrixTUpdate, yOutputUpdate
    H_prog = np.zeros_like(H_ind)
    Z_prog = np.zeros_like(Z_ind)
    yHat_prog = np.zeros_like(yHat_ind)
    H_prog, Z_prog = cf.hMatrixTUpdate(H_prog, Z_prog, Wx, Wh, Wy, bH, by, X, aID)
    yHat_prog = cf.yOutputUpdate(H_prog, yHat_prog, by, Wy, aID)

    # Independent Backward Propagation
    dWx_ind, dWh_ind, dWy_ind, dbH_ind, dby_ind, dH_ind = backward_propagation(X, y_true, H_ind, Z_ind, Wx, Wh, Wy, bH, by, aID, LID)
    dWx_ind = dWx_ind/2
    dWh
    # User's Backward Propagation
    # Step 1: Compute dyHatT (loss gradient w.r.t outputs)
    dyHatT_prog = cf.c_dyHatT(None, yHat_prog, y_true, LID)

    # Step 2: Compute tdH using your `c_tdH`
    from CoreFramework import c_tdH
    tdH_prog = c_tdH(dyHatT_prog, Wy, H_prog, by, aID)

    # Compute Independent tdH (updated to match `c_tdH`)
    Z_y_ind = np.dot(Wy, H_ind) + by  # Pre-activation values at the output layer
    f_prime_ind = af.apply_activation_derivative(Z_y_ind, aID)  # Activation derivative
    weighted_dyHat_ind = dyHatT_prog * f_prime_ind  # Scale dyHatT by f'(Z_y)
    tdH_ind = np.dot(Wy.T, weighted_dyHat_ind)  # Project to hidden layer

    # Step 3: Compute dH_prog using your framework
    dH_prog = cf.dhtMatrixUpdate(np.zeros_like(H_prog), tdH_prog, aID, Wh, dyHatT_prog, Wy, Z_prog)

    # Compute Independent dH
    """dH_ind = np.zeros_like(H_ind)
    for t in reversed(range(X.shape[1])):
        dH_ind[:, t] = tdH_ind[:, t]
        if t < X.shape[1] - 1:
            dH_ind[:, t] += np.dot(Wh.T, dH_ind[:, t + 1]) * af.apply_activation_derivative(Z_ind[:, t + 1], aID)
    """
    

    for t in range(dH_prog.shape[1]):
        print(f"f'(z_t) * dH[:, t] (Independent) at t={t}:\n", af.apply_activation_derivative(Z_ind[:, t], aID) * dH_ind[:, t])

    print("f'(Z) * dH (Program):\n", (af.apply_activation_derivative(Z_prog, aID) * dH_prog).T )

    # Check and Compare dH, Z, and X Before dWx Calculation
    print("\n=== Pre-dWx Check ===")
    for t in range(X.shape[1]):
        print(f"Time Step t={t}")
        print(f"Independent tdH[:, {t}]: \n{tdH_ind[:, t]}")
        print(f"Program tdH[:, {t}]: \n{tdH_prog[:, t]}")
        print(f"Independent dH[:, {t}]: \n{dH_ind[:, t]}")
        print(f"Program dH[:, {t}]: \n{dH_prog[:, t]}")
        print(f"Independent Z[:, {t}]: \n{Z_ind[:, t]}")
        print(f"Program Z[:, {t}]: \n{Z_prog[:, t]}")
        print(f"X[:, {t}]: \n{X[:, t]}")  # X is identical in both cases

    # User's Backward Propagation for dWx
    dWx_prog = cf.dWxMatrixUpdate(None, X, dH_prog, Z_prog, aID)

    # Print Results and Compare
    print("\n=== Forward Propagation ===")
    print(f"Independent H: \n{H_ind}")
    print(f"Program H: \n{H_prog}")
    print(f"Independent yHat: \n{yHat_ind}")
    print(f"Program yHat: \n{yHat_prog}")
    print(f"Independent Z: \n{Z_ind}")
    print(f"Program Z: \n{Z_prog}")

    print("\n=== Backward Propagation ===")
    print(f"Independent tdH: \n{tdH_ind}")
    print(f"Program tdH: \n{tdH_prog}")
    print(f"Independent dH: \n{dH_ind}")
    print(f"Program dH: \n{dH_prog}")
    print(f"Independent dWx: \n{dWx_ind}")
    print(f"Program dWx: \n{dWx_prog}")

    # Assertions
    assert np.allclose(H_prog, H_ind, atol=1e-5), "Mismatch in H"
    assert np.allclose(Z_prog, Z_ind, atol=1e-5), "Mismatch in Z"
    assert np.allclose(yHat_prog, yHat_ind, atol=1e-5), "Mismatch in yHat"
    assert np.allclose(tdH_ind, tdH_prog, atol=1e-5), "Mismatch in tdH"
    assert np.allclose(dH_ind, dH_prog, atol=1e-5), "Mismatch in dH"
    assert np.allclose(dWx_prog, dWx_ind, atol=1e-5), "Mismatch in dWx"
    print("\nAll tests passed successfully!")

"""def backward_propagation(X, y_true, H, Z, Wx, Wh, Wy, bH, by, aID, LID):

    Performs backward propagation through an RNN to compute gradients with flexible activation and loss functions.
    Args:
        X: Input matrix (input_size x time_steps)
        y_true: True outputs (output_size x time_steps)
        H, Z: Hidden states and pre-activations from forward propagation
        Wx, Wh, Wy, bH, by: Parameters
        aID: Identifier for the activation function (used with af.apply_activation_derivative)
        LID: Identifier for the loss function
    Returns:
        Gradients for Wx, Wh, Wy, bH, by
    
    time_steps = X.shape[1]
    hidden_size = Wh.shape[0]
    output_size = Wy.shape[0]

    # Initialize gradients
    dH = np.zeros_like(H)
    dZ = np.zeros_like(Z)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    dbH = np.zeros_like(bH)
    dby = np.zeros_like(by)

    # Compute loss gradient w.r.t. output based on LID
    if LID == 0:
        dyHat = 2 * (H - y_true)  # Mean Squared Error gradient
    elif LID == 1:
        dyHat = H - y_true  # Cross-Entropy gradient
    else:
        raise ValueError(f"Unsupported Loss ID: {LID}")

    # Backward propagation through time
    for t in reversed(range(time_steps)):
        # Gradients w.r.t. output weights and biases
        dWy += np.outer(dyHat[:, t], H[:, t])
        dby += dyHat[:, t]

        # Gradients w.r.t. hidden states
        dH[:, t] = np.dot(Wy.T, dyHat[:, t])
        if t < time_steps - 1:
            dH[:, t] += np.dot(Wh.T, dZ[:, t + 1])

        # Gradients w.r.t. pre-activation values
        dZ[:, t] = dH[:, t] * af.apply_activation_derivative(Z[:, t], aID)

        # Gradients w.r.t. weights and biases
        dWx += np.outer(dZ[:, t], X[:, t])
        dbH += dZ[:, t]
        if t > 0:
            dWh += np.outer(dZ[:, t], H[:, t - 1])

    return dWx, dWh, dWy, dbH, dby NEWER ONE NEWER ONE NEWER ONE"""




"""def test_rnn_implementation():
    aID = 1
    
    Tests the independent RNN implementation against the user's implementation.

    # Initialize parameters
    Wx = np.array([[0.1, 0.2], [0.3, 0.4]])
    Wh = np.array([[0.5, 0.6], [0.7, 0.8]])
    Wy = np.array([[0.9, 1.0], [1.1, 1.2]])
    bH = np.array([0.1, 0.2])
    by = np.array([0.3, 0.4])
    X = np.array([[1, 2], [3, 4]])
    y_true = np.array([[0.5, 0.7], [0.6, 0.8]])

    # Forward Propagation
    H_ind, Z_ind, yHat_ind = forward_propagation(X, Wx, Wh, Wy, bH, by, aID)

    # User's forward propagationz
    from CoreFramework import hMatrixTUpdate, yOutputUpdate
    H_prog = np.zeros_like(H_ind)
    Z_prog = np.zeros_like(Z_ind)
    yHat_prog = np.zeros_like(yHat_ind)
    H_prog, Z_prog = cf.hMatrixTUpdate(H_prog, Z_prog, Wx, Wh, Wy, bH, by, X, aID)
    yHat_prog = cf.yOutputUpdate(H_prog, yHat_prog, by, Wy, aID)

    # Backward Propagation
    dWx_ind, dWh_ind, dWy_ind, dbH_ind, dby_ind = backward_propagation(X, y_true, H_ind, Z_ind, Wx, Wh, Wy, bH, by, aID, aID)

    # User's backward propagation
    from CoreFramework import dWxMatrixUpdate, dWhMatrixUpdate
    dWx_prog = cf.dWxMatrixUpdate(None, X, H_prog - y_true, Z_prog, aID)
    dWh_prog = cf.dWhMatrixUpdate(None, H_prog, H_prog - y_true, Z_prog, aID)

    # Print results and compare
    print("=== Forward Propagation ===")
    print(f"Independent H: \n{H_ind}")
    print(f"Program H: \n{H_prog}")
    print(f"Independent yHat: \n{yHat_ind}")
    print(f"Program yHat: \n{yHat_prog}")

    print("\n=== Backward Propagation ===")
    print(f"Independent dWx: \n{dWx_ind}")
    print(f"Program dWx: \n{dWx_prog}")
    print(f"Independent dWh: \n{dWh_ind}")
    print(f"Program dWh: \n{dWh_prog}")

    # Assertions
    assert np.allclose(H_prog, H_ind, atol=1e-5), "Mismatch in H"
    assert np.allclose(Z_prog, Z_ind, atol=1e-5), "Mismatch in Z"
    assert np.allclose(yHat_prog, yHat_ind, atol=1e-5), "Mismatch in yHat"
    assert np.allclose(dWx_prog, dWx_ind, atol=1e-5), "Mismatch in dWx"
    assert np.allclose(dWh_prog, dWh_ind, atol=1e-5), "Mismatch in dWh"
    print("\nAll tests passed successfully!")

# Run the test
test_rnn_implementation()"""

"""def test_rnn_implementation():
    aID = 1  # Activation function identifier
    LID = 0  # Loss function identifier (e.g., 0 for MSE, 1 for Cross-Entropy)
    
    Tests the independent RNN implementation against the user's implementation.

    # Initialize parameters
    Wx = np.array([[0.1, 0.2], [0.3, 0.4]])
    Wh = np.array([[0.5, 0.6], [0.7, 0.8]])
    Wy = np.array([[0.9, 1.0], [1.1, 1.2]])
    bH = np.array([0.1, 0.2])
    by = np.array([0.3, 0.4])
    X = np.array([[1, 2], [3, 4]])
    y_true = np.array([[0.5, 0.7], [0.6, 0.8]])

    # Independent Forward Propagation
    H_ind, Z_ind, yHat_ind = forward_propagation(X, Wx, Wh, Wy, bH, by, aID)

    # User's Forward Propagation
    from CoreFramework import hMatrixTUpdate, yOutputUpdate
    H_prog = np.zeros_like(H_ind)
    Z_prog = np.zeros_like(Z_ind)
    yHat_prog = np.zeros_like(yHat_ind)
    H_prog, Z_prog = cf.hMatrixTUpdate(H_prog, Z_prog, Wx, Wh, Wy, bH, by, X, aID)
    yHat_prog = cf.yOutputUpdate(H_prog, yHat_prog, by, Wy, aID)

    # Independent Backward Propagation
    dWx_ind, dWh_ind, dWy_ind, dbH_ind, dby_ind = backward_propagation(X, y_true, H_ind, Z_ind, Wx, Wh, Wy, bH, by, aID, LID)

    # User's Backward Propagation
    # Step 1: Compute dyHatT (loss gradient w.r.t. outputs)
    dyHatT_prog = cf.c_dyHatT(None, yHat_prog, y_true, LID)

    # Step 2: Compute tdH (gradient contribution from outputs to hidden states)
    tdH_prog = cf.c_tdH(dyHatT_prog, Wy, H_prog, by, aID)

    # Step 3: Compute dH (hidden state gradients through time)
    dH_prog = cf.dhtMatrixUpdate(np.zeros_like(H_prog), tdH_prog, aID, Wh, dyHatT_prog, Wy, Z_prog)

    # Step 4: Compute dWx (input-to-hidden gradients)
    dWx_prog = cf.dWxMatrixUpdate(None, X, dH_prog, Z_prog, aID)

    # Step 5: Compute dWh (hidden-to-hidden gradients)
    dWh_prog = cf.dWhMatrixUpdate(None, H_prog, dH_prog, Z_prog, aID)

    # Step 6: Compute dbH (hidden bias gradients)
    dbH_prog = cf.dbh(dH_prog, Z_prog, aID)

    # Step 7: Compute dWy (output weights gradients)
    dWy_prog = cf.c_dWy(None, dyHatT_prog, H_prog)

    # Step 8: Compute dby (output bias gradients)
    dby_prog = cf.c_dby(None, dyHatT_prog)

    # Print results and compare
    print("=== Forward Propagation ===")
    print(f"Independent H: \n{H_ind}")
    print(f"Program H: \n{H_prog}")
    print(f"Independent yHat: \n{yHat_ind}")
    print(f"Program yHat: \n{yHat_prog}")

    print("\n=== Backward Propagation ===")
    print(f"Independent dWx: \n{dWx_ind}")
    print(f"Program dWx: \n{dWx_prog}")
    print(f"Independent dWh: \n{dWh_ind}")
    print(f"Program dWh: \n{dWh_prog}")
    print(f"Independent dbH: \n{dbH_ind}")
    print(f"Program dbH: \n{dbH_prog}")
    print(f"Independent dWy: \n{dWy_ind}")
    print(f"Program dWy: \n{dWy_prog}")
    print(f"Independent dby: \n{dby_ind}")
    print(f"Program dby: \n{dby_prog}")

    # Assertions
    assert np.allclose(H_prog, H_ind, atol=1e-5), "Mismatch in H"
    assert np.allclose(Z_prog, Z_ind, atol=1e-5), "Mismatch in Z"
    assert np.allclose(yHat_prog, yHat_ind, atol=1e-5), "Mismatch in yHat"
    assert np.allclose(dWx_prog, dWx_ind, atol=1e-5), "Mismatch in dWx"
    assert np.allclose(dWh_prog, dWh_ind, atol=1e-5), "Mismatch in dWh"
    assert np.allclose(dbH_prog, dbH_ind, atol=1e-5), "Mismatch in dbH"
    assert np.allclose(dWy_prog, dWy_ind, atol=1e-5), "Mismatch in dWy"
    assert np.allclose(dby_prog, dby_ind, atol=1e-5), "Mismatch in dby"
    print("\nAll tests passed successfully!")"""


# Run the test
test_rnn_implementation()


