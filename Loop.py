import numpy as np
import pandas as pd 
import Activate as af
import dyHatT as dy
import CoreFramework as cf
import DataIntake as di

class network:
    
    def __init__(self, name_input):
        self.name = name_input
        self.X = None

        self.Wx = None
        self.Wh = None
        self.Wy = None

        self.bh = None
        self.by = None

        self.Z = None



    def initialize_rnn_matrices(self, X, hidden_units, output_units):
        """
        Initializes the matrices required for an RNN, including weight matrices, 
        bias vectors, and the hidden state matrix Z.

        Args:
            X (np.ndarray): Input matrix (shape: num_features x num_time_steps).
            hidden_units (int): Number of hidden units in the RNN.
            output_units (int): Number of output units.

        Returns:
            dict: A dictionary containing the initialized matrices:
                - Wxh: Input-to-hidden weight matrix
                - Whh: Hidden-to-hidden weight matrix
                - Why: Hidden-to-output weight matrix
                - bh: Hidden bias vector
                - by: Output bias vector
                - Z: Hidden states matrix initialized to zeros
        """
        input_units = X.shape[0]  # Number of input features
        time_steps = X.shape[1]  # Number of time steps

        # Initialize weight matrices with small random values
        self.Wx = np.random.randn(hidden_units, input_units) * 0.01  # Input-to-hidden
        self.Wh = np.random.randn(hidden_units, hidden_units) * 0.01  # Hidden-to-hidden
        self.Wy = np.random.randn(output_units, hidden_units) * 0.01  # Hidden-to-output

        # Initialize bias vectors with zeros
        self.bh = np.zeros((hidden_units, 1))  # Hidden bias
        self.by = np.zeros((output_units, 1))  # Output bias

        # Initialize hidden state matrix Z with zeros
        # Z has shape: hidden_units x (time_steps + 1), accounting for t=0 initialization
        Z = np.zeros((hidden_units, time_steps + 1))

        # Return all matrices in a dictionary
        return {
            "Wx": Wx,
            "Wh": Wh,
            "Wy": Wy,
            "bh": bh,
            "by": by,
            "Z": Z
        }
    


    def startNet(self, tickers, start_date, end_date, interval, data_type, hidden_units, output_units):
        self.X = di.getX(tickers, start_date, end_date, interval, data_type)



    def forwardProp():


