import pandas as pd
import numpy as np

def bridge_imputation(data_matrix):
    """
    Fills missing values in a data matrix using the average of the points 
    before and after. If only one consecutive point exists, uses that value.

    Parameters:
    data_matrix (numpy.ndarray): Stock prices matrix (rows: stocks, cols: time steps).

    Returns:
    numpy.ndarray: Data matrix with missing values filled.
    """
    for row in data_matrix:
        for i in range(len(row)):
            if np.isnan(row[i]):  # Check for missing value
                # Find the previous and next valid values
                prev_val = row[i - 1] if i > 0 and not np.isnan(row[i - 1]) else None
                next_val = row[i + 1] if i < len(row) - 1 and not np.isnan(row[i + 1]) else None
                
                # Fill based on available values
                if prev_val is not None and next_val is not None:
                    row[i] = (prev_val + next_val) / 2  # Average of before and after
                elif prev_val is not None:
                    row[i] = prev_val  # Use the previous value
                elif next_val is not None:
                    row[i] = next_val  # Use the next value
                # If both are None, leave as NaN (handle later if needed)
    return data_matrix


import numpy as np

def standardize_data_matrix(data_matrix):
    """
    Standardizes each row (stock) of the data matrix and tracks the mean and standard deviation
    for each row to enable de-transformation.

    Parameters:
    data_matrix (numpy.ndarray): Stock prices matrix (rows: stocks, cols: time steps).

    Returns:
    tuple:
        standardized_matrix (numpy.ndarray): Standardized data matrix.
        stats (list of dict): List of dictionaries containing mean and stddev for each row.
    """
    standardized_matrix = np.empty_like(data_matrix, dtype=np.float64)
    stats = []

    for i, row in enumerate(data_matrix):
        mean = np.nanmean(row)  # Compute mean, ignoring NaN
        std = np.nanstd(row)    # Compute stddev, ignoring NaN
        
        # Avoid division by zero for rows with constant values
        if std > 0:
            standardized_matrix[i] = (row - mean) / std
        else:
            standardized_matrix[i] = row - mean  # Standardize to 0 if stddev is 0
        
        # Store mean and stddev for de-transformation
        stats.append({"mean": mean, "std": std})
    
    return standardized_matrix, stats

def round_data_matrix(data_matrix, decimals=3):
    """
    Rounds all values in the data matrix to a specified number of decimal places.

    Parameters:
    data_matrix (numpy.ndarray): The data matrix to round.
    decimals (int): Number of decimal places to round to (default is 3).

    Returns:
    numpy.ndarray: Rounded data matrix.
    """
    return np.round(data_matrix, decimals=decimals)

def CleanUp(data_matrix):
    data_matrix1 = bridge_imputation(data_matrix)
    data_matrix1, stats = standardize_data_matrix(data_matrix1)
    return data_matrix1, stats
    