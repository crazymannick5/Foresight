
import pandas as pd
import numpy as np


def load_csv_to_matrix(file_path):
    df = pd.read_csv(file_path) #sets up the data frame
    stock_names = df.columns[1:] #selects the stock name starting from index 1 place 2
    data_matrix = df.iloc[:,1:].values.T # matrix creation, again using slicing, using integer position
    data_matrix = np.where(data_matrix == -1, np.nan, data_matrix) #handle missing data
    return data_matrix, stock_names #out put, data_matrix:numpy.ndarray stock_names:pandas.Index
"""This function (load_csv_to_matrix) takes the stock data in the following form:
    the first row:
    date label, and the consecutively stock labes
    
    first column: first the date label and then consecutively the dates
    """

def print_matrix_and_names(stock_names, data_matrix):
    """
    Prints the stock names and data matrix in a readable format.

    Parameters:
    stock_names (pandas.Index): Names of the stocks (columns in the original CSV).
    data_matrix (numpy.ndarray): Stock prices in matrix form.
    """
    # Print the stock names
    print("Stock Names:")
    print(list(stock_names))  # Convert to list for better readability
    print("\nData Matrix:")
    
    # Print the data matrix
    for row in data_matrix:
        print(row)


data_matrix, stock_names = load_csv_to_matrix("stock_prices_sample.csv")
print_matrix_and_names(stock_names, data_matrix)

