
import pandas as pd
import numpy as np


def load_csv_to_matrix(file_path):
    df = pd.read_csv(file_path) #sets up the data frame
    stock_names = df.columns[1:] #selects the stock name starting from index 1 place 2
    data_matrix = df.iloc[:,1:].values # matrix creation, again using slicing, using integer position
    data_matrix = np.where(data_matrix == -1, np.nan, data_matrix) #handle missing data
    return data_matrix, stock_names