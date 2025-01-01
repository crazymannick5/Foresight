
import pandas as pd
import numpy as np


def load_csv_to_matrix(file_path):
    df = pd.read_csv(file_path)
    stock_names = df.columns[1:]
    data_matrix = df.iloc[:,1:].values