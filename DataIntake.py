import numpy as np
import pandas as pd 

def fetch_stock_data(tickers, start_date, end_date, interval):
    """
    Fetches stock data for given tickers using yfinance and returns it as a DataFrame.

    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval ('1d', '1wk', '1mo', etc.).

    Returns:
        pandas.DataFrame: DataFrame containing the stock data.
    """
    try:
        # Fetch data
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by="ticker"
        )
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_data_by_type(data, data_type):
    """
    Extracts a specific type of data (e.g., 'Open', 'Close') for each ticker 
    and returns it as a DataFrame.

    Args:
        data (pandas.DataFrame): The original DataFrame from yfinance.
        data_type (str): The type of data to extract ('Open', 'High', 'Low', 'Close', 'Adj Close').

    Returns:
        pandas.DataFrame: DataFrame with tickers as columns and specified data as rows.
    """
    if data_type not in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        raise ValueError("Invalid data_type. Choose from 'Open', 'High', 'Low', 'Close', 'Adj Close'.")
    
    try:
        # Check if DataFrame is multi-indexed (multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # Extract specified data type for all tickers
            extracted_data = data.xs(key=data_type, axis=1, level=1)
        else:
            # Single ticker; keep the data as is
            extracted_data = data[[data_type]]
        
        return extracted_data
    except KeyError as e:
        print(f"Key error: {e}")
        return None


def fill_missing_values(data):
    """
    Fills missing values in a DataFrame by using the average of the values on 
    either side of the missing value (time-wise).

    Args:
        data (pandas.DataFrame): DataFrame with tickers as columns and a datetime 
                                 index as rows.

    Returns:
        pandas.DataFrame: DataFrame with missing values filled.
    """
    try:
        # Interpolate missing values with linear method (time-based average)
        filled_data = data.interpolate(method='linear', axis=0)
        
        return filled_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def dataframe_to_matrix(data):
    """
    Converts a DataFrame with tickers as columns and time steps as rows into 
    a NumPy matrix where each column is a time step and each row is a ticker.

    Args:
        data (pandas.DataFrame): DataFrame with tickers as columns and time steps as rows.

    Returns:
        tuple: A NumPy matrix (tickers as rows, time steps as columns) and a list of tickers.
    """
    try:
        # Ensure data is sorted by columns (tickers) and rows (time steps)
        data = data.sort_index(axis=1).sort_index(axis=0)

        # Convert DataFrame to NumPy matrix
        matrix = data.values.T  # Transpose so rows = tickers, columns = time steps

        # Extract the tickers (column names) in order
        tickers = data.columns.tolist()

        return matrix, tickers
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    

def getX(tickers, start_date, end_date, interval, data_type):
    X = dataframe_to_matrix(fill_missing_values(extract_data_by_type(fetch_stock_data(tickers, start_date, end_date, interval),data_type)))
    return X
    

def defaultGetX():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    interval = "1d"
    data_type = "Adj Close"
    X = getX(tickers, start_date, end_date, interval, data_type)
    return X