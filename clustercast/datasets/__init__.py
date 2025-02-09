import os
import pandas as pd

# helper function to grab the path of the datasets
def _get_data_path(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'data', filename)

# loads in the store sales dataset
def load_store_sales():
    path = _get_data_path('store_sales.csv')
    return pd.read_csv(path, parse_dates=['YM'])

# loads in the airline passengers dataset
def load_airline_passengers():
    path = _get_data_path('airline_passengers.csv')
    return pd.read_csv(path, parse_dates=['YM'])