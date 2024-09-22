import os, json, torch
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from gluonts.time_feature import time_features_from_frequency_str


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))


class InferenceStockDataset(torch.utils.data.Dataset):
    def __init__(self, data, config, future_dates):
        self.data = data
        self.config = config
        self.future_dates = future_dates
        self.temporal_functions = time_features_from_frequency_str('1D')
        with open(os.path.join(BASE_PATH, f'data/sp500/sp500-2024Q3.json'), 'r') as f:
            self.symbol_id_map = json.load(f)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = minmax_norm(compute_differential(self.data[idx]))

        # Convert the 'Symbol' column to a static_categorical_feature
        stock_id = self.symbol_id_map[window['Symbol'][0]]
        # Check cardinality of the model in case of use_static_categorical_features
        if self.config.cardinality and self.config.cardinality[0] == 1:
            stock_id = 0

        # Apply each time feature function to the DatetimeIndex
        date_index = pd.DatetimeIndex(window['Date'])
        past_temporal_features = pd.DataFrame({f.__name__: f(date_index) for f in self.temporal_functions})
        future_temporal_features = pd.DataFrame({f.__name__: f(self.future_dates) for f in self.temporal_functions})

        # Convert DataFrame to PyTorch tensor
        window.drop(columns=['Date', 'Symbol'], inplace=True)
        window = torch.tensor(window.values, dtype=torch.float32)
        return {
            'past_time_features': torch.tensor(past_temporal_features.values, dtype=torch.float32),
            'past_values': window,
            'past_observed_mask': torch.ones(window.shape, dtype=torch.float32),
            'future_time_features': torch.tensor(future_temporal_features.values, dtype=torch.float32),
            'static_categorical_features': torch.tensor([stock_id], dtype=torch.long),
        }
    

class TrainStockDataset(torch.utils.data.Dataset):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.temporal_functions = time_features_from_frequency_str('1D')
        with open(os.path.join(BASE_PATH, f'data/sp500/sp500-2024Q3.json'), 'r') as f:
            self.symbol_id_map = json.load(f)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = minmax_norm(compute_differential(self.data[idx]), self.config.prediction_length)

        # Convert the 'Date' column to a DatetimeIndex
        date_index = pd.DatetimeIndex(window['Date'])

        # Convert the 'Symbol' column to a static_categorical_feature
        stock_id = self.symbol_id_map[window['Symbol'][0]]
        # Check cardinality of the model in case of use_static_categorical_features
        if self.config.cardinality and self.config.cardinality[0] == 1:
            stock_id = 0

        # Apply each time feature function to the DatetimeIndex
        past_temporal_features = pd.DataFrame({f.__name__: f(date_index) for f in self.temporal_functions})
        window = past_temporal_features.join(window)

        # Convert DataFrame to PyTorch tensor
        window.drop(columns=['Date', 'Symbol'], inplace=True)
        window = torch.tensor(window.values, dtype=torch.float32)

        return {
            'past_time_features': window[:self.config.context_length + self.config.prediction_length, :3],
            'past_values': window[:self.config.context_length + self.config.prediction_length, 3:],
            'past_observed_mask': torch.ones(self.config.context_length + self.config.prediction_length, 6, dtype=torch.float32),
            'future_time_features': window[-self.config.prediction_length:, :3],
            'future_values': window[-self.config.prediction_length:, 3:],
            'future_observed_mask': torch.ones(self.config.prediction_length, 6, dtype=torch.float32),
            'static_categorical_features': torch.tensor([stock_id], dtype=torch.long),
        }
    

def fetch_yf_prices_for_inference(prediction_length, symbols, end_date):
    context_length = prediction_length * 4
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d") 
    start_date = end_date - timedelta(days=365*20)

    stock_dfs = []
    for symbol in symbols:
        stock_df = yf.download(symbol, start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)
        stock_df.insert(1, 'Symbol', symbol)
        stock_dfs.append(stock_df.iloc[-prediction_length-context_length-1:].reset_index(drop=True))
    return stock_dfs


def fetch_yf_prices(symbols=None, start_date=None, end_date=None):
    if symbols is None:
        symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        symbols = symbols[0]['Symbol'].tolist()

    file_path = os.path.join(BASE_PATH,'data/sp500/sp500-2024Q3.json')
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump({symbol: i for i, symbol in enumerate(symbols)}, f)

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_date = datetime.now().date()
 
    stock_dfs = []
    for i, symbol in enumerate(symbols):
        print(f'[{i}] {symbol}', end='\t')

        file_path = os.path.join(BASE_PATH, f'data/sp500/2014-2024/{symbol}.csv')
        if os.path.exists(file_path) and start_date is None:
            print('Loading...', end='  ')
            stock_df = pd.read_csv(file_path)
            print('1 of 1 completed')

        else:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            else:
                # Use past 20 years of data
                start_date = end_date - timedelta(days=365*20)
            
            stock_df = yf.download(symbol, start=start_date, end=end_date)
            stock_df.reset_index(inplace=True)
            stock_df.insert(1, 'Symbol', symbol)

            # Save data to local storage to avoid redownloading data
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            stock_df.to_csv(file_path, index=False)
        
        stock_dfs.append(stock_df)

    return stock_dfs


def create_sliding_windows(stock_dfs, prediction_length, context_length, stride):
    train, val = [], []

    for stock_df in stock_dfs:
        num_rows = stock_df.shape[0]
        for start in range(0, num_rows, stride):
            end = start + context_length + 3*prediction_length + 1
            if end > num_rows:
                break

            train.append(stock_df.iloc[start : end - prediction_length].reset_index(drop=True))
            val.append(stock_df.iloc[start + prediction_length : end].reset_index(drop=True))

    return train, val


def minmax_norm(window, prediction_length=0):
    for col in window:
        if col != 'Date' and col != 'Symbol':
            if prediction_length:
                window_min = window[col][:-prediction_length].min()
                window_max = window[col][:-prediction_length].max()
            else:
                window_min = window[col].min()
                window_max = window[col].max()
            numerator = (window[col] - window_min)
            window[col] = numerator / (window_max - window_min)
    return window


def compute_differential(window):
    diff_window = window[window.columns.difference(['Date', 'Symbol'])].diff()
    diff_window.insert(0, 'Symbol', window['Symbol'])
    diff_window.insert(0, 'Date', window['Date'])
    diff_window = diff_window.iloc[1:].reset_index(drop=True)
    return diff_window


def revert_preprocessing(original, prediction, future_dates):
    diff_window = compute_differential(original)
    forecast = pd.DataFrame({'Date': future_dates, 'Symbol': original['Symbol'].values[0]})

    for i, col in enumerate(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']):
        unnormalized = prediction[:, i] * (diff_window[col].max() - diff_window[col].min()) + diff_window[col].min()
        forecast[col] = unnormalized.to('cpu').numpy()

    diff_columns = forecast.columns.difference(['Date', 'Symbol'])
    window = forecast[diff_columns].cumsum()
    window += original[diff_columns].iloc[-1]
    window.insert(0, 'Symbol', forecast['Symbol'])
    window.insert(0, 'Date', forecast['Date'])
    return window

