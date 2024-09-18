import os, argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn.functional as F
from transformers import TimeSeriesTransformerForPrediction

from stockformer.config import StockformerConfig
from stockformer.data import InferenceStockDataset, fetch_yf_prices_for_inference, revert_preprocessing


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))


def load_args():
    parser = argparse.ArgumentParser(description="Script to train Stockformer")
    parser.add_argument(
        "--symbols",
        required=True,
        nargs="+",
        type=str,
        help="The stock trading symbol(s) for inference. Must be symbol(s) included in S&P500."
    )
    parser.add_argument(
        "--load_checkpoint",
        required=True,
        type=str,
        help="The directory name to load the checkpoint model for inferencing"
    )
    parser.add_argument(
        "--device_ids",
        default=[],
        nargs="+",
        type=int,
        help="The IDs of the device on the machine if using distributed inferencing"
    )
    return parser.parse_args()


def init_config(checkpoint):
    pretrained_path = os.path.join(BASE_PATH, 'models/stockformer/', checkpoint)
    if os.path.exists(pretrained_path):
        print(f'Loading config checkpoint from path: {pretrained_path}')
        config_dict = torch.load(os.path.join(BASE_PATH, pretrained_path, f'{checkpoint}-config.pth'))
        return StockformerConfig.from_dict(config_dict)
    else:
        raise Exception(f'The model checkpoint was not found under: {pretrained_path}')


def init_model(config, checkpoint):
    model = TimeSeriesTransformerForPrediction(config).to(config.device)
    pretrained_path = os.path.join(BASE_PATH, 'models/stockformer/', checkpoint)
    if os.path.exists(pretrained_path):
        print(f'Loading model checkpoint from path: {pretrained_path}')
        model_state_dict = torch.load(os.path.join(BASE_PATH, pretrained_path, f'{checkpoint}-state.pth'), map_location=config.device)
        if "module." in list(model_state_dict.keys())[0]:
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        return model
    else:
        raise Exception(f'The model checkpoint was not found under: {pretrained_path}')


def predict(symbols, prediction_length, model, config, end_date=datetime.now().date()):
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500 = sp500[0]['Symbol'].tolist()
    for symbol in symbols:
        if not symbol in sp500:
            raise Exception("Symbol Not Supported - S&P500 Stocks Only!")
    stock_dfs = fetch_yf_prices_for_inference(prediction_length, symbols, end_date)
    future_dates = pd.date_range(start=pd.Timestamp(end_date), periods=prediction_length, freq='B')

    dataset = InferenceStockDataset(stock_dfs, config, future_dates)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)

    model.eval()
    forecasts = []
    for i, batch in enumerate(dataloader):
        with torch.cuda.amp.autocast():
            outputs = model.generate(
                past_time_features=batch['past_time_features'].to(config.device), # torch.Size([1, 150, 3])
                past_values=batch['past_values'].to(config.device), # torch.Size([1, 150, 6])
                past_observed_mask=batch['past_observed_mask'].to(config.device), # torch.Size([1, 150, 6])
                future_time_features=batch['future_time_features'].to(config.device), # torch.Size([1, 30, 3])
                static_categorical_features=batch['static_categorical_features'].to(config.device) if config.use_static_categorical_features else None, # torch.Size([1, 1])
            )
            prediction = outputs.sequences.mean(dim=1).squeeze()
            # print(outputs.sequences.quantile(0.025, dim=1))
            # print(outputs.sequences.quantile(0.975, dim=1))
        forecast = pd.concat([dataset.data[i], revert_preprocessing(dataset.data[i], prediction, future_dates)], ignore_index=True)
        # Drop the first `prediction_length` due to inference with lags_sequence
        forecast = forecast.iloc[prediction_length:]
        forecasts.append(forecast)
    return forecasts


def main():
    args = load_args()

    config = init_config(args.load_checkpoint)
    model = init_model(config, args.load_checkpoint)
    
    forecasts = predict(args.symbols, model, config)

    for i, forecast in enumerate(forecasts):
        print(args.symbols[i])
        print(forecast)


if __name__ == '__main__':
    main()
