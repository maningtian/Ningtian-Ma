import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import TimeSeriesTransformerForPrediction
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import StockformerConfig
from data import TrainStockDataset, fetch_yf_prices, create_sliding_windows


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))


def load_args():
    parser = argparse.ArgumentParser(description="Script to train Stockformer")
    # Model params
    parser.add_argument(
        "--symbol",
        required=False,
        type=str,
        help="The stock trading symbol for the model to train on"
    )
    parser.add_argument(
        "--prediction_length",
        required=True,
        type=int,
        help="The prediction length for the decoder. In other words, the prediction horizon of the model (in trading days)"
    )
    parser.add_argument(
        "--context_length",
        required=False,
        type=int,
        help="The context length for the encoder. Default is `4 * prediction_length`"
    )
    parser.add_argument(
        "--stride",
        required=False,
        type=int,
        help="The stride for sliding windows over the time-series data. Default is `2 * prediction_length`"
    )
    parser.add_argument(
        "--use_static_categorical_features",
        action="store_true",
        help="The model will use a static categorical feature (symbol) for each sliding windows over the time-series data"
    )
    # Training params
    parser.add_argument(
        "--end_date",
        required=False,
        type=str,
        help="The date, in format `YYYY-MM-DD`, up to which the training data will accumulate sliding windows of. Default is `present`"
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="The batch size for the training loop"
    )
    parser.add_argument(
        "--num_epochs",
        default=1,
        type=int,
        help="The number of epochs for the training loop"
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="The number of workers for data pre-loading"
    )
    # Hyper params
    parser.add_argument(
        "--learning_rate",
        "--lr",
        default=5e-5,
        type=float,
        help="The learning rate for the training loop"
    )
    parser.add_argument(
        "--weight_decay",
        "--wd",
        default=0.01,
        type=float,
        help="The weight decay for the training loop"
    )
    parser.add_argument(
        "--lags_sequence",
        default=[1, 2, 3, 5, 7, 10, 15, 20, 30],
        nargs="+",
        type=int,
        help="The lags of the input time series as covariates often dictated by the frequency of the data. Can be tuned to capture very short-term, medium-term, and long-term dependencies. Default is `[1, 2, 3, 5, 7, 10, 14, 21]`"
    )
    parser.add_argument(
        "--d_model",
        default=64,
        type=int,
        help="Dimensionality of the transformer layers"
    )
    parser.add_argument(
        "--encoder_layers",
        default=2,
        type=int,
        help="Number of encoder layers"
    )
    parser.add_argument(
        "--decoder_layers",
        default=2,
        type=int,
        help="Number of encoder layers"
    )
    parser.add_argument(
        "--encoder_attention_heads",
        default=2,
        type=int,
        help="Number of attention heads for each attention layer in the Transformer encoder"
    )
    parser.add_argument(
        "--decoder_attention_heads",
        default=2,
        type=int,
        help="Number of attention heads for each attention layer in the Transformer decoder"
    )
    parser.add_argument(
        "--encoder_ffn_dim",
        default=32,
        type=int,
        help="Dimension of the feed-forward layer in encoder"
    )
    parser.add_argument(
        "--decoder_ffn_dim",
        default=32,
        type=int,
        help="Dimension of the feed-forward layer in decoder"
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="The dropout probability for all fully connected layers in the encoder, and decoder"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        type=str,
        help="Device to train on"
    )
    parser.add_argument(
        "--device_ids",
        default=[],
        nargs="+",
        type=int,
        help="The IDs of the device on the machine if using distributed training"
    )
    parser.add_argument(
        "--load_checkpoint",
        required=False,
        type=str,
        help="The directory name to load the checkpoint model for training"
    )
    parser.add_argument(
        "--save_name",
        required=False,
        type=str,
        help="The directory name to save the model checkpoint after training. It will be saved in the files: `models/stockformer/<save_name>/<save_name>-config.pth` and `checkpoints/<save_name>/<save_name>-state.pth`"
    )
    return parser.parse_args()


def train(model, config, args, train_dataset, val_dataset, save_name):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=args.num_workers, shuffle=True)
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    train_loss = []
    val_loss = []
    for epoch in range(config.num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Training"):
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                outputs = model(
                    past_time_features=batch['past_time_features'].to(config.device), # torch.Size([1, 150, 3])
                    past_values=batch['past_values'].to(config.device), # torch.Size([1, 150, 6])
                    past_observed_mask=batch['past_observed_mask'].to(config.device), # torch.Size([1, 150, 6])
                    future_time_features=batch['future_time_features'].to(config.device), # torch.Size([1, 30, 3])
                    future_values=batch['future_values'].to(config.device), # torch.Size([1, 30, 6])
                    future_observed_mask=batch['future_observed_mask'].to(config.device), # torch.Size([1, 30, 6])
                    static_categorical_features=batch['static_categorical_features'].to(config.device) if config.use_static_categorical_features else None, # torch.Size([1, 1])
                )
                # Calculate loss
                loss = outputs.loss
                epoch_train_loss += loss.item()
            
             # Scale the loss and perform backward pass
            scaler.scale(loss).backward()

            # Gradient clipping (optional but common for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Unscale gradients and perform optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Validation"):
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(
                        past_time_features=batch['past_time_features'].to(config.device), # torch.Size([1, 150, 3])
                        past_values=batch['past_values'].to(config.device), # torch.Size([1, 150, 6])
                        past_observed_mask=batch['past_observed_mask'].to(config.device), # torch.Size([1, 150, 6])
                        future_time_features=batch['future_time_features'].to(config.device), # torch.Size([1, 30, 3])
                        future_values=batch['future_values'].to(config.device), # torch.Size([1, 30, 6])
                        future_observed_mask=batch['future_observed_mask'].to(config.device), # torch.Size([1, 30, 6])
                        static_categorical_features=batch['static_categorical_features'].to(config.device) if config.use_static_categorical_features else None, # torch.Size([1, 1])
                    )
                    # Calculate loss
                    loss = outputs.loss
                    epoch_val_loss += loss.item()
        
        # Adjust the learning rate based on validation loss
        scheduler.step(epoch_val_loss)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"\tTraining Loss: {avg_train_loss:.4f}")
        print(f"\tValidation Loss: {avg_val_loss:.4f}")
        print(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}")

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
    
    if save_name:
        # Plot and save loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss) + 1), train_loss, 'b-', label='Training Loss')
        plt.plot(range(1, len(train_loss) + 1), val_loss, 'r-', label='Validation Loss')
        plt.title(f'Training vs Validation Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.join(BASE_PATH, 'visuals'), exist_ok=True)
        plt.savefig(os.path.join(BASE_PATH, f'visuals/{save_name}.png'), format='png')


def main():
    args = load_args()

    if not args.context_length:
        args.context_length = 4 * args.prediction_length
    if not args.stride:
        args.stride = 2 * args.prediction_length
    if not args.symbol:
        args.use_static_categorical_features = True

    print('Preprocessing Dataset...')
    start_time = time.time()
    stock_df = fetch_yf_prices(symbols=[args.symbol] if args.symbol else None, end_date=args.end_date)
    print('Loaded Yahoo Finance S&P500 Data (Elapsed Time):\t', time.time() - start_time, 'seconds')
    start_time = time.time()
    train_data, val_data = create_sliding_windows(stock_df, args.prediction_length, args.context_length, args.stride)
    print('Created Sliding Windows (Elapsed Time):\t', time.time() - start_time, 'seconds')

    print('\nPreparing Model Configuration...')
    if args.load_checkpoint:
        pretrained_path = os.path.join(BASE_PATH, 'models/stockformer/', args.load_checkpoint)
        if os.path.exists(pretrained_path):
            print(f'\tLoading config checkpoint from path: {pretrained_path}')
            config_dict = torch.load(os.path.join(BASE_PATH, pretrained_path, f'{args.load_checkpoint}-config.pth'))
            config = StockformerConfig.from_dict(config_dict)
        else:
            raise Exception(f'The model checkpoint was not found under: {pretrained_path}')
    else:
        if args.use_static_categorical_features:
            num_static_categorical_features = 1
            cardinality = [len(stock_df)]
            embedding_dimension = [2]
        else:
            num_static_categorical_features = 0
            cardinality = None
            embedding_dimension = None
        config = StockformerConfig(
            symbol=args.symbol,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            stride=args.stride,
            use_static_categorical_features=args.use_static_categorical_features,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lags_sequence=args.lags_sequence,
            input_size=6,
            num_time_features=3,
            num_dynamic_real_features=0,
            num_static_categorical_features=num_static_categorical_features,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            d_model=args.d_model,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            encoder_attention_heads=args.encoder_attention_heads,
            decoder_attention_heads=args.decoder_attention_heads,
            encoder_ffn_dim=args.encoder_ffn_dim,
            decoder_ffn_dim=args.decoder_ffn_dim,
            dropout=args.dropout,
            device=args.device,
        )
    for attr, value in vars(config).items():
        print(f"{attr}: {value}")

    print('\nIntializing Dataset...')
    train_dataset = TrainStockDataset(train_data, config)
    val_dataset = TrainStockDataset(val_data, config)
    print('\tDone.')
    print(f'\tTraining Dataset Length: {len(train_dataset)}')
    print(f'\tValidation Dataset Length: {len(val_dataset)}')

    print('\nInitializing Model...')
    model = TimeSeriesTransformerForPrediction(config).to(config.device)

    if args.load_checkpoint:
        pretrained_path = os.path.join(BASE_PATH, 'models/stockformer/', args.load_checkpoint)
        if os.path.exists(pretrained_path):
            print(f'\tLoading model checkpoint from path: {pretrained_path}')
            model_state_dict = torch.load(os.path.join(BASE_PATH, pretrained_path, f'{args.load_checkpoint}-state.pth'), map_location=config.device)
            if "module." in list(model_state_dict.keys())[0]:
                model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
            model.load_state_dict(model_state_dict)
            print('\tResuming training from this model state dict and config...')
        else:
            raise Exception(f'The model checkpoint was not found under: {pretrained_path}')

    if len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    print('\nStarting Training...')
    train(model, config, args, train_dataset, val_dataset, args.save_name)

    if args.save_name:
        print(f'\nSaving Model to {args.save_name}')
        model.cpu()
        torch.cuda.empty_cache()
        save_path = os.path.join(BASE_PATH, 'models/stockformer/', args.save_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(config.to_dict(), os.path.join(save_path, f'{args.save_name}-config.pth'))
        torch.save(model.state_dict(), os.path.join(save_path, f'{args.save_name}-state.pth'))
        print('\tDone.\n')


if __name__ == '__main__':
    main()