# Stockformer: A Time Series Transformer for Stock Prediction

## Model Architecture
The Time Series Transformer implemented in Hugging Face's Transformers library is a vanilla encoder-decoder Transformer architecture adapted for time series forecasting, including stock price prediction. This model is designed to handle the complex, non-linear patterns often present in financial data.

### Model Hyperparameters
Key hyperparameters for the Time Series Transformer include:
- `prediction_length`: The number of future time steps to predict (e.g., 24 for 24 months of stock price forecasts)
- `context_length`: The number of past time steps used as input (typically set to 2x prediction_length)
- `d_model`: Dimension of the transformer layers (e.g., 32)
- `num_attention_heads`: Number of attention heads in each layer
- `num_encoder_layers` and `num_decoder_layers`: Number of layers in the encoder and decoder (e.g., 4 each)
- `dropout`: Dropout probability for regularization
- `lags_sequence`: Specifies how far back in time to look for additional features


## Data
### Data Preprocessing
For stock price prediction, several preprocessing steps are crucial:
1. **Computing Differentials**: Calculate first-order differences of stock prices to focus on price changes rather than absolute values.
2. **Stationarizing the Data**: Apply techniques like differencing or detrending to make the time series stationary, which is often necessary for accurate forecasting.
3. **Normalizing the Data**: Use techniques like min-max scaling or z-score normalization to bring all features to a similar scale.
4. **Feature Engineering**: Create additional features such as:
  - Technical indicators (e.g., Moving Averages, RSI)
  - Time-based features (e.g., day of week, month of year)
  - Lagged values of the target variable
### Sliding Window
Implement a sliding window approach with a 5:1 ratio (80:20 split):
- Use 80% of each window for input (`context_length`)
- Use 20% for prediction (`prediction_length`)
This approach allows the model to learn from historical patterns while forecasting future values.


## Training Procedure
### Preprocessing
1. **Data Splitting**: Divide the dataset into training, validation, and test sets.
2. **Feature Creation**:
  - Generate `past_time_features` and `future_time_features` for temporal context.
  - Create `static_categorical_features` (e.g., stock ticker) and `static_real_features` if applicable.
3. **Batching**: Use GluonTS to create appropriate training, validation, and test batches5.
### Training
1. Model Initialization: Instantiate the TimeSeriesTransformerForPrediction model with the configured hyperparameters.
2. **Training Loop**:
  - Implement teacher forcing, where the model uses true past values to predict the next step during training.
  - Use a suitable loss function, such as Mean Squared Error (MSE) for point forecasts or a probabilistic loss for distribution forecasts.
.3 **Optimization**:
  - Use an appropriate optimizer (e.g., Adam) with learning rate scheduling.
  - Implement early stopping based on validation loss to prevent overfitting.
**Postprocessing**
1. **Denormalization**: Convert model outputs back to the original scale of stock prices.
2. **Destationarization**: Convert model outputs back to the original values of stock prices.
3. **Evaluation**:
  - For point forecasts, use metrics like Mean Absolute Error (MAE) or Root Mean Square Error (RMSE).
  - For probabilistic forecasts, employ metrics like Continuous Ranked Probability Score (CRPS).
4. **Visualization**: Generate plots comparing predicted vs. actual stock prices over the forecast horizon.


## Inference
During inference, use autoregressive generation:
1. Feed the last value of `past_values` to the decoder.
2. Sample from the model to predict the next time step.
3. Use the prediction as input for the subsequent time step1.


## Sample Training Command:
```bash
python3 code/stockformer/train.py \
--prediction_length 30 \
--batch_size 1024 \
--num_epochs 100 \
--num_workers 8 \
--lr 5e-5 \
--wd 0.01 \
--d_model 64 \
--encoder_layers 8 \
--decoder_layers 8 \
--encoder_attention_heads 16 \
--decoder_attention_heads 16 \
--encoder_ffn_dim 64 \
--decoder_ffn_dim 64 \
--save_name <MODEL_NAME>
```
> This will save the model under the directory `models/stockformer/<MODEL_NAME>`<br/>
> If you would like to load a model from a checkpoint directory and resume training from its current state, use the option: `--load_checkpoint <MODEL_NAME>`<br/>
> A full list of options and many more training details can be found with `python3 code/stockformer/train.py --help`
