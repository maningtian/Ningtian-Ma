from typing import List, Optional, Union
from transformers import TimeSeriesTransformerConfig

class StockformerConfig(TimeSeriesTransformerConfig):
    def __init__(
        self,
        symbol: Optional[str] = None,
        prediction_length: Optional[int] = None,
        context_length: Optional[int] = None,
        stride: Optional[int] = None,
        use_static_categorical_features: bool = False,
        batch_size: int = 1,
        num_epochs: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        distribution_output: str = "student_t",
        loss: str = "nll",
        input_size: int = 6,
        lags_sequence: List[int] = [1, 2, 3, 5, 7, 10, 15, 20, 30],
        scaling: Optional[Union[str, bool]] = "mean",
        num_dynamic_real_features: int = 0,
        num_static_categorical_features: int = 0,
        num_static_real_features: int = 0,
        num_time_features: int = 3,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        encoder_ffn_dim: int = 32,
        decoder_ffn_dim: int = 32,
        encoder_attention_heads: int = 2,
        decoder_attention_heads: int = 2,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        is_encoder_decoder: bool = True,
        activation_function: str = "gelu",
        d_model: int = 64,
        dropout: float = 0.1,
        encoder_layerdrop: float = 0.1,
        decoder_layerdrop: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        num_parallel_samples: int = 100,
        init_std: float = 0.02,
        use_cache=True,
        device: str = 'cpu',
        **kwargs,
    ):
        self.symbol = symbol
        self.stride = stride
        self.use_static_categorical_features = use_static_categorical_features
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        super().__init__(
            prediction_length,
            context_length,
            distribution_output,
            loss,
            input_size,
            lags_sequence,
            scaling,
            num_dynamic_real_features,
            num_static_categorical_features,
            num_static_real_features,
            num_time_features,
            cardinality,
            embedding_dimension,
            encoder_ffn_dim,
            decoder_ffn_dim,
            encoder_attention_heads,
            decoder_attention_heads,
            encoder_layers,
            decoder_layers,
            is_encoder_decoder,
            activation_function,
            d_model,
            dropout,
            encoder_layerdrop,
            decoder_layerdrop,
            attention_dropout,
            activation_dropout,
            num_parallel_samples,
            init_std,
            use_cache,
            **kwargs,
        )
