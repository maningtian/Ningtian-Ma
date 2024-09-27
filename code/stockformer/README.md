## Pre-Training Stockformer Script
Example training command:

```bash
python3 code/stockformer/train.py \
--prediction_length 30 \
--lags_sequence 1 2 3 5 7 10 15 20 25 30 \
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

This will save the model under the directory `models/stockformer/<MODEL_NAME>/`

If you would like to load a model from a checkpoint directory and resume training from its current state, use the option: `--load_checkpoint <MODEL_NAME>`

A full list of options and many more training details can be found with `python3 code/stockformer/train.py --help`
