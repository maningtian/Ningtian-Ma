
Example training command:

```bash
python3 code/stockformer/train.py \
--prediction_length 30 \
--stride 15 \
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
--save_name <NAME_FOR_MODEL>
```
