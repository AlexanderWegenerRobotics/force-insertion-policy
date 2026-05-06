python -m transformer.train --config configs/data_config.yaml --seq_len 10 --hidden_dim 128 --num_layers 1 --epochs 500 --batch_size 2048 --save_dir checkpoints/transformer_s10_h128
python -m transformer.train --config configs/data_config.yaml --seq_len 20 --hidden_dim 256 --num_layers 2 --epochs 500 --batch_size 2048 --save_dir checkpoints/transformer_s20_h256
python -m transformer.train --config configs/data_config.yaml --seq_len 40 --hidden_dim 256 --num_layers 2 --epochs 500 --batch_size 1024 --save_dir checkpoints/transformer_s40_h256
