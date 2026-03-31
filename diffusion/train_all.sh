#!/bin/bash
python -m diffusion.train --config configs/data_config.yaml --hidden_dim 128 --epochs 1500 --save_dir checkpoints/df1
python -m diffusion.train --config configs/data_config.yaml --hidden_dim 256 --epochs 1500 --save_dir checkpoints/df2
python -m diffusion.train --config configs/data_config.yaml --hidden_dim 512 --epochs 1500 --save_dir checkpoints/df3
python -m diffusion.train --config configs/data_config.yaml --hidden_dim 1024 --epochs 1500 --save_dir checkpoints/df4
