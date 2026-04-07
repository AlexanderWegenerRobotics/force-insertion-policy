python -m cvae.train --config configs/data_config.yaml --hidden_dim 128 --latent_dim 8 --epochs 1500 --batch_size 32768 --save_dir checkpoints/cvae1
python -m cvae.train --config configs/data_config.yaml --hidden_dim 256 --latent_dim 16 --epochs 1500 --batch_size 32768 --save_dir checkpoints/cvae2
python -m cvae.train --config configs/data_config.yaml --hidden_dim 512 --latent_dim 16 --epochs 1500 --batch_size 32768 --save_dir checkpoints/cvae3
python -m cvae.train --config configs/data_config.yaml --hidden_dim 1024 --latent_dim 32 --epochs 1500 --batch_size 32768 --save_dir checkpoints/cvae4
