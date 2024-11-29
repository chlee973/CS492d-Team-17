#!/bin/bash

python train_transformer.py \
    --gpu 0 \
    --what_sketches ./data/sketches_rdp.h5 \
    --categories cat \
    --batch_size 128 \
    --train_num_steps 200000 \
    --warmup_steps 200 \
    --log_interval 2000 \
    --seed 63 \
    --default_scheduler 0 \
    --ema 0 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --num_diffusion_train_timesteps 1000 \
    --num_inference_timesteps 30 \
    --sample_method ddim \
    --Nmax 96 \
    --dropout 0.1 \
    --hidden_dim 256 \
    --num_layers 6 \
    --num_heads 8 \
    --add_name cat_batch128_256dim_6layers_8heads