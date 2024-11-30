#!/bin/bash

python train.py \
    --gpu 0 \
    --what_sketches ./data/sketches_rdp.h5 \
    --categories cat \
    --batch_size 256 \
    --train_num_steps 120000 \
    --warmup_steps 200 \
    --log_interval 2000 \
    --seed 63 \
    --default_scheduler 0 \
    --ema 0 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --num_diffusion_train_timesteps 500 \
    --num_inference_timesteps 30 \
    --sample_method ddim \
    --Nmax 96 \
    --num_res_blocks 4 \
    --num_heads 6 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --add_name cat_1000step_vectors_transformer