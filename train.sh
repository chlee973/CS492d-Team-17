#!/bin/bash

python train.py \
    --gpu 0 \
    --what_sketches ./data/sketches.h5 \
    --categories cat \
    --batch_size 512 \
    --train_num_steps 300000 \
    --warmup_steps 200 \
    --log_interval 2000 \
    --seed 63 \
    --default_scheduler 1 \
    --ema 1 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --num_diffusion_train_timesteps 100 \
    --num_inference_timesteps 20 \
    --sample_method ddim \
    --Nmax 96 \
    --num_res_blocks 3 \
    --num_heads 4 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --add_name ema-scheduler