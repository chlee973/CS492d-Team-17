#!/bin/bash

python train.py \
    --gpu 0 \
    --what_sketches ./data/sketches.h5 \
    --categories cat \
    --batch_size 512 \
    --train_num_steps 100000 \
    --warmup_steps 200 \
    --log_interval 2000 \
    --test_interval 20000 \
    --seed 63 \
    --default_scheduler 0 \
    --ema 0 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --num_diffusion_train_timesteps 100 \
    --num_inference_timesteps 20 \
    --sample_method ddim \
    --Nmax 96 \
    --num_res_blocks 5 \
    --num_heads 6 \
    --pen_state_loss_weight 0.1 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --add_name cat_num_heads6