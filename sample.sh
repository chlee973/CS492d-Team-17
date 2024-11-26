#!/bin/bash

python sampling.py \
    --batch_size 4 \
    --num_samples 20 \
    --gpu 0 \
    --ckpt_path results/diffusion-ddim-11-26-023320-helicopter_numhead_resblock_train_step/last.ckpt\
    --save_dir samples/ \
    --sample_method ddim \
    --num_inference_timesteps 20 \