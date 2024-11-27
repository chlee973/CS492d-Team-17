#!/bin/bash

python sampling.py \
    --batch_size 4 \
    --num_samples 20 \
    --gpu 0 \
    --ckpt_path results/diffusion-ddim-11-27-070834-cat_1000step_vectors_transformer/last.ckpt \
    --save_dir samples/ \
    --sample_method ddim \
    --num_inference_timesteps 50 \