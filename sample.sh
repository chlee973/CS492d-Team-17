#!/bin/bash

python sampling.py \
    --batch_size 4 \
    --num_samples 20 \
    --gpu 0 \
    --ckpt_path path_to_ckpt \
    --save_dir samples/ \
    --sample_method ddim \
    --num_inference_timesteps 50 \