#!/bin/bash

python ../sample_with_ckpt.py \
    --batch_size 4 \
    --num_samples 20 \
    --gpu 0 \
    --ckpt_path "../results/diffusion-ddim-11-29-134407-cat_batch128_256dim_6layers_8heads/step=150000.ckpt" \
    --pen_ckpt_path "../results/final_pen_cat/pen.ckpt" \
    --save_dir "../samples" \
    --save_category "cat" \
    --sample_method ddim \
    --num_inference_timesteps 200 \
    --no_pen 1 \
    --sample_image 1 \
    --sample_ndjson 1 \