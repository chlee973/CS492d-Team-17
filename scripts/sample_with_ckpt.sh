#!/bin/bash

python ../sample_with_ckpt.py \
    --batch_size 4 \
    --num_samples 20 \
    --gpu 0 \
    --ckpt_path "../results/cat_top1000_step=198000.ckpt" \
    --pen_ckpt_path "../results/final_pen_cat/pen.ckpt" \
    --save_dir "../samples" \
    --save_category "cat" \
    --sample_method ddim \
    --num_inference_timesteps 100 \
    --no_pen 0 \
    --sample_image 1 \
    --sample_ndjson 0 \