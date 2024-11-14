NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 mpiexec -n $NUM_GPUS python sample.py \
                  --model_path ./logs/model002000.pt\
                  --pen_break 0.1 \
                  --save_path ./samples \
                  --use_ddim True \
                  --log_dir ./samples \
                  --diffusion_steps 100 \
                  --noise_schedule linear \
                  --image_size 96 \
                  --num_channels 96 \
                  --num_res_blocks 3
