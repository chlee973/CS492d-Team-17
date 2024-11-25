import os
import argparse
import numpy as np
import torch
from sketch_diffusion.dataset import pen_state_to_binary, tensor_to_pil_image
from sketch_diffusion.model import DiffusionModule
from sketch_diffusion.scheduler import DDPMScheduler
from pathlib import Path
import cv2
import random
from torchvision import transforms
from torchvision.utils import save_image

# 기존 함수는 그대로 유지합니다.
def canvas_size_google(sketch):
    vertical_sum = np.cumsum(sketch[1:], axis=0)  
    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)
    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]  
    start_y = -ymin - sketch[0][1]
    return [int(start_x), int(start_y), int(h)+1, int(w)+1]

def scale_sketch(sketch, size=(256, 256)):
    [_, _, h, w] = canvas_size_google(sketch)
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=float)
    return sketch_rescale.astype("int16")

def draw_three(sketch, img_size=256):
    thickness = 1
    sketch = scale_sketch(sketch, (img_size, img_size))  
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += 2  
    start_y += 2
    canvas_size = max(h, w) + 6  
    canvas = np.ones((canvas_size, canvas_size, 3), dtype='uint8') * 255
    color = (0, 0, 0)
    pen_now = np.array([start_x, start_y])
    first_zero = False
    for stroke in sketch:
        delta_x_y = stroke[0:2]
        state = stroke[2]
        if first_zero:  
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        if int(state) == 1:  
            first_zero = True
        pen_now += delta_x_y
    canvas = cv2.resize(canvas, (img_size, img_size))
    return canvas

def bin_pen(x, pen_break=0.5):
    pen_states = (x[:, :, 2] >= pen_break).float()
    x[:, :, 2] = pen_states
    return x[:, :, :3]

# main을 함수화하여 재사용 가능하도록 변경
def run_test_sampling(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    images_dir = save_dir
    images_dir.mkdir(exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ckpt_path)
    ddpm.eval()
    ddpm = ddpm.to(device)

    num_train_timesteps = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=1e-4,
        beta_T=0.02,
        mode="linear",
    ).to(device)

    if args.sample_method == 'ddpm':
        num_inference_timesteps = num_train_timesteps
        eta = 1.0
    else:
        num_inference_timesteps = args.num_inference_timesteps
        eta = 0.0

    total_num_samples = args.total_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    sample_count = 0
    for i in range(num_batches):
        B = min(args.batch_size, total_num_samples - sample_count)

        if args.use_cfg:
            assert ddpm.network.use_cfg, f"The model was not trained to support CFG."
            vectors, pen_states = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                num_inference_timesteps=num_inference_timesteps,
                eta=eta,
                guidance_scale=args.cfg_scale,
            )
        else:
            vectors, pen_states = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                num_inference_timesteps=num_inference_timesteps,
                eta=eta,
                guidance_scale=0.0,
            )

        samples = torch.cat((vectors, pen_states), dim=-1)
        samples = bin_pen(samples, 0.5)
        samples = samples.cpu().numpy()
        
        for sample in samples:
            sketch_cv = draw_three(sample, img_size=args.img_size)
            image_path = images_dir / f"sample_{sample_count:06d}.png"
            cv2.imwrite(str(image_path), sketch_cv)
            sample_count += 1

    print(f"{sample_count}개의 이미지가 '{images_dir}' 폴더에 저장되었습니다.")
