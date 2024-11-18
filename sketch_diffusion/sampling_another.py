import argparse

import numpy as np
import torch
from dataset import pen_state_to_binary, tensor_to_pil_image
from model import DiffusionModule
from scheduler import DDPMScheduler
from pathlib import Path
import cv2
import random
from torchvision import transforms
from torchvision.utils import save_image
import os

def canvas_size_google(sketch):
    vertical_sum = np.cumsum(sketch[1:], axis=0)  

    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)

    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]  
    start_y = -ymin - sketch[0][1]
    return [int(start_x), int(start_y), int(h)+1, int(w)+1]


def scale_sketch(sketch, size=(448, 448)):
    [_, _, h, w] = canvas_size_google(sketch)
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=float)
    return sketch_rescale.astype("int16")


def draw_three(sketch, window_name="google", padding=30,
               random_color=False, time=1, show=False, img_size=256):
    thickness = int(img_size * 0.025)

    sketch = scale_sketch(sketch, (img_size, img_size))  
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += thickness + 1
    start_y += thickness + 1
    canvas = np.ones((max(h, w) + 3 * (thickness + 1), max(h, w) + 3 * (thickness + 1), 3), dtype='uint8') * 255
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        color = (0, 0, 0)
    pen_now = np.array([start_x, start_y])
    first_zero = False
    for stroke in sketch:
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:  
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        if int(state) == 1:  
            first_zero = True
            if random_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = (0, 0, 0)
        pen_now += delta_x_y
    return cv2.resize(canvas, (img_size, img_size))


def bin_pen(x, pen_break=0.005):
    result = x
    print(x.shape)
    for i in range(x.size()[0]):
        for j in range(x.size()[1]):
                pen = x[i][j][2]
                if pen >= pen_break:
                    result[i][j][2] = 1
                else:
                    result[i][j][2] = 0
    return result[:, :, :3]


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

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

    total_num_samples = 8
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    all_images = []
    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:  # Enable CFG sampling
            assert ddpm.network.use_cfg, f"The model was not trained to support CFG."
            vectors, pen_states = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                guidance_scale=args.cfg_scale,
            )
        else:
            vectors, pen_states = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                guidance_scale=0.0,
            )

        samples = torch.cat((vectors, pen_states), dim=-1)
        samples = bin_pen(samples, 0.5)
        samples = samples.cpu().numpy()
        
        for sample in samples:
            sketch_cv = draw_three(sample, img_size=256)
            tensor = transforms.ToTensor()(sketch_cv)
            all_images.append(tensor)
    save_image(torch.stack(all_images), 'output.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default='results/sketch-rnn-80000/step=88000.ckpt')
    parser.add_argument("--save_dir", type=str, default='samples/')
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
