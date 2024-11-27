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
    """
    모든 점을 순차적으로 연결하여 스케치를 그립니다. pen_state는 무시됩니다.
    
    Parameters:
        sketch (numpy.ndarray): (M, 3) 형태의 배열, 각 행은 [x, y, pen_state].
        img_size (int): 출력 이미지 크기 (픽셀 단위).
    
    Returns:
        numpy.ndarray: 그려진 스케치 이미지 (흰색 배경에 검은 선).
    """
    # 흰색 배경의 캔버스 생성
    canvas = np.ones((img_size, img_size, 3), dtype='uint8') * 255
    
    # 스케치의 x, y 좌표 추출 및 정수형으로 변환
    points = sketch[:, :2].astype(int)
    
    # 모든 점을 순차적으로 연결
    for i in range(1, len(points)):
        start_point = tuple(points[i - 1])
        end_point = tuple(points[i])
        cv2.line(canvas, start_point, end_point, color=(0, 0, 0), thickness=1)
    
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

    # print(f"{sample_count}개의 이미지가 '{images_dir}' 폴더에 저장되었습니다.")
