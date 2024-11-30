import argparse

import numpy as np
import torch
from sketch_diffusion.dataset import pen_state_to_binary, tensor_to_pil_image
from sketch_diffusion.model import DiffusionModule
from sketch_diffusion.scheduler import DDPMScheduler
from train_pen_transformer import TransformerPenet
from pathlib import Path


import numpy as np
import torch
import json
from pathlib import Path

import numpy as np
import torch
import json
from pathlib import Path

def tensors_to_ndjson(category, batch_tensors, batch_index, save_dir):

    entries = []

    batch_size = batch_tensors.shape[0]
    for idx in range(batch_size):
        sample = batch_tensors[idx]  # Shape: (96, 3)
        if sample.is_cuda:
            sample = sample.cpu()
        sample = sample.numpy()

        dx = sample[:, 0]
        dy = sample[:, 1]
        pen_state = sample[:, 2]

        # 누적 합계를 사용하여 절대 위치 계산
        x = np.cumsum(dx)
        y = np.cumsum(dy)

        # pen_state에 따라 스트로크 분할
        strokes = []
        current_stroke_x = []
        current_stroke_y = []

        for i in range(len(pen_state)):
            current_x = x[i]
            current_y = y[i]
            current_pen = pen_state[i]

            current_stroke_x.append(current_x)
            current_stroke_y.append(current_y)

            if current_pen == 0 and i != len(pen_state) - 1:
                # 펜 리프트 감지; 현재 스트로크 종료
                strokes.append([current_stroke_x, current_stroke_y])
                current_stroke_x = []
                current_stroke_y = []

        # 마지막 스트로크 추가
        if current_stroke_x:
            strokes.append([current_stroke_x, current_stroke_y])

        # 모든 좌표를 사용하여 스케일링 계산
        all_x = np.concatenate([s[0] for s in strokes])
        all_y = np.concatenate([s[1] for s in strokes])

        min_x = np.min(all_x)
        max_x = np.max(all_x)
        min_y = np.min(all_y)
        max_y = np.max(all_y)

        # 0~255 범위로 스케일링 (비율 유지)
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_max = max(range_x, range_y)

        if range_max == 0:
            range_max = 1

        scale = 255.0 / range_max

        # 좌표 스케일링 및 이동
        for stroke in strokes:
            stroke[0] = ((np.array(stroke[0]) - min_x) * scale).tolist()
            stroke[1] = ((np.array(stroke[1]) - min_y) * scale).tolist()

        # NDJSON 엔트리 생성
        entry = {
            "word": category,
            "countrycode": "",
            "timestamp": "",
            "recognized": True,
            "key_id": f"{category}_{batch_index}_{idx}",
            "drawing": strokes
        }
        entries.append(entry)

    # 저장할 디렉토리 생성
    ndjson_dir = save_dir
    ndjson_dir.mkdir(exist_ok=True, parents=True)

    # NDJSON 파일 경로
    ndjson_path = ndjson_dir / f"{category}.ndjson"

    # NDJSON 파일에 엔트리 추가 저장
    with open(ndjson_path, 'a', encoding='utf-8') as f:
        for entry in entries:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')


def main(args):

    save_dir_image = Path(args.save_dir+"/"+args.save_category + "/image")
    save_dir_image.mkdir(exist_ok=True, parents=True)

    save_dir_ndjson = Path(args.save_dir+"/"+args.save_category + "/ndjson")
    save_dir_ndjson.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ckpt_path)
    ddpm.eval()
    ddpm = ddpm.to(device)

    if args.no_pen == 0:
        penmodel = TransformerPenet(hidden_dim = 320, num_layers=6, num_heads = 10).to(device)
        penmodel.load(args.pen_ckpt_path)
        penmodel.eval()

    num_train_timesteps = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        mode="linear",
    ).to(device)

    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    if args.sample_method == 'ddpm':
        ## DDPM Sampling
        num_inference_timesteps = num_train_timesteps
        eta = 1.0
    else:
        # DDIM Sampling
        num_inference_timesteps = args.num_inference_timesteps
        eta = 0.0

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:  # Enable CFG sampling
            assert ddpm.network.use_cfg, f"The model was not trained to support CFG."
            vectors = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                num_inference_timesteps=num_inference_timesteps,
                eta=eta,
                guidance_scale=args.cfg_scale,
            )
        else:
            vectors = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                num_inference_timesteps=num_inference_timesteps,
                eta=eta,
                guidance_scale=0.0,
            )

        if args.no_pen:
            pen_states = torch.ones((vectors.shape[0], vectors.shape[1], 1), device=vectors.device)
        else:
            pen_states = penmodel(vectors)
            pen_states = (pen_states >= 0.5).float().unsqueeze(-1)

        samples = torch.cat((vectors, pen_states), dim=-1)

        if args.sample_image==1:
            pil_images = [tensor_to_pil_image(sample) for sample in samples]
            for j, img in zip(range(sidx, eidx), pil_images):
                img.save(save_dir_image / f"{j}.png")
                print(f"Saved the {j}-th image.")
        
        if args.sample_ndjson==1:
            tensors_to_ndjson(args.save_category,samples,num_batches,save_dir_ndjson)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default='')
    parser.add_argument("--pen_ckpt_path", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='samples/')
    parser.add_argument("--save_category", type=str, default='cat')
    parser.add_argument("--beta_1", type=float, default=1e-5)
    parser.add_argument("--beta_T", type=float, default=0.005)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, choices=['ddpm', 'ddim'], default="ddim")
    parser.add_argument("--num_inference_timesteps", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--no_pen", type=float, default=0)
    parser.add_argument("--sample_image", type=float, default=0)
    parser.add_argument("--sample_ndjson", type=float, default=0)
    

    args = parser.parse_args()
    main(args)
