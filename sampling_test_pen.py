import argparse

import numpy as np
import torch
from sketch_diffusion.dataset import pen_state_to_binary, tensor_to_pil_image
from sketch_diffusion.model import DiffusionModule
from sketch_diffusion.scheduler import DDPMScheduler
from train_pen import Penet
from pathlib import Path


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    ddpm = DiffusionModule(None, None)
    # ddpm.load(args.ckpt_path)
    ddpm.load("results/diffusion-ddim-11-28-053924-cat_1000step_vectors_transformer_16head/last.ckpt")
    ddpm.eval()
    ddpm = ddpm.to(device)

    penmodel = Penet(dims=1, channels=96).to(device)
    penmodel.load("results/pen-state-prediction/pen.ckpt")
    penmodel.eval()

    num_train_timesteps = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=1e-4,
        beta_T=0.02,
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
        
        pen_states_ones = torch.ones((vectors.shape[0], vectors.shape[1], 1), device=vectors.device)
        pen_states = penmodel(vectors)
        binary_pen_states = (pen_states >= 0.5).float().unsqueeze(-1)

        samples1 = torch.cat((vectors, pen_states_ones), dim=-1)
        pil_images1 = [tensor_to_pil_image(sample) for sample in samples1]
        for j, img in zip(range(sidx, eidx), pil_images1):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")

        
        samples2 = torch.cat((vectors, binary_pen_states), dim=-1)
        pil_images2 = [tensor_to_pil_image(sample) for sample in samples2]
        for j, img in zip(range(sidx, eidx), pil_images2):
            img.save(save_dir / f"{j}-modified.png")
            print(f"Saved the {j}-th image.")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='samples')
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, choices=['ddpm', 'ddim'], default="ddim")
    parser.add_argument("--num_inference_timesteps", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
