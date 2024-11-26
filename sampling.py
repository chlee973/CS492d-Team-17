import argparse

import numpy as np
import torch
from sketch_diffusion.dataset import pen_state_to_binary, tensor_to_pil_image
from sketch_diffusion.model import DiffusionModule
from sketch_diffusion.scheduler import DDPMScheduler
from pathlib import Path


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
        samples = pen_state_to_binary(samples)
        pil_images = [tensor_to_pil_image(sample) for sample in samples]

        for j, img in zip(range(sidx, eidx), pil_images):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='samples/')
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, choices=['ddpm', 'ddim'], default="ddim")
    parser.add_argument("--num_inference_timesteps", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
