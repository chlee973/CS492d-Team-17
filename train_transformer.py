import argparse
import json
from datetime import datetime
from pathlib import Path
import os
from cleanfid import fid


import matplotlib
import matplotlib.pyplot as plt
import torch
from sketch_diffusion.dataset import SketchDataModule, get_data_iterator, pen_state_to_binary, tensor_to_pil_image
from dotmap import DotMap
from sketch_diffusion.model import DiffusionModule
from sketch_diffusion.network import UNet
from sketch_diffusion.transformer_network import TransformerModel
from pytorch_lightning import seed_everything
from sketch_diffusion.scheduler import DDPMScheduler
from tqdm import tqdm
import subprocess
from PIL import Image
from sampling_test import run_test_sampling
from sampling_another import run_another_sampling



matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    save_dir = Path(f"results/diffusion-{args.sample_method}-{now}-{args.add_name}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    ds_module = SketchDataModule(
        data_path=args.what_sketches,
        categories=config.categories,
        Nmax=config.Nmax,
        label_offset=1,
        batch_size=config.batch_size,
        num_workers=4,
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    # Set up the scheduler
    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode="linear",
    )
    # check
    print(var_scheduler.register_buffer)
    
    network = TransformerModel(
        d_model=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        T=1000,
        max_len=config.Nmax
    )

    ddpm = DiffusionModule(network, var_scheduler)
    ddpm = ddpm.to(config.device)

    initial_lr = 2e-4
    weight_decay = 1e-5

    optimizer = torch.optim.AdamW(ddpm.network.parameters(), lr=initial_lr, weight_decay=weight_decay)
    

    step = 0

    if args.resume_ckpt:
        ddpm.load(args.resume_ckpt, map_location=config.device)
        step = args.resume_step
        print(f"Resumed from checkpoint: {args.resume_ckpt} at step {step}")

    if config.ema:
        ema_rate = 0.9999
        ema_params = [param.clone().detach().to(config.device) for param in ddpm.network.parameters()]
        for param in ema_params:
            param.requires_grad = False


    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:

            if step % config.log_interval == 0 and step!=0:
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                if config.sample_method == 'ddpm':
                    eta = 1.0
                    num_inference_timesteps = config.num_diffusion_train_timesteps
                else:
                    # ddim
                    eta = 0.0
                    num_inference_timesteps = config.num_inference_timesteps

                vectors = ddpm.sample(
                    8, 
                    num_inference_timesteps=num_inference_timesteps,
                    eta=eta,
                    return_traj=False
                )
                pen_states = torch.ones((vectors.shape[0], vectors.shape[1], 1), device=vectors.device)
                samples = torch.cat((vectors, pen_states), dim=-1)
                pil_images = [tensor_to_pil_image(sample, show_hidden=True) for sample in samples]

                widths, heights = zip(*(img.size for img in pil_images))
                total_width = sum(widths)
                max_height = max(heights)
                new_image = Image.new("RGB", (total_width, max_height))
                x_offset = 0
                for img in pil_images:
                    new_image.paste(img, (x_offset, 0))
                    x_offset += img.width
                new_image.save(save_dir / f"step={step}-total-1.png")
                ddpm.save(f"{save_dir}/step={step}.ckpt")
                
                if step % config.test_interval == 0:
                    args_test = argparse.Namespace(
                        ckpt_path=f"{save_dir}/last.ckpt",
                        save_dir=f"{save_dir}/step={step}-test",
                        sample_method="ddim",
                        gpu=0,
                        batch_size=4,
                        total_samples=8,
                        img_size=256,
                        use_cfg=False,
                        num_inference_timesteps=20,
                        cfg_scale=7.5,
                    )

                    run_test_sampling(args_test)
                    fdir1="./sketch_data/cat/images_test"
                    fdir2=f"{save_dir}/step={step}-test"
                    
                    # compute FID
                    score_fid = fid.compute_fid(fdir1, fdir2)
                    score_kid = fid.compute_kid(fdir1, fdir2)

                    print("========================")
                    print(f"- FID score: {score_fid}")
                    print(f"- KID score: {score_kid}")
                    print("========================")
                ddpm.train()

            img, label = next(train_it)
            img, label = img.to(config.device).to(torch.float32), label.to(torch.float32)

            loss = ddpm.get_loss(img)
            
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if config.ema:
                for param, ema_param in zip(ddpm.network.parameters(), ema_params):
                    ema_param.data.mul_(ema_rate).add_(param.data, alpha=1 - ema_rate)
            
            losses.append(loss.item())

            step += 1
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    # DataLoader
    parser.add_argument("--what_sketches", type=str, default="./data/sketches.h5") # 데이터 종류
    parser.add_argument('--categories', nargs='+', type=str)    


    # Trainer & Logger & Scheduler
    parser.add_argument("--batch_size", type=int, default=512) # originally 4
    parser.add_argument("--train_num_steps", type=int, default=500000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=2000)
    parser.add_argument("--test_interval", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--default_scheduler", type=int, default=0)
    parser.add_argument("--ema", type=int, default=0)
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--resume_step", type=int, default=None)

    # Diffusion Scheduler
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--num_diffusion_train_timesteps", type=int, default=100)#100 origin
    parser.add_argument("--num_inference_timesteps", type=int, default=20)#100 origin
    parser.add_argument("--sample_method", type=str, choices=['ddpm', 'ddim'], default="ddim")

    # Network
    parser.add_argument("--Nmax", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--add_name", type=str, default="transformer")

    args = parser.parse_args()
    main(args)
