import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import SketchDataModule, get_data_iterator, pen_state_to_binary, tensor_to_pil_image
from dotmap import DotMap
from model import DiffusionModule
from network import UNet
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from tqdm import tqdm

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
    if args.use_cfg:
        save_dir = Path(f"results/cfg_diffusion-{args.sample_method}-{now}")
    else:
        save_dir = Path(f"results/diffusion-{args.sample_method}-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    ds_module = SketchDataModule(
        data_path="../data/sketches.h5",
        categories=['cat'],
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
    
    network = UNet(
        T=config.num_diffusion_train_timesteps,
        ch=config.Nmax,
        ch_mult=[1, 2, 3, 4],
        attn=[1],
        num_res_blocks=3,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    ddpm = DiffusionModule(network, var_scheduler)
    ddpm = ddpm.to(config.device)

    if config.ema:
        ema_rate = 0.9999
        ema_params = [param.clone().detach().to(config.device) for param in ddpm.network.parameters()]
        for param in ema_params:
            param.requires_grad = False
    
    initial_lr = 2e-4
    lr_anneal_steps = config.train_num_steps

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=initial_lr)
    #################### Implement Scheduler
    if not config.default_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
        )

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if config.default_scheduler:
                frac_done = step / lr_anneal_steps
                frac_done = min(frac_done, 1.0)  # frac_done이 1을 넘지 않도록 제한
                lr = initial_lr * (1 - frac_done)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            if step % config.log_interval == 0:
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()
                if args.use_cfg:  # Conditional, CFG training
                    vectors, pen_states = ddpm.sample(
                        4,
                        class_label=torch.randint(1, 4, (4,)).to(config.device),
                        return_traj=False,
                    )
                else:  # Unconditional training
                    vectors, pen_states = ddpm.sample(4, return_traj=False)

                samples = torch.cat((vectors, pen_states), dim=-1)
                samples = pen_state_to_binary(samples)
                pil_images = [tensor_to_pil_image(sample) for sample in samples]
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                if config.ema:
                    original_params = [param.clone() for param in ddpm.network.parameters()]
                    for param, ema_param in zip(ddpm.network.parameters(), ema_params):
                        param.data.copy_(ema_param.data)
                    ddpm.save(f"{save_dir}/step={step}_ema.ckpt")
                    for param, original_param in zip(ddpm.network.parameters(), original_params):
                        param.data.copy_(original_param.data)
                else:
                    ddpm.save(f"{save_dir}/step={step}.ckpt")

                ddpm.train()

            img, label = next(train_it)
            img, label = img.to(config.device).to(torch.float32), label.to(torch.float32)

            if args.use_cfg:  # Conditional, CFG training
                loss = ddpm.get_loss(img, class_label=label)
            else:  # Unconditional training
                loss = ddpm.get_loss(img)
            
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if config.ema:
                for param, ema_param in zip(ddpm.network.parameters(), ema_params):
                    ema_param.data.mul_(ema_rate).add_(param.data, alpha=1 - ema_rate)
            
            if not config.default_scheduler:
                scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512) # originally 4
    
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=2000)
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--Nmax", type=int, default=96)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--ema", type=int, default=0)
    parser.add_argument("--default_scheduler", type=int, default=0)
    args = parser.parse_args()
    main(args)
