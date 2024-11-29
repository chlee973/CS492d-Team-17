from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .scheduler import BaseScheduler


class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler: BaseScheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        # x0 is of shape [B, C, 3], C is 96
        assert x0.dtype == torch.float32
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)
        if noise is None:
            noise = torch.randn_like(x0[:, :, :2], device=self.device)
        xt, noise = self.var_scheduler.q_sample(x0[:, :, :2], timestep, noise)

        noise_pred = self.network(xt, timestep, class_label)

        noise_criterion = nn.MSELoss()

        loss = noise_criterion(noise_pred, noise)
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def Nmax(self):
        return self.network.Nmax

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        num_inference_timesteps,
        eta=0.0,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 0.0,
    ):
        x_T = torch.randn([batch_size, self.Nmax, 2], device=self.device, dtype=torch.float32)

        traj = [x_T]
        step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps
        timesteps = torch.from_numpy(
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        prev_timesteps = timesteps - step_ratio
        for t, t_prev in zip(timesteps, prev_timesteps):
            x_t = traj[-1]
            noise_pred = self.network(
                x_t,
                timestep=t.to(self.device),
                class_label=class_label,
            )

            x_t_prev = self.var_scheduler.ddim_p_sample(x_t, t.to(self.device), t_prev.to(self.device), noise_pred, eta)
            
            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path, map_location="cuda:0"):
        dic = torch.load(file_path, map_location)
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
