from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init


class UNet(nn.Module):
    def __init__(self, ch=96, ch_mult=[1,2,3,4], attn=[], num_res_blocks=3, num_heads=4, dropout=0.1, use_cfg=False, cfg_dropout=0.1, num_classes=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        self.Nmax = ch
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(tdim)

        # classifier-free guidance
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        if use_cfg:
            assert num_classes is not None
            cdim = tdim
            self.class_embedding = nn.Embedding(num_classes+1, cdim)

        self.head = nn.Linear(2, 128)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), num_heads=num_heads))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True, num_heads=num_heads),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False, num_heads=num_heads),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), num_heads=num_heads))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Linear(128, 2)
        )

        self.pen_state_tail = nn.Sequential(
            nn.Linear(128, 2)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight, gain=1e-5)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)
        init.xavier_uniform_(self.pen_state_tail[0].weight, gain=1e-5)
        init.zeros_(self.pen_state_tail[0].bias)

    def forward(self, x, timestep, class_label=None):
        # Timestep embedding
        temb = self.time_embedding(timestep)

        if self.use_cfg and class_label is not None:
            if self.training:
                assert not torch.any(class_label == 0) # 0 for null.
                
                ######## TODO ########
                # DO NOT change the code outside this part.
                # Assignment 2-2. Implement random null conditioning in CFG training.
                raise NotImplementedError("TODO")
                #######################
            
            ######## TODO ########
            # DO NOT change the code outside this part.
            # Assignment 2-1. Implement class conditioning
            raise NotImplementedError("TODO")
            #######################

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        vectors = self.tail(h)
        pen_states = self.pen_state_tail(h)
        assert len(hs) == 0
        assert vectors.shape[0] == pen_states.shape[0]
        return vectors, pen_states
