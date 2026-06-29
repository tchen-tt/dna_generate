"""UNet architecture for DNA diffusion model."""
from functools import partial

import torch
import torch.nn as nn
from memory_efficient_attention_pytorch import Attention as EfficientAttention

from layers import (
    Attention,
    Downsample,
    LayerNorm2d,
    LearnedSinusoidalPosEmb,
    LinearAttention,
    PreNorm,
    Residual,
    ResnetBlock,
    Upsample,
    default,
    exists,
)


class UNet(nn.Module):
    def __init__(
        self,
        dim: int = 200,
        init_dim: int | None = None,
        dim_mults: list = [1, 2, 4],
        channels: int = 1,
        resnet_block_groups: int = 4,
        learned_sinusoidal_dim: int = 18,
        num_classes: int = 10,
        output_attention: bool = False,
    ) -> None:
        super().__init__()

        self.channels = channels          # no longer overwritten
        self.output_attention = output_attention

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, (7, 7), padding=3)

        dims = [init_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # ------------------------------------------------------------------
        # Time + class embedding
        # ------------------------------------------------------------------
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1   # +1 for raw t prepended

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # ------------------------------------------------------------------
        # Encoder (downs)
        # ------------------------------------------------------------------
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # ------------------------------------------------------------------
        # Decoder (ups)
        # ------------------------------------------------------------------
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        # ------------------------------------------------------------------
        # Output head + cross-attention tail
        # ------------------------------------------------------------------
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)

        self.cross_attn = EfficientAttention(
            dim=200,
            dim_head=64,
            heads=1,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )
        self.norm_to_cross = nn.LayerNorm(dim * 4)   # LayerNorm(800)

    # ----------------------------------------------------------------------

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor):
        x = self.init_conv(x)
        r = x.clone()

        # Compute time embedding once
        t = self.time_mlp(time)                           # (B, time_dim)

        # Compute class embedding once and add to t (fix: was 4× redundant)
        if classes is not None:
            t = t + self.label_emb(classes)

        h = []

        # Encoder
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Decoder
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)                           # (B, 1, 4, 200)

        # Cross-attention tail: x queries against time+class context
        x_seq = x.reshape(-1, 4, 200)                    # (B, 4, 200)
        t_seq = self.norm_to_cross(t).reshape(-1, 4, 200)  # (B, 4, 200)

        cross_out = self.cross_attn(x_seq, context=t_seq)  # (B, 4, 200)
        cross_out = cross_out.view(-1, 1, 4, 200)

        x = x + cross_out

        if self.output_attention:
            return x, cross_out
        return x
