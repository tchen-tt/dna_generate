"""Neural network building blocks for the UNet."""
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn


def exists(x):
    """Check if value is not None."""
    return x is not None


def default(val, d):
    """Return val if exists, else d (or d() if callable)."""
    if exists(val):
        return val
    return d() if callable(d) else d


def l2norm(t):
    """L2 normalize tensor along last dimension."""
    return F.normalize(t, dim=-1)


# ---------------------------------------------------------------------------
# Positional / time embeddings
# ---------------------------------------------------------------------------

class LearnedSinusoidalPosEmb(nn.Module):
    """Learnable sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,)
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        # prepend raw t so the network also sees the raw timestep value
        return torch.cat((x, fouriered), dim=-1)  # (B, dim+1)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 2-D feature maps (B, C, H, W)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    """Apply normalization before the function."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class Residual(nn.Module):
    """Residual connection wrapper."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


# ---------------------------------------------------------------------------
# Up / Down sampling
# ---------------------------------------------------------------------------

def Upsample(dim: int, dim_out: int | None = None) -> nn.Sequential:
    """Upsample layer using nearest neighbor interpolation."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim: int, dim_out: int | None = None) -> nn.Conv2d:
    """Downsample layer using strided convolution."""
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


# ---------------------------------------------------------------------------
# ResNet blocks
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Conv + GroupNorm + optional FiLM conditioning + SiLU."""

    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: tuple | None = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    """ResNet block with FiLM conditioning applied to both sub-blocks."""

    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None, groups: int = 8) -> None:
        super().__init__()
        # Projects time embedding to scale+shift for each block
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)                          # (B, dim_out*2)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h, scale_shift=scale_shift)  # fix: apply FiLM to block2 as well
        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """O(n) linear attention for encoder/decoder stages."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """Full O(n²) self-attention for the UNet bottleneck."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, scale: int = 10) -> None:
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)

        q, k = map(l2norm, (q, k))
        sim = einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
