"""Utility functions shared across the project."""
import math

import numpy as np
import torch
import torch.nn.functional as F


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


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple, device=None) -> torch.Tensor:
    """
    Gather values from buffer `a` at indices `t`, then reshape to broadcast
    over the remaining dimensions of x_shape.
    """
    if device is not None:
        a = a.to(device)
        t = t.to(device)

    out = a.gather(-1, t)
    result = out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    if device is not None:
        result = result.to(device)   # fix: was result.to(device) without assignment
    return result


def one_hot_encode(seq: str, alphabet: list[str], max_seq_len: int) -> np.ndarray:
    """One-hot encode a DNA sequence to shape (max_seq_len, len(alphabet))."""
    seq_array = np.zeros((max_seq_len, len(alphabet)), dtype=np.float32)
    for i, ch in enumerate(seq):
        if i >= max_seq_len:
            break
        seq_array[i, alphabet.index(ch)] = 1
    return seq_array


def convert_to_seq(x: np.ndarray, alphabet: list[str], seq_len: int = 200) -> str:
    """Convert model output (4, seq_len) array to nucleotide string via argmax."""
    return "".join([alphabet[i] for i in np.argmax(x.reshape(4, seq_len), axis=0)])


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Create linear noise schedule for diffusion process."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Create cosine noise schedule for diffusion process (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
