"""DDPM diffusion model wrapper."""
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from utils import default, extract, linear_beta_schedule


class Diffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.2,
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @property
    def device(self):
        """Return the device where model parameters are stored."""
        return self.betas.device

    # ------------------------------------------------------------------
    # Public sampling API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, classes, shape, cond_weight):
        """Return list of arrays, one per reverse timestep (length T)."""
        imgs, _ = self.p_sample_loop(classes=classes, image_size=shape, cond_weight=cond_weight)
        return imgs

    @torch.no_grad()
    def sample_cross(self, classes, shape, cond_weight):
        """Return (imgs, cross_maps), both lists of length T."""
        return self.p_sample_loop(classes=classes, image_size=shape, cond_weight=cond_weight)

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_loop(self, classes, image_size, cond_weight):
        """
        Full reverse diffusion loop.

        Returns
        -------
        imgs : list[np.ndarray]  – denoised samples at each step
        cross_maps : list[np.ndarray | None]  – cross-attention maps (None if uncond)
        """
        b = image_size[0]
        device = self.device

        img = torch.randn(image_size, device=device)
        imgs = []
        cross_maps = []

        if classes is not None:
            # Classifier-free guidance: duplicate batch for cond + uncond
            n_sample = classes.shape[0]
            context_mask = torch.ones_like(classes, device=device)
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 0.0
            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                cond_weight=cond_weight,
                context_mask=context_mask,
            )
        else:
            sampling_fn = self._p_sample_uncond

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img, cross_map = sampling_fn(x=img, t=t, t_index=i)
            imgs.append(img.cpu().numpy())
            cross_maps.append(cross_map.cpu().numpy() if cross_map is not None else None)

        return imgs, cross_maps

    @torch.no_grad()
    def _p_sample_uncond(self, x, t, t_index):
        """Single reverse step without class conditioning."""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_t * (
            x - betas_t * self.model(x, time=t, classes=None) / sqrt_one_minus_t
        )

        if t_index == 0:
            return model_mean, None
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise, None

    @torch.no_grad()
    def p_sample_guided(self, x, classes, t, t_index, context_mask, cond_weight):
        """Single reverse step with classifier-free guidance."""
        batch_size = x.shape[0]
        device = self.device

        t_double = t.repeat(2).to(device)
        x_double = x.repeat(2, 1, 1, 1).to(device)

        betas_t = extract(self.betas, t_double, x_double.shape)
        sqrt_one_minus_t = extract(self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape)
        sqrt_recip_t = extract(self.sqrt_recip_alphas, t_double, x_double.shape)

        classes_masked = (classes * context_mask).long()

        # Temporarily enable attention output to capture cross-attention map
        self.model.output_attention = True
        preds, cross_map_full = self.model(x_double, time=t_double, classes=classes_masked)
        self.model.output_attention = False

        cross_map = cross_map_full[:batch_size]

        # CFG interpolation: eps = (1 + w)*eps_cond - w*eps_uncond
        x_t = (1 + cond_weight) * preds[:batch_size] - cond_weight * preds[batch_size:]

        model_mean = sqrt_recip_t[:batch_size] * (
            x - betas_t[:batch_size] * x_t / sqrt_one_minus_t[:batch_size]
        )

        if t_index == 0:
            return model_mean, cross_map

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise, cross_map

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to clean data at timestep t."""
        device = self.device
        noise = default(noise, lambda: torch.randn_like(x_start, device=device))

        sqrt_alphas_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_t * x_start + sqrt_one_minus_t * noise

    def p_losses(self, x_start, t, classes, noise=None, p_uncond=0.1):
        """Compute training loss by predicting noise at timestep t."""
        device = self.device
        noise = default(noise, lambda: torch.randn_like(x_start, device=device))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Randomly mask class labels for classifier-free guidance training
        context_mask = torch.bernoulli(
            torch.full((classes.shape[0],), 1 - p_uncond, device=device)
        )
        classes = (classes * context_mask).long()

        predicted_noise = self.model(x_noisy, t, classes)
        return F.smooth_l1_loss(noise, predicted_noise)

    def forward(self, x, classes):
        """Training forward pass: sample random timestep and compute loss."""
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        classes = classes.long()
        return self.p_losses(x, t, classes)
