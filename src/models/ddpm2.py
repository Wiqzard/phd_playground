import math
import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torchvision import transforms

# If you have any custom modules, import them, e.g.:
# from .components.ddpm_helpers import get_beta_schedule, sample_timestep, ...
# from .components.plotting import plot_samples, plot_trajectory, store_trajectories
# from .utils import get_wandb_logger


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPMLitModule(LightningModule):
    """
    A Lightning module for DDPM (Denoising Diffusion Probabilistic Models).
    This closely mirrors the structure of CFMLitModule, but for diffusion.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        beta_schedule: str = "linear",
        num_timesteps: int = 1000,
        ema_decay: float = 0.9999,
        dim: Any = (3, 32, 32),
        plot: bool = False,
        nice_name: str = "DDPM",
    ) -> None:
        """
        Args:
            model: The UNet or other backbone that predicts noise \epsilon_\theta(x_t, t).
            optimizer: The torch optimizer to be used with the model's parameters.
            scheduler: (Optional) A learning rate scheduler.
            beta_schedule: The schedule for betas. (e.g., 'linear', 'cosine', etc.)
            num_timesteps: Number of diffusion timesteps (T).
            ema_decay: EMA decay for model parameters (if using EMA).
            dim: The dimension of the input (e.g., (3, 32, 32) for CIFAR-10).
            plot: Whether to log intermediate images/plots during validation/test.
            nice_name: A short name for logging or printing.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "optimizer", "scheduler"], logger=False
        )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_timesteps = num_timesteps
        self.dim = dim
        self.is_image = isinstance(dim, tuple)
        self.plot = plot

        # Define the forward diffusion parameters (betas, alphas, etc.)
        # For example:
        self.betas = self.get_beta_schedule(beta_schedule, num_timesteps)
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
        )

        # Store them as buffers for sampling on GPU
        self.register_buffer("betas_t", self.betas)
        self.register_buffer("alphas_t", alphas)
        self.register_buffer("alphas_cumprod_t", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev_t", alphas_cumprod_prev)

        # Optionally, set up Exponential Moving Average of parameters
        self.ema_decay = ema_decay
        self.model_ema = None
        # If you want an EMA copy, you can do something like:
        # self.model_ema = copy.deepcopy(model)
        # for param in self.model_ema.parameters():
        #     param.requires_grad = False

        self.parametrizations = [
            "velocity",
            "x0-prediction",
            "x1-prediction",
            "score",
        ]
        # f^B_t(x) = a_t * x + b_t * f^A_t(x)
        # X_t = alpha_t * X_1 + sigma_t * X_0
        # X_0 noise, X_1 data

        self.schedules = [
            "linear",  # beta_t = beta_start + (beta_end - beta_start) * t / T
            "cosine",  # beta_t = 1 - (cos(pi * t / T) + 1) / 2
            "sigmoid",  # beta_t = 1 - (sigmoid((t - T/2) / tau) - sigmoid(-T/2) / (sigmoid(T/2) - sigmoid(-T/2))
            "linear-interpolation",  # X_0 = (1 - alpha_t) * X_1 + alpha_t * X_0
        ]

        # Loss function (L2 MSE is typical for DDPM)
        self.criterion = torch.nn.MSELoss()

    def get_beta_schedule(self, schedule_type: str, T: int):
        """
        Return a beta schedule for T timesteps.
        For instance, a linear schedule from beta_start to beta_end.
        """
        if schedule_type == "linear":
            return linear_beta_schedule(T)
        elif schedule_type == "cosine":
            return cosine_beta_schedule(T)
        elif schedule_type == "sigmoid":
            return sigmoid_beta_schedule(T)

    def q_sample_sbm(
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        schedule: str = "linear",
        s: float = 0.008,
        variance_mode: str = "VP",
    ) -> torch.Tensor:
        """
        Sample from the SBM (Stochastic Backward Model), i.e., produce x_t given x_0.
        the formula is  is X_t = alpha_t * X_1 + sigma_t * X_0

        Args:
            x0 (torch.Tensor): The original sample (batch, *shape).
            t (torch.Tensor): Time in [0, 1] (can be broadcastable to x0.shape).
            noise (torch.Tensor, optional): Optional noise tensor of same shape as x0.
                If None, it will be sampled from a standard normal distribution.
            beta_min (float): Minimum beta (used in linear schedule).
            beta_max (float): Maximum beta (used in linear schedule).
            schedule (str): Schedule type. One of ['linear', 'cosine'].
            s (float): Offset for the cosine schedule (Nichol & Dhariwal).

        Returns:
            torch.Tensor: The noised sample x_t.
        """
        # If user doesn't provide noise, sample standard Gaussian
        if noise is None:
            noise = torch.randn_like(x0)

        if schedule == "linear":

            B_t = beta_min * t + 0.5 * (beta_max - beta_min) * (t**2)
            alpha_t = torch.exp(-0.5 * B_t)  # = e^{-1/2 * B(t)}
            if variance_mode == "VP":
                # ----------------------------------------------------
                # Linear schedule: beta(t) = beta_min + (beta_max - beta_min)*t
                #
                # We define: B(t) = \int_0^t beta(s) ds
                #           = beta_min*t + 1/2 * (beta_max - beta_min) * t^2
                # Then the mean coefficient (for a variance-preserving SDE) is:
                #   alpha_t = exp(-0.5 * B(t))
                # and the variance coefficient is:
                #   sigma_t^2 = 1 - exp(-B(t)).
                # So the forward noising is:
                #   x_t = alpha_t * x0 + sigma_t * noise
                # ----------------------------------------------------
                sigma_t = torch.sqrt(1.0 - torch.exp(-B_t))  # = sqrt(1 - e^{-B(t)})

            elif variance_mode == "sub-VP":
                sigma_t = torch.sqrt(1.0 - alpha_t**2)

            elif variance_mode == "VE":
                alpha_t = 1
                sigma_t = beta_min * (beta_max / beta_min) ** t

        elif schedule == "cosine":
            # ----------------------------------------------------
            # Cosine schedule (from "Improved Denoising Diffusion Probabilistic Models")
            #
            # alpha_bar(t) = [ f(t) / f(0) ]^2,   where f(u) = cos( (u + s)/(1 + s) * (pi/2) )
            # Then alpha(t) = sqrt(alpha_bar(t)), sigma(t) = sqrt(1 - alpha_bar(t)).
            # ----------------------------------------------------
            f_t = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5)
            f_0 = torch.cos((s / (1 + s)) * math.pi * 0.5)
            alpha_bar_t = (f_t / f_0) ** 2

            alpha_t = torch.sqrt(alpha_bar_t)
            sigma_t = torch.sqrt(1.0 - alpha_bar_t)

        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. Must be 'linear' or 'cosine'."
            )

        # Combine x0 with noise to get x_t
        return alpha_t * x0 + sigma_t * noise

    def q_sample(self, x0, t, noise=None):
        """
        Diffuse the data (forward process):
          x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod_t[t])[
            :, None, None, None
        ]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod_t[t])[
            :, None, None, None
        ]
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_loss(self, x0, t):
        """
        Compute the denoising objective.
        1) Sample x_t from q(x_t | x0)
        2) Predict the noise \epsilon_\theta(x_t, t)
        3) Compare predicted noise with true noise using MSE
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        # Predict noise using model
        noise_pred = self.model(x_t, t)
        return self.criterion(noise_pred, noise)

    def forward(self, x, t):
        """
        Forward pass for direct usage (rarely used in typical training).
        """
        return self.model(x, t)

    def training_step(self, batch: Any, batch_idx: int):
        """
        Standard training step for DDPM: sample a random t, compute loss.
        """
        x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
        # Sample t uniformly
        t = torch.randint(
            0, self.num_timesteps, (x0.shape[0],), device=x0.device
        ).long()
        loss = self.p_loss(x0, t)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        # Optionally update EMA
        # self.update_ema()
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Compute the same diffusion loss on validation data.
        """
        x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
        t = torch.randint(
            0, self.num_timesteps, (x0.shape[0],), device=x0.device
        ).long()
        loss = self.p_loss(x0, t)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Optionally log or plot samples if self.plot is True.
        """
        if self.plot:
            # Generate some sample images from pure noise
            with torch.no_grad():
                sample = self.sample(batch_size=16, shape=self.dim, device=self.device)
                # plot_samples(sample, ...)
                pass

    def sample(self, batch_size: int, shape: tuple, device: torch.device):
        """
        Algorithm to sample from the reverse diffusion:
          x_T ~ N(0, I)
          For t = T...1:
            x_{t-1} = 1/sqrt(alpha_t) * ( x_t - ( (1 - alpha_t)/sqrt(1 - alpha_cumprod_t) ) * \epsilon_\theta(x_t,t) )
                      + sigma_t * z
        """
        img = torch.randn(batch_size, *shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i], device=device, dtype=torch.long)
            betas_t = self.betas_t[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(
                1.0 - self.alphas_cumprod_t[t]
            )[:, None, None, None]
            sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas_t[t])[
                :, None, None, None
            ]

            # Predict noise
            noise_pred = self.model(img, t.expand(batch_size))

            # x_{t-1}
            img = sqrt_recip_alphas_t * (
                img - betas_t / sqrt_one_minus_alphas_cumprod_t * noise_pred
            )
            if i > 0:
                # sigma_t = sqrt(beta_t)
                sigma_t = torch.sqrt(betas_t)
                noise = torch.randn_like(img)
                img += sigma_t * noise
        return img

    def configure_optimizers(self):
        """
        Standard optimizer + optional scheduler.
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer
        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def update_ema(self):
        """
        If you maintain an EMA copy of your model, update it here.
        """
        if self.model_ema is not None:
            with torch.no_grad():
                ema_params = dict(self.model_ema.named_parameters())
                model_params = dict(self.model.named_parameters())
                for k in ema_params.keys():
                    ema_params[k].data.mul_(self.ema_decay).add_(
                        model_params[k].data, alpha=(1.0 - self.ema_decay)
                    )
