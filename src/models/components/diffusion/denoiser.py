from typing import Dict, Union

import torch
import torch.nn as nn

from src.utils.torch_utils import append_dims

from .denoiser_scaling import DenoiserScaling


class Denoiser(nn.Module):
    def __init__(self, scaling: DenoiserScaling, num_frames: int = 25):
        super().__init__()
        self.scaling = scaling
        self.num_frames = num_frames

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        noised_input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        cond_mask: torch.Tensor,
    ):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, noised_input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return (
            network(noised_input * c_in, c_noise, cond, cond_mask, self.num_frames) * c_out
            + noised_input * c_skip
        )
