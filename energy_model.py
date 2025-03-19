from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.transformer.dit import DiT_models


class EnergyModel(nn.Module):
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        model_type: str = "DiT-B/4",
        attention_mode="math",
        max_frames: int = 32,
    ):
        super(EnergyModel, self).__init__()
        self.dit = DiT_models[model_type](
            max_frames=max_frames,
            in_channels=in_channels,
            input_size=input_size,
            attention_mode=attention_mode,
            out_channels=256,
        )
        self.energy_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dit(x, t, y, return_latents=True)
        x = x.mean(dim=1)  # Global average pooling
        energy = self.energy_head(x).squeeze(-1)
        return energy


if __name__ == "__main__":
    model = EnergyModel(input_size=32, max_frames=16)
    input = torch.randn(2, 4, 16, 32, 32)
    t = torch.randn(
        1,
    )
    y = torch.randint(1, 10, (2,))
    out = model(input, t, y)
    print(0)
