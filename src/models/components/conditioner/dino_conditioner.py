from torch import nn
import hydra
from omegaconf import DictConfig
from typing import List, Tuple, Dict, Any
from einops import rearrange
import torch
from .dino_features_provider import DINOFeaturesProvider
import numpy as np

from utils.torch_utils import posemb_sincos_1d


class ConditionGenerator(nn.Module):
    conditioning_provider: DINOFeaturesProvider

    def __init__(self, conditioning_provider):
        super().__init__()

        self.conditioning_provider = conditioning_provider
        self.cond_projection = nn.Sequential(
            nn.Linear(
                self.conditioning_provider.dino_channels,
                self.conditioning_provider.proj_channels,
            ),
            nn.LayerNorm(self.conditioning_provider.proj_channels),
        )

        self.cond_masked_tokens = nn.Parameter(
            torch.randn(self.conditioning_provider.num_condition_tokens * 2, 1024),
            requires_grad=True,
        )
        self.cond_pos_emb = posemb_sincos_1d(
            self.conditioning_provider.num_pos_emb_tokens, 1024
        )

    def drop_pixels(
        self, pixel_values: torch.Tensor, patches_xy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_size = self.conditioning_provider.patch_size
        for coord in patches_xy:
            l, y, x = coord
            if l != 0:
                continue

            # Calculate pixel range for the patch in the grid
            x_start = x * patch_size
            x_end = (x + 1) * patch_size
            y_start = y * patch_size
            y_end = (y + 1) * patch_size

            # Set the corresponding region in the tensor to 0 for all batches and layers
            pixel_values[..., y_start:y_end, x_start:x_end] = 0
        return pixel_values

    def forward(
        self,
        pixel_values: torch.Tensor,
        generator: torch.Generator,
        enable_pixels_dropout: bool,
        return_grouped_tokens: bool = False,
        force_cond_feats: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConditionGenerator model that generates condition features for the given pixel values.

        Args:
            pixel_values (torch.Tensor): [ B 3 H W ]
            generator (torch.Generator): RNG generator
            enable_pixels_dropout (bool): if True, will drop patches based on condition tokens
            return_grouped_tokens (bool, optional): If set to true, returns grouped tokens for debugging/visualization. Defaults to False.
            force_roi (RectangleRegion, optional): Forces feature extraction ROI to be the given one. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cond_feats [ B N 1024 ] and pixel_values [ B 3 H W ]
        """

        grouped_tokens = None
        cond_feats = (
            self.conditioning_provider.get_features(pixel_values)
            if force_cond_feats is None
            else force_cond_feats
        )
        # ensure that cond_feats' w and h is divisible by 2
        # NOTE: not needed ATM since patches should be divisible by 2 already
        # cond_feats = cond_feats[:, :, :, :cond_feats.size(3)-(cond_feats.size(3) % 2), :cond_feats.size(4)-(cond_feats.size(4) % 2)]

        b, l, c, h, w = cond_feats.size()

        cond_feats = self.conditioning_provider.group_tokens(cond_feats)

        if return_grouped_tokens:
            grouped_tokens = cond_feats.clone()
        patches_to_pick, patches_xy = self.conditioning_provider.get_patches_to_select(
            cond_feats
        )

        cond_feats = self.cond_projection(cond_feats)

        cond_feats = cond_feats.view(
            b, l, h // 2, w // 2, self.conditioning_provider.proj_channels * 4
        )

        cond_feats = rearrange(cond_feats, "b l h w c -> b (l h w) c")
        cond_feats = cond_feats + self.cond_pos_emb.to(cond_feats.device).unsqueeze(0)

        cond_feats = rearrange(
            cond_feats, "b (l h w) c -> b l c h w", l=l, h=h // 2, w=w // 2
        )
        cond_feats = self.conditioning_provider.pick_patches(
            cond_feats, patches_to_pick
        )

        # flows
        flows, masks = self.conditioning_provider.get_flow_features(pixel_values)
        flow_tokens = self.conditioning_provider.sample_flow_tokens(flows, masks)

        if enable_pixels_dropout:
            for b in range(patches_xy.size(0)):
                sample_patches_xy = patches_xy[b]

                patches_mask = (
                    torch.rand(
                        sample_patches_xy.size(0),
                        device=pixel_values.device,
                        generator=generator,
                    )
                    < self.conditioning_provider.image_token_dropout_prob
                )
                masked_patches_xy = sample_patches_xy[patches_mask]
                if masked_patches_xy.size(0) > 0:
                    pixel_values[b, [0], ...] = self.drop_pixels(
                        pixel_values[b, [0], ...], masked_patches_xy
                    )

        cond_feats = torch.cat([cond_feats, flow_tokens], dim=1)

        if grouped_tokens is None:
            return cond_feats, pixel_values
        else:
            return cond_feats, pixel_values, grouped_tokens
