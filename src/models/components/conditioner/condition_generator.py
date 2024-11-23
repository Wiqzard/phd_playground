from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import nn

from ..utils import posemb_sincos_1d
from .dino_features_provider import DINOFeaturesProvider


class ConditionGenerator(nn.Module):
    def __init__(
        self,
        clip_provider: Optional[Any],
        dino_provider: Optional[Any] = None,
        flow_provider: Optional[Any] = None,
    ):
        super().__init__()
        self.clip_provider = clip_provider
        self.dino_provider = dino_provider
        self.flow_provider = flow_provider

        if clip_provider is None and dino_provider is None and flow_provider is None:
            raise ValueError("Must at least specify one of the providers")

        if flow_provider is not None:
            raise NotImplementedError("Not implemented yet")

        if dino_provider is not None:
            self.cond_projection = nn.Sequential(
                nn.Linear(
                    self.dino_provider.dino_channels,
                    self.dino_provider.proj_channels,
                ),
                nn.LayerNorm(self.dino_provider.proj_channels),
            )

            self.cond_masked_tokens = nn.Parameter(
                torch.randn(self.dino_provider.num_condition_tokens, 1024),
                requires_grad=True,
            )
            self.cond_pos_emb = posemb_sincos_1d(self.dino_provider.num_pos_emb_tokens, 1024)

    def drop_pixels(
        self, pixel_values: torch.Tensor, patches_xy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_size = self.dino_provider.patch_size
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

    def create_random_mask(self, cond_feats: torch.Tensor) -> torch.Tensor:
        """Last step.

        Given the cond_feats, apply token dropout and return the mask.
        Args:
            cond_feats (torch.Tensor): [b T c]
        Returns:
            torch.Tensor: [b T 1]
        """
        return torch.bernoulli(
            torch.full(
                (cond_feats.size(0), cond_feats.size(1), 1),
                1 - self.token_dropout_prob,
                device=cond_feats.device,
            )
        )

    def clip_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip_provider(pixel_values)

    def dino_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        grouped_tokens = None
        cond_feats = (
            self.dino_provider.get_features(pixel_values)
            if force_cond_feats is None
            else force_cond_feats
        )
        # ensure that cond_feats' w and h is divisible by 2
        # NOTE: not needed ATM since patches should be divisible by 2 already
        # cond_feats = cond_feats[:, :, :, :cond_feats.size(3)-(cond_feats.size(3) % 2), :cond_feats.size(4)-(cond_feats.size(4) % 2)]

        b, l, c, h, w = cond_feats.size()
        cond_feats = self.dino_provider.group_tokens(cond_feats)
        if return_grouped_tokens:
            grouped_tokens = cond_feats.clone()
        patches_to_pick, patches_xy = self.dino_provider.get_patches_to_select(cond_feats)

        cond_feats = self.cond_projection(cond_feats)
        cond_feats = cond_feats.view(b, l, h // 2, w // 2, self.dino_provider.proj_channels * 4)
        cond_feats = rearrange(cond_feats, "b l h w c -> b (l h w) c")
        cond_feats = cond_feats + self.cond_pos_emb.to(cond_feats.device).unsqueeze(0)

        cond_feats = rearrange(cond_feats, "b (l h w) c -> b l c h w", l=l, h=h // 2, w=w // 2)
        cond_feats = self.dino_provider.pick_patches(cond_feats, patches_to_pick)

        if enable_pixels_dropout:
            for b in range(patches_xy.size(0)):
                sample_patches_xy = patches_xy[b]

                patches_mask = (
                    torch.rand(
                        sample_patches_xy.size(0),
                        device=pixel_values.device,
                    )
                    < self.dino_provider.image_token_dropout_prob
                )
                masked_patches_xy = sample_patches_xy[patches_mask]
                if masked_patches_xy.size(0) > 0:
                    pixel_values[b, [0], ...] = self.drop_pixels(
                        pixel_values[b, [0], ...], masked_patches_xy
                    )

        if grouped_tokens is None:
            return cond_feats, pixel_values
        else:
            return cond_feats, pixel_values, grouped_tokens

    def flow_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pass

    def mask_features(self, features):
        batch_size = features.size(0)
        cond_mask = self.create_random_mask(features)

        enable_cond = (
            torch.rand(batch_size, device=features.device, generator=self.generator)
            > self.cond_generator.conditioning_provider.no_condition_prob
        )
        for b in range(batch_size):
            enable_condition = enable_cond[b,]
            cond_mask[b,] *= enable_condition

        masked_features = cond_mask * features + (
            1 - cond_mask
        ) * self.dino_provider.cond_masked_tokens.unsqueeze(0)
        return masked_features

    def forward(
        self,
        pixel_values: torch.Tensor,
        enable_pixels_dropout: bool,
        mask_features: Optional[bool] = False,
        return_grouped_tokens: bool = False,
        force_cond_feats: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the ConditionGenerator model that generates condition features for the
        given pixel values.

        Args:
            pixel_values (torch.Tensor): [ B 3 H W ]
            generator (torch.Generator): RNG generator
            enable_pixels_dropout (bool): if True, will drop patches based on condition tokens
            return_grouped_tokens (bool, optional): If set to true, returns grouped tokens for debugging/visualization. Defaults to False.
            force_roi (RectangleRegion, optional): Forces feature extraction ROI to be the given one. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cond_feats [ B N 1024 ] and pixel_values [ B 3 H W ]
        """

        if self.clip_provider is not None:
            clip_features = self.clip_forward(pixel_values)
            return clip_features

        if self.dino_provider is not None:
            dino_features = self.dino_forward(pixel_values)
            if mask_features:
                dino_features = self.mask_features(dino_features)
            return dino_features

        if self.flow_forward is not None:
            pass
