from typing import List, Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .utils.embeddings import build_position_encoding

# class FlowRepresentationNetwork(nn.Module):
#    """
#    Model that encodes an optical flow into a state vector
#    """
#
#    def __init__(
#        self,
#        in_channels: int,
#        out_channels: int,
#        pool: bool = False,
#    ):
#        super(FlowRepresentationNetwork, self).__init__()
#
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.pool = pool
#
#        # Configuration for convolutional layers
#        self.conv_configs = [
#            {"out_channels": 32, "kernel_size": 7, "stride": 2, "padding": 3},
#            {"out_channels": 64, "kernel_size": 5, "stride": 2, "padding": 2},
#            {"out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1},
#            {"out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1},
#            {"out_channels": 512, "kernel_size": 3, "stride": 2, "padding": 1},
#        ]
#
#        # Build convolutional layers
#        self.activation = nn.ReLU()
#        self.conv_layers = self._build_conv_layers()
#
#        # Global Average Pooling
#        if self.pool:
#            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#            # Final fully connected layer
#        self.fc = nn.Linear(self.conv_configs[-1]["out_channels"], out_channels)
#
#        # Activation function
#
#    def _build_conv_layers(self) -> nn.ModuleList:
#        layers: List[nn.Module] = []
#        in_channels = self.in_channels
#
#        for config in self.conv_configs:
#            conv = nn.Conv2d(in_channels, **config)
#            bn = nn.BatchNorm2d(config["out_channels"])
#            layers.extend([conv, bn, self.activation])
#            in_channels = config["out_channels"]
#
#        return nn.ModuleList(layers)
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#
#        if x.dim() == 5:
#            x = x.reshape(-1, *x.shape[2:])
#
#        # Apply convolutional layers
#        for layer in self.conv_layers:
#            x = layer(x)
#
#        if self.pool:
#            x = self.global_avg_pool(x)
#            # Flatten and apply final fully connected layer
#            x = x.view(x.size(0), -1)
#
#        x = self.fc(x)
#
#        return x
#
#    def __repr__(self) -> str:
#        return (
#            f"FlowRepresentationNetwork("
#            f"in_channels={self.in_channels}, "
#            f"out_channels={self.out_channels}, "
#            f"in_res={self.in_res})"
#        )


class FlowRepresentationNetwork(nn.Module):
    """Model that encodes an optical flow into a state."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        tile_size: Tuple[int, int],
        out_res: Tuple[int, int],
        depth: int = 4,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.tiling = nn.Sequential(
            nn.Conv2d(
                in_channels,
                256,
                kernel_size=self.tile_size,
                stride=self.tile_size,
                padding=(0, 0),
            ),
            nn.GELU(),
        )
        self.encoding_layers = nn.ModuleList()
        for _ in range(depth):
            self.encoding_layers.append(
                nn.Sequential(
                    Rearrange("b c h w -> b c (h w)"),
                    nn.BatchNorm1d(256),
                    Rearrange("b c (h w) -> b c h w", h=out_res[0]),
                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.GELU(),
                )
            )
        self.out = nn.Sequential(
            Rearrange("b c h w -> b c (h w)"),
            nn.BatchNorm1d(256),
            Rearrange("b c (h w) -> b c h w", h=out_res[0]),
            nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.pos = build_position_encoding(out_channels, position_embedding_name="learned")

    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        """Computes the state corresponding to each observation :param flows: (bs, 3, height,
        width) tensor :return: (bs, state_features, state_height, state_width) tensor of states."""
        # Tile flows
        x = self.tiling(flows)
        # Forward residual layers
        for layer in self.encoding_layers:
            x = x + layer(x)
        # Project to out_channels
        x = self.out(x)
        # Add position encodings
        pos = self.pos(x)
        x = x + pos
        return x
