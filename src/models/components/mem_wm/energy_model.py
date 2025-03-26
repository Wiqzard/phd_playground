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
        # init to zero
        # self.energy_head[1].weight.data.zero_()
        # self.energy_head[1].bias.data.zero_()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:

        x = self.dit(x, t, y, return_latents=True)
        x = x.mean(dim=1)  # Global average pooling
        energy = self.energy_head(x).squeeze(-1)
        return energy


from typing import List, Tuple


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3D convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        dilation=dilation,
        bias=False,
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 3D convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck3D(nn.Module):
    """
    A 3D version of the standard ResNet bottleneck block.
    expansion=4 means if planes=64, the final conv has 256 output channels.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init__()
        width = int(
            planes * (base_width / 64.0)
        )  # Adjust if you want narrower/wider blocks

        # 1x1x1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = nn.BatchNorm3d(width)

        # 3x3x3
        self.conv2 = conv3x3x3(
            width, width, stride=stride, groups=groups, dilation=dilation
        )
        self.bn2 = nn.BatchNorm3d(width)

        # 1x1x1
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """
    A 3D ResNet-like backbone using Bottleneck blocks.
    By default, we use layers=[3, 4, 6, 3] (similar to ResNet-50),
    but we reduce base_width to get ~10M parameters total.
    """

    def __init__(
        self,
        block,
        layers: List[int],
        in_channels: int = 4,
        base_channels: int = 32,  # smaller than the typical 64
        groups: int = 1,
        width_per_group: int = 32,  # also smaller than typical 64
    ):
        super().__init__()

        self.inplanes = base_channels
        self.groups = groups
        self.base_width = width_per_group

        # First conv layer
        self.conv1 = nn.Conv3d(
            in_channels,
            self.inplanes,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),  # keep T stride=1, spatial stride=2
            padding=(1, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # Optional: a pooling layer for time/spatial
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        # ResNet layers
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

        # For classification/embedding
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global average pooling

        # By default, the embedding size = base_channels*8 * block.expansion
        self.out_channels = base_channels * 8 * block.expansion

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        planes: base number of channels before expansion
        blocks: number of blocks in this layer
        stride: stride for the first block
        """
        downsample = None
        outplanes = planes * block.expansion

        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, outplanes, stride=stride),
                nn.BatchNorm3d(outplanes),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
            )
        )
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, T, H, W)
        Output: A feature of shape (B, out_channels) after pooling
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # shape: (B, out_channels, 1, 1, 1)
        x = torch.flatten(x, 1)  # (B, out_channels)
        return x


class EnergyModel3DResNet(nn.Module):
    """
    An Energy Model with a 3D ResNet backbone (Bottleneck blocks).
    Produces a scalar “energy” output per sample.
    """

    def __init__(
        self,
        in_channels: int = 4,
        layers: Tuple[int] = (3, 4, 6, 3),
        base_channels: int = 32,  # reduce from 64 to keep ~10M params
        width_per_group: int = 32,  # reduce from 64
    ):
        super().__init__()
        # 3D ResNet backbone
        self.backbone = ResNet3D(
            block=Bottleneck3D,
            layers=layers,
            in_channels=in_channels,
            base_channels=base_channels,
            width_per_group=width_per_group,
        )

        # Simple linear head to produce a single scalar
        self.energy_head = nn.Sequential(
            nn.LayerNorm(self.backbone.out_channels),
            nn.Linear(self.backbone.out_channels, 1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor = None, y: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: (B, in_channels=4, T, H, W)
        t, y: additional inputs (not used here, but preserved for consistency).
        """
        features = self.backbone(x)  # (B, out_channels)
        energy = self.energy_head(features)  # (B, 1)
        return energy.squeeze(-1)  # (B,)


if __name__ == "__main__":
    model = EnergyModel(input_size=32, max_frames=16)
    input = torch.randn(2, 4, 16, 32, 32)
    t = torch.randn(
        1,
    )
    y = torch.randint(1, 10, (2,))
    out = model(input, t, y)
    print(0)
