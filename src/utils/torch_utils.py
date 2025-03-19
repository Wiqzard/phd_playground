from typing import Optional
import torch
from torch.types import _size
import torch.nn as nn


def freeze_model(model: nn.Module) -> None:
    """Freeze the torch model"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze the torch model"""
    model.train()
    for param in model.parameters():
        param.requires_grad = True
