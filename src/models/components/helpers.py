import torch

def to_video(x: torch.Tensor) -> torch.Tensor:
    return (((torch.clamp(x, -1.0, 1.0) + 1.0) / 2.0).detach().cpu() * 255).to(torch.uint8)