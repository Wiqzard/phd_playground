import torch

ckpt_path = "/home/ss24m050/Documents/phd_playground/logs/train/runs/2025-02-21_22-04-15/checkpoints/epoch_000.ckpt"
ckpt = torch.load(ckpt_path, weights_only=False)  # Explicitly disable weights_only
ckpt.keys()
print(0)
