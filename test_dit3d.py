import torch


from einops import rearrange, repeat
from typing import Optional

from insert_memory import MemoryDiT3D
import torch

# Suppose the code for MemoryDiT3D is in memory_dit_3d.py
# from memory_dit_3d import MemoryDiT3D


def quick_test_forward():
    # 1) Create a dummy model
    model = MemoryDiT3D(
        x_shape=torch.Size([3, 64, 64]),  # (channels, height, width)
        max_tokens=4,  # e.g. up to T=4 frames if you like
        external_cond_dim=32,
        hidden_size=128,
        patch_size=8,
        variant="full",
        pos_emb_type="learned_1d",
        depth=4,  # shorter depth to keep it small for a quick test
        num_heads=4,
        mlp_ratio=2.0,
        use_gradient_checkpointing=False,
        use_fourier_noise_embedding=False,
        external_cond_dropout=0.1,
        learn_sigma=False,
        chunk_size=16,
        batch_size=16,
        depth_memory=2,
        memory_layer_indices=[1, 3],  # some small set of memory slots
        momentum=True,
    )

    # 2) Create a dummy input
    # Let's say batch_size=2, T=2 frames, C=3 channels, H=64, W=64
    # So x has shape [B, T, C, H, W]
    x = torch.randn(2, 2, 3, 64, 64)

    # 3) Create noise levels (shape [B, T] or [B], depending on your usage).
    # Here let's just keep T=2 for demonstration:
    noise_levels = torch.randn(2, 2)

    # 4) Optionally, create dummy external conditions
    # Shape can be [B, external_cond_dim], or [B, T, external_cond_dim],
    # depending on your usage. We'll do [B, external_cond_dim].
    external_cond = torch.randn(2, 32)

    # 5) Forward pass
    with torch.no_grad():
        out, memory_cache, aux_output = model(
            x,
            noise_levels,
            external_cond=external_cond,
            neural_memory_cache=None,  # no previous memory states
        )

    # 6) Print some info
    print("Output shape:", out.shape)
    if memory_cache:
        print("Memory cache has length:", len(memory_cache))
        for i, item in enumerate(memory_cache):
            if item is not None:
                print(f"  Memory cache[{i}] shape(s): {[t.shape for t in item]}")


if __name__ == "__main__":
    quick_test_forward()


# def quick_test_dit3d():
#    # Example input parameters
#    B, T = 2, 4   # Batch size, temporal length
#    H, W = 64, 64 # Spatial resolution
#    C = 3         # Number of channels
#    x_shape = (C, H, W)
#    max_tokens = T
#    external_cond_dim = 16
#
#    # Instantiate the model
#    model = DiT3D(
#        x_shape=x_shape,
#        max_tokens=max_tokens,
#        external_cond_dim=external_cond_dim,
#        hidden_size=256,    # e.g., same as noise_level_emb_dim
#        patch_size=8,
#        variant="full",
#        pos_emb_type="learned_1d",
#        depth=4,
#        num_heads=8,
#        mlp_ratio=4.0,
#        use_gradient_checkpointing=False,
#        use_fourier_noise_embedding=False,
#        external_cond_dropout=0.1,
#    )
#
#    # Create dummy inputs
#    # x has shape (B, T, C, H, W)
#    x = torch.randn(B, T, C, H, W)
#    noise_levels = torch.randn(B, T)       # (B, T, noise_level_dim)
#    external_cond = torch.randn(B, T, external_cond_dim)          # (B, T, external_cond_dim)
#
#    # Forward pass
#    with torch.no_grad():
#        output = model(x, noise_levels, external_cond)
#
#    # The shape should be (B, T, C, H, W)
#    print(f"Input shape:  {x.shape}")
#    print(f"Output shape: {output.shape}")
#    print(0)
#
# if __name__ == "__main__":
#    quick_test_dit3d()
#
