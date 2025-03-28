import torch
import torch.profiler
from torch.profiler import ProfilerActivity
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose
from torch.cuda.amp import autocast

import hydra
from hydra import initialize, compose
from torch.cuda.amp import autocast


# Your config dictionary
config_dict = {
    "_target_": "insert_memory.MemoryDiT3D",
    "x_shape": [4, 32, 32],
    "cat_conditioning": True,
    "max_tokens": 50,
    "hidden_size": 192,
    "patch_size": 2,
    "variant": "full",
    "pos_emb_type": "rope_3d",  # sinusoidal_1d #learned_1d # rope_3d #learned_3d
    "depth": 12,
    "num_heads": 4,
    "mlp_ratio": 4,
    "use_gradient_checkpointing": False,
    "use_fourier_noise_embedding": False,
    "external_cond_dropout": 0.0,
    "external_cond_dim": 0,
    "learn_sigma": False,
    "memory_layer_indices": [1, 5, 9],
    "memory_cfg": {
        # "model_type": "titans",
        # "chunk_size": 4,
        # "batch_size": 256,
        # "depth_memory": 2,
        # "heads": 1,
        # "dim_head": None,  ##96,
        # "momentum": False,
        # "qkv_receives_diff_views": False,
        # "max_grad_norm": 10,
        # "memory_layer_indices": [1, 5, 9],
        "_target_": "insert_memory.TTTConfig",
        "model_type": "ttt",
        "pre_conv": False,
        "hidden_size": 192,
        "ttt_layer_type": "mlp",
        "num_attention_heads": 4,
        "mini_batch_size": 256,
        "use_gate": False,
        "share_qk": False,
        "use_cache": True,
    },
}

@torch.no_grad()
def profile_forward(cfg, device, dtype):
    """
    Profile the forward pass:
    Run multiple forward passes and measure time spent in each operator/layer.
    """
    print("---- Profiling Forward Pass ----")
    model = hydra.utils.instantiate(cfg, _recursive_=True)
    model = model.to(device=device, dtype=dtype)
    b, t, c, h, w = 16, 50, 8, 32, 32
    dummy_input = torch.randn(b, t, c, h, w, device=device, dtype=dtype)
    noise_levels = torch.randint(0, 256, (b, t), device=device, dtype=dtype)

    with torch.autocast(dtype=dtype, device_type=device.type), torch.inference_mode():
        # Warm-up runs (not profiled)
        for _ in range(2):
            _ = model(dummy_input, noise_levels)

        # Actual profiling
        with torch.profiler.profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            num_iters = 5
            num_runs = 5
            for i in range(num_iters):
                neural_memory_cache = None
                for j in range(num_runs):
                    output, neural_memory_cache, aux_output = model(
                        dummy_input, noise_levels, neural_memory_cache=neural_memory_cache, profiler=prof
                    )

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=200))

    del model
    torch.cuda.empty_cache()


def profile_backward(cfg, device, dtype):
    """
    Profile the backward pass:
    Run multiple forward + backward passes and measure time spent in each operator/layer.
    """
    print("---- Profiling Backward Pass ----")
    model = hydra.utils.instantiate(cfg)
    model = model.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    b, t, c, h, w = 2, 50, 8, 32, 32
    dummy_input = torch.randn(b, t, c, h, w, device=device, dtype=dtype)
    noise_levels = torch.randint(0, 256, (b, t), device=device)

    with torch.autocast(dtype=dtype, device_type=device.type):
        # Warm-up runs (not profiled)
        for _ in range(2):
            output, _, _ = model(dummy_input, noise_levels)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Actual profiling
        with torch.profiler.profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            num_iters = 5
            num_runs = 1
            for i in range(num_iters):
                outputs = []
                neural_memory_cache = None
                for j in range(num_runs):
                    output, neural_memory_cache, aux_output = model(
                        dummy_input, noise_levels, neural_memory_cache=neural_memory_cache, profiler=prof
                    )
                    outputs.append(output)
                outputs = torch.stack(outputs)
                optimizer.zero_grad()

                loss = outputs.sum()
                from torchviz import make_dot
                make_dot(loss, params=dict(model.named_parameters())).render("loss_graph", format="pdf")
                #loss.backward()
                for name, param in model.named_parameters():
                    if 'W1' in name:
                        W1 = param
                        break
                grads = torch.autograd.grad(loss, W1, create_graph=True)
                optimizer.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=200))

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = OmegaConf.create(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # You can select one of the floating types below
    # dtype = torch.float16
    # dtype = torch.float32
    dtype = torch.bfloat16

    profile_forward(cfg, device, dtype)
    profile_backward(cfg, device, dtype)
