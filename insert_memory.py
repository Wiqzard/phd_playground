from types import SimpleNamespace

import torch
import torch.nn as nn

from einops import rearrange

from titans.titans_pytorch.neural_memory import NeuralMemory
from titans.titans_pytorch.memory_models import MemoryMLP

# from titans.titans_pytorch.ttt_custom import Block
from titans.titans_pytorch.ttt_custom import TTTConfig, TTTLinear, TTTMLP, Block, TTTCache

# from src.models.components.dit import DiT, modulate
from src.models.components.transformer.dit import DiT, modulate


# from src.models.components.dit3d import DiT3D

# class PassthroughMemory(nn.Identity):
# """A wrapper around Identity to match NeuralMemory's API."""
# def forward(self, x, memory_state):
# return x, memory_state  # Ensures consistency with NeuralMemory's API


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class DummyConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def get(self, key, default=None):
        return self.__dict__.get(key, default)

class MemoryBlock(nn.Module):
    def __init__(self, hidden_size, depth_memory=2):
        super().__init__()
        self.memory_layer = NeuralMemory(
            dim=hidden_size,
            # dim_head=96,
            momentum=False,
            chunk_size=16,
            batch_size=None,
            model=MemoryMLP(dim=hidden_size, depth=depth_memory),
            # model=MemoryMLP(dim=96, depth=depth_memory),
            qkv_receives_diff_views=False,
            use_accelerated_scan=False,
            default_step_transform_max_lr=1e-1,
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        rank = 8
        self.lora_A = nn.Parameter(torch.randn(hidden_size, rank))
        self.lora_B = nn.Parameter(torch.randn(hidden_size, rank))


        # self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, c, state=None):
        
        retrieved, memory_state, surprises = self.memory_layer(x, state=state)
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        x = x + retrieved * gate.unsqueeze(1)
        return x, memory_state


class DiT3D(DiT):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        out_channels=None,
        chunk_size=16,
        batch_size=None,
        depth_memory=2,
        qkv_receives_diff_views=False,
        max_frames=32,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            out_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            class_dropout_prob,
            num_classes,
            learn_sigma,
            max_frames=max_frames,
        )
        self.config = DummyConfig(model_type="DiT3D")


        memory_slots = [1, 3, 5, 9]
        self.memory_layers = torch.nn.ModuleList([])
        for i in range(depth):
            if i in memory_slots:
                memory_layer = torch.nn.ModuleList(
                    [
                        NeuralMemory(
                            dim=hidden_size,
                            # dim_head=96,
                            momentum=False,
                            chunk_size=chunk_size,
                            batch_size=batch_size,
                            model=MemoryMLP(dim=hidden_size, depth=depth_memory),
                            # model=MemoryMLP(dim=96, depth=depth_memory),
                            qkv_receives_diff_views=qkv_receives_diff_views,
                            use_accelerated_scan=False,
                            default_step_transform_max_lr=1e-1,
                        ),
                        nn.Sequential(
                            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
                        ),
                        # nn.LayerNorm(hidden_size),  # intentionally commented out?
                    ]
                )
                self.memory_layers.append(memory_layer)
            else:
                self.memory_layers.append(torch.nn.Identity())

        # Corrected loop: Zero-out modulation layers' parameters for non-Identity layers
        for layer in self.memory_layers:
           if isinstance(layer, torch.nn.ModuleList):
               modulation = layer[1]  # This is your nn.Sequential(SiLU, Linear)
               nn.init.constant_(modulation[-1].weight, 0.0)
               nn.init.constant_(modulation[-1].bias, 0.0)

    def forward(self, x, timestep, cond=None, cache_params=None, use_cache=True, run=0):# memory_states=None, return_memory=False):
        memory_states = default(cache_params, [])
        return_memory = True #exists(cache_params)

        bs, t, c, h, w = x.shape

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=bs)
        num_patches = x.shape[1]
        x = x + self.pos_embed[:, :num_patches]
        t_emb = self.t_embedder(timestep)  # (N, D)

        if cond is not None:
            y = self.y_embedder(cond, self.training)  # (N, D)
            c = t_emb + y  # (N, D)
        else:
            c = t_emb

        # for block, memory_layer in zip(self.blocks, self.memory_modules):

        neural_mem_caches = iter(default(memory_states, []))
        next_neural_mem_caches = []
        for j, (block, memory_layer) in enumerate(zip(self.blocks, self.memory_layers)):
            x = block(x, c)  # (N, T, D)

            #if True: #not isinstance(memory_layer, nn.Identity):
            if not isinstance(memory_layer, nn.Identity):
                memory_layer, modulation = memory_layer
                #memory_layer = memory_layer[0]
                # input memory_state and overwriting the output is questionable
                retrieved, memory_state, surprises = memory_layer(
                    x, state=next(neural_mem_caches, None), return_surprises=True
                )
                # memory_state)
                if torch.isnan(retrieved).any():
                    print("retrieved has nan")
                next_neural_mem_caches.append(memory_state)
                shift, scale, gate = modulation(c).chunk(3, dim=-1)
                x = x + retrieved * gate.unsqueeze(1) #* modulate(retrieved, shift, scale)
                #x = x + retrieved  # gate.unsqueeze(1) * modulate(retrieved, shift, scale)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=num_patches // t)
        )  # (B * T, H, W, C)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=bs)  # (B, T, C, H, W)
        # x = rearrange(x, "(b t) h w c -> b t c h w", b=bs)  # (B, T, C, H, W)
        if return_memory:
            return x, next_neural_mem_caches #, surprises

        return x

    #def prepare_inputs_for_generation(self, input_ids, **kwargs):
    #    # Dummy implementation for compatibility with PEFT
    #    return {"input_ids": input_ids}


from src.models.components.transformer.dit3d_base import DiT3D, BaseBackbone
from src.models.components.transformer.ditv2 import DiTBase

class MemoryDiT3D(BaseBackbone):
    


class DiT3DTTT(DiT):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        out_channels=None,
        chunk_size=16,
        batch_size=None,
        depth_memory=2,
        qkv_receives_diff_views=False,
        max_frames=32,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            out_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            class_dropout_prob,
            num_classes,
            learn_sigma,
            max_frames=max_frames,
        )
        self.memory_layer_indices = [1, 4, 7, 9, 11]
        self.layers = torch.nn.ModuleList([])

        config = TTTConfig(
            pre_conv=False,
            hidden_size=hidden_size,
            ttt_layer_type="mlp",
            num_attention_heads=4,
            mini_batch_size=256,
            # rms_norm_eps=,
            # conv_kernel=,
            # mamba settings
            use_gate=False,
            share_qk=False,
            use_cache=True,  # enabling caching is optional
        )
        self.config = config
        for i in range(depth):
            if i in self.memory_layer_indices:
                memory_layer = Block(config, layer_idx=i)
                self.layers.append(memory_layer)
            else:
                self.layers.append(torch.nn.Identity())

        # Corrected loop: Zero-out modulation layers' parameters for non-Identity layers
        # for layer in self.memory_layers:
        #    if isinstance(layer, torch.nn.ModuleList):
        #        modulation = layer[1]  # This is your nn.Sequential(SiLU, Linear)
        #        nn.init.constant_(modulation[-1].weight, 0.0)
        #        nn.init.constant_(modulation[-1].bias, 0.0)

    def forward(
        self, x, timestep, cond=None, position_ids=None, cache_params=None, use_cache=False, run=1
    ):
        bs, t, c, h, w = x.shape

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=bs)
        num_patches = x.shape[1]

        # -----------------------
        # -----------------------
        # -----------------------
        # -----------------------
        #x = x + self.pos_embed[:, :num_patches]

        t_emb = self.t_embedder(timestep)  # (N, D)

        if cond is not None:
            y = self.y_embedder(cond, self.training)  # (N, D)
            c = t_emb + y  # (N, D)
        else:
            c = t_emb

        # for block, memory_layer in zip(self.blocks, self.memory_modules):
        if cache_params is None and use_cache:
            cache_params = TTTCache(self, bs, device=x.device)
            cache_params.seqlen_offset = run * num_patches

        if position_ids is None:
            position_ids = torch.arange(
                run * num_patches, (run + 1) * num_patches, dtype=torch.long, device=x.device
            ).unsqueeze(0)

        attention_mask = torch.ones_like(position_ids)

        next_neural_mem_caches = []
        # -----------------------
        # -----------------------
        # -----------------------
        # -----------------------
        x = x + (c.unsqueeze(1)) * 0.2
        # -----------------------
        # -----------------------
        # -----------------------
        #print(cache_params.ttt_params_dict["W1_states"][1][:2,:5,0,0])

        for j, (block, memory_layer) in enumerate(zip(self.blocks, self.layers)):
            #x = block(x, c)  # (N, T, D)
            if not isinstance(memory_layer, nn.Identity):
                # memory_layer# , modulation= memory_layer
                # input memory_state and overwriting the output is questionable
                x = memory_layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_params=cache_params,
                )
                if torch.isnan(x).any():
                    print("retrieved has nan")

                # next_neural_mem_caches.append(memory_state)
                # shift, scale, gate = modulation(c).chunk(3, dim=-1)
                # x = x + retrieved #gate.unsqueeze(1) * modulate(retrieved, shift, scale)

        #print(cache_params.ttt_params_dict["W1_states"][1][:2,:5,0,0])
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=num_patches // t)
        )  # (B * T, H, W, C)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=bs)  # (B, T, C, H, W)
        # x = rearrange(x, "(b t) h w c -> b t c h w", b=bs)  # (B, T, C, H, W)

        if use_cache:
            return x, cache_params

        return x


if __name__ == "__main__":
    model = DiT3DTTT(depth=12, hidden_size=768, patch_size=8, num_heads=12, max_frames=16).cuda()
    # model = DiT3D(depth=12, hidden_size=768, patch_size=8, num_heads=12, max_frames=16).cuda()
    # print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    test_input = torch.randn(2, 8, 4, 32, 32).cuda()
    output = model(test_input, torch.tensor([1]).cuda(), use_cache=True)

    print(model)
