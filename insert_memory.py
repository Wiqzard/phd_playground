from types import SimpleNamespace

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint
from typing import Literal, Optional, Tuple, Callable, Union
from timm.models.vision_transformer import PatchEmbed
from einops import repeat, rearrange

from titans.titans_pytorch.neural_memory import NeuralMemory
from titans.titans_pytorch.memory_models import MemoryMLP

# from titans.titans_pytorch.ttt_custom import Block
from titans.titans_pytorch.ttt_custom import TTTConfig, TTTLinear, TTTMLP, Block, TTTCache

# from src.models.components.dit import DiT, modulate
from src.models.components.transformer.dit import DiT, modulate

from src.models.components.transformer.dit3d_base import DiT3D, BaseBackbone
from src.models.components.transformer.ditv2 import DiTBase, DiTBlock, DITFinalLayer, RotaryEmbedding3D, Variant, PosEmb, SinusoidalPositionalEmbedding
from src.models.components.transformer.ditv2_blocks import AdaLayerNormZero

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


class MemoryDiT3D(BaseBackbone):
    def __init__(
        self,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        hidden_size: int,
        patch_size: int,
        cat_conditioning: bool = False,
        variant: str = "full",
        pos_emb_type: str = "learned_1d",
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_gradient_checkpointing: bool = False,
        use_fourier_noise_embedding: bool = False,
        external_cond_dropout: float = 0.0,
        learn_sigma: bool = False,
        chunk_size: int = 16,
        batch_size: int = 16,
        depth_memory: int = 2,
        memory_layer_indices: list = [1, 3, 5, 9],
        momentum: bool = True,
    ):

        self.hidden_size = hidden_size
        self.patch_size = patch_size

        super().__init__(
            x_shape=x_shape,
            max_tokens=max_tokens,
            external_cond_dim=external_cond_dim,
            noise_level_emb_dim=hidden_size,  # e.g. same as hidden_size
            use_fourier_noise_embedding=use_fourier_noise_embedding,
            external_cond_dropout=external_cond_dropout,
        )

        channels, resolution, *_ = x_shape
        self.cat_conditioning = cat_conditioning
        if cat_conditioning:
            channels  = channels * 2
        assert (
            resolution % self.patch_size == 0
        ), "Resolution must be divisible by patch size."

        self.num_patches = (resolution // self.patch_size) ** 2
        out_channels = self.patch_size**2 * channels

        self.patch_embedder = PatchEmbed(
            img_size=resolution,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size,
            bias=True,
        )


        self._check_args(self.num_patches, variant, pos_emb_type)
        self.learn_sigma = learn_sigma
        self.out_channels = out_channels * (2 if learn_sigma else 1)
        self.max_temporal_length = max_tokens 
        self.max_tokens = self.max_temporal_length * (self.num_patches or 1)
        self.hidden_size = hidden_size
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.variant = variant
        self.pos_emb_type = pos_emb_type
        self.use_gradient_checkpointing = use_gradient_checkpointing

        match self.pos_emb_type:
            case "learned_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                    learnable=True,
                )
            case "sinusoidal_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                )

            case "sinusoidal_3d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(
                        self.max_temporal_length,
                        self.spatial_grid_size,
                        self.spatial_grid_size,
                    ),
                )
            case "sinusoidal_factorized":
                self.spatial_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.spatial_grid_size, self.spatial_grid_size),
                )
                self.temporal_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_temporal_length,),
                )
            case "rope_3d":
                rope = RotaryEmbedding3D(
                    dim=self.hidden_size // num_heads,
                    sizes=(
                        self.max_temporal_length,
                        self.spatial_grid_size,
                        self.spatial_grid_size,
                    ),
                )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=(
                        mlp_ratio if self.variant != "factorized_attention" else None
                    ),
                    rope=rope if self.pos_emb_type == "rope_3d" else None,
                )
                for _ in range(depth)
            ]
        )

        assert (
            max(memory_layer_indices) < depth  
        ), f"Memory slots must be less than the depth of the model ({depth}) but got {memory_slots}"

        self.memory_layer_indices = memory_layer_indices
        self.memory_layers = torch.nn.ModuleList([])
        for i in range(depth):
            if i in memory_layer_indices:
                memory_layer = torch.nn.ModuleList(
                    [
                        NeuralMemory(
                            dim=hidden_size,
                            momentum=momentum,
                            chunk_size=chunk_size,
                            batch_size=batch_size,
                            model=MemoryMLP(dim=hidden_size, depth=depth_memory),
                            qkv_receives_diff_views=False,
                            use_accelerated_scan=False,
                            default_step_transform_max_lr=1e-1,
                        ),
                        AdaLayerNormZero(hidden_size)
                    ]
                )
                self.memory_layers.append(memory_layer)
            else:
                self.memory_layers.append(torch.nn.Identity())

        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)

        self.initialize_weights()

    @property
    def is_factorized(self) -> bool:
        return self.variant in {"factorized_encoder", "factorized_attention"}

    @property
    def is_pos_emb_absolute_once(self) -> bool:
        return self.pos_emb_type in {"learned_1d", "sinusoidal_1d", "sinusoidal_3d"}

    @property
    def is_pos_emb_absolute_factorized(self) -> bool:
        return self.pos_emb_type == "sinusoidal_factorized"

    @property
    def spatial_grid_size(self) -> Optional[int]:
        if self.num_patches is None:
            return None
        grid_size = int(self.num_patches**0.5)
        assert (
            grid_size * grid_size == self.num_patches
        ), "num_patches must be a square number"
        return grid_size

    @staticmethod
    def _check_args(num_patches: Optional[int], variant: Variant, pos_emb_type: PosEmb):
        if variant not in {"full", "factorized_encoder", "factorized_attention"}:
            raise ValueError(f"Unknown variant {variant}")
        if pos_emb_type not in {
            "learned_1d",
            "sinusoidal_1d",
            "sinusoidal_3d",
            "sinusoidal_factorized",
            "rope_3d",
        }:
            raise ValueError(f"Unknown positional embedding type {pos_emb_type}")
        if num_patches is None:
            assert (
                variant == "full"
            ), "For 1D inputs, factorized variants are not supported"
            assert pos_emb_type in {
                "learned_1d",
                "sinusoidal_1d",
            }, "For 1D inputs, only 1D positional embeddings are supported"

        if pos_emb_type == "rope_3d":
            assert variant == "full", "Rope3D is only supported with full variant"

    def checkpoint(self, module: nn.Module, *args):
        if self.use_gradient_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _dit_forward(self, x: torch.Tensor, c: torch.Tensor, cache_params=None) -> torch.Tensor:
        memory_states = default(cache_params, [])
        aux_output = None

        if x.size(1) > self.max_tokens:
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds the maximum length {self.max_tokens}"
            )
        batch_size = x.size(0)
        # 1) Absolute positional embeddings if needed
        if self.is_pos_emb_absolute_once:
            x = self.pos_emb(x)

        # 2) Factorized absolute positional embeddings (non-factorized case)
        if self.is_pos_emb_absolute_factorized and not self.is_factorized:
            # Factor out space, then time, then refold:
            # (b (t p) c) -> (b t) p c -> (b p) t c -> ...
            def add_pos_emb(x: torch.Tensor, batch_size: int) -> torch.Tensor:
                x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                x = self.spatial_pos_emb(x)
                x = rearrange(x, "(b t) p c -> (b p) t c", b=batch_size)
                x = self.temporal_pos_emb(x)
                x = rearrange(x, "(b p) t c -> b (t p) c", b=batch_size)
                return x

            x = add_pos_emb(x, batch_size)


        neural_mem_caches = iter(default(memory_states, []))
        next_neural_mem_caches = []
        for i, (block, memory_layer) in enumerate(
            zip(self.blocks, self.memory_layers or [None for _ in range(self.depth)])
        ):
            x = self.checkpoint(block, x, c)

            if i in self.memory_layer_indices:
                memory_layer, norm = memory_layer
                retrieved, memory_state, aux_output = memory_layer(
                    x, state=next(neural_mem_caches, None), return_surprises=True
                )
                # memory_state)
                if torch.isnan(retrieved).any():
                    print("retrieved has nan")
                next_neural_mem_caches.append(memory_state)
                x, gate_memory = norm(x, c)
                x = x + retrieved * gate_memory

        x = self.final_layer(x, c)

        return x, next_neural_mem_caches, aux_output

    @property
    def in_channels(self) -> int:
        return self.x_shape[0] if not self.cat_conditioning else self.x_shape[0] * 2

    @staticmethod
    def _patch_embedder_init(embedder: PatchEmbed) -> None:
        # Initialize patch_embedder like nn.Linear (instead of nn.Conv2d):
        w = embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(embedder.proj.bias)

    def initialize_weights(self) -> None:
        self._patch_embedder_init(self.patch_embedder)

        # Initialize noise level embedding and external condition embedding MLPs:
        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    @property
    def external_cond_emb_dim(self) -> int:
        # If there's an external conditioning dimension, match hidden_size,
        # otherwise 0
        return self.hidden_size if self.external_cond_dim else 0

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patchified tensor of shape (B, num_patches, patch_size**2 * C)
        Returns:
            unpatchified tensor of shape (B, H, W, C)
        """
        return rearrange(
            x,
            "b (h w) (p q c) -> b (h p) (w q) c",
            h=int(self.num_patches**0.5),
            p=self.patch_size,
            q=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        neural_memory_cache=None,
        start_idx=0,
    ) -> torch.Tensor:
        input_batch_size = x.shape[0]
        # Merge (B, T) into one dimension for patch embedding
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.patch_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        emb = self.noise_level_pos_embedding(noise_levels)

        if external_cond is not None:
            emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
        emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)

        # Pass to DiTBase
        x, neural_memory_cache, aux_output = self._dit_forward(x, emb, neural_memory_cache)  # (B, N, C)

        # Unpatchify
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B*T, H, W, C)

        # Reshape back to (B, T, ...)
        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )  # (B, T, C, H, W)
        return x, neural_memory_cache, aux_output


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
