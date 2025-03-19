import torch


from einops import rearrange, repeat
from typing import Optional


class MemoryDiT3D(Bac):
    def __init__(
        self,
        num_patches: Optional[int] = None,
        max_temporal_length: int = 16,
        out_channels: int = 4,
        variant: Variant = "full",
        pos_emb_type: PosEmb = "learned_1d",
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self._check_args(num_patches, variant, pos_emb_type)
        self.learn_sigma = learn_sigma
        self.out_channels = out_channels * (2 if learn_sigma else 1)
        self.num_patches = num_patches
        self.max_temporal_length = max_temporal_length
        self.max_tokens = self.max_temporal_length * (num_patches or 1)
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
        self.temporal_blocks = (
            nn.ModuleList(
                [
                    DiTBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(depth)
                ]
            )
            if self.is_factorized
            else None
        )

        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)

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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DiTBase model (simplified, single-token-set version).
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, OC).
        """
        # If you still need to check input length vs. max_tokens:
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

        # 3) Factorize the sequence (if required)
        if self.is_factorized:
            x, c = rearrange_contiguous_many((x, c), "b (t p) c -> (b t) p c", p=self.num_patches)
            if self.is_pos_emb_absolute_factorized:
                x = self.spatial_pos_emb(x)

        # 4) Transformer Blocks
        for i, (block, temporal_block) in enumerate(
            zip(self.blocks, self.temporal_blocks or [None for _ in range(self.depth)])
        ):
            # Spatial block
            x = self.checkpoint(block, x, c)

            if self.is_factorized:
                # Switch space <-> time
                x, c = rearrange_contiguous_many((x, c), "(b t) p c -> (b p) t c", b=batch_size)
                # Possibly apply temporal pos emb the first time
                if i == 0 and self.pos_emb_type == "sinusoidal_factorized":
                    x = self.temporal_pos_emb(x)
                # Temporal block
                x = self.checkpoint(temporal_block, x, c)
                # Switch back
                x, c = rearrange_contiguous_many((x, c), "(b p) t c -> (b t) p c", b=batch_size)

        # 5) Un-factorize if needed
        if self.is_factorized:
            x, c = rearrange_contiguous_many((x, c), "(b t) p c -> b (t p) c", b=batch_size)

        # 6) Final layer
        x = self.final_layer(x, c)

        return x







    def __init__(self, 
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        hidden_size: int,
        patch_size: int,
        variant: str = "full",
        pos_emb_type: str = "learned_1d",
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_gradient_checkpointing: bool = False,
        use_fourier_noise_embedding: bool = False,
        external_cond_dropout: float = 0.0,
        ):
        super().__init__(
            x_shape=x_shape,
            max_tokens=max_tokens,
            external_cond_dim=external_cond_dim,
            hidden_size=hidden_size,
            patch_size=patch_size,
            variant=variant,
            pos_emb_type=pos_emb_type,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_gradient_checkpointing=use_gradient_checkpointing,  
            use_fourier_noise_embedding=use_fourier_noise_embedding,
            external_cond_dropout=external_cond_dropout,
        )

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
                            qkv_receives_diff_views=False,
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
        
        def custom_dit_forward(self, x, c):
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

            for i, (block, temporal_block) in enumerate(
                zip(self.blocks, self.temporal_blocks or [None for _ in range(self.depth)])
            ):
                # Spatial block
                x = self.checkpoint(block, x, c)

            x = self.final_layer(x, c)

            return x

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
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
        x = self._dit_base_forward(x, emb)  # (B, N, C)

        # Unpatchify
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B*T, H, W, C)

        # Reshape back to (B, T, ...)
        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )  # (B, T, C, H, W)
        return x

        


def quick_test_dit3d():
    # Example input parameters
    B, T = 2, 4   # Batch size, temporal length
    H, W = 64, 64 # Spatial resolution
    C = 3         # Number of channels
    x_shape = (C, H, W)
    max_tokens = T
    external_cond_dim = 16

    # Instantiate the model
    model = DiT3D(
        x_shape=x_shape,
        max_tokens=max_tokens,
        external_cond_dim=external_cond_dim,
        hidden_size=256,    # e.g., same as noise_level_emb_dim
        patch_size=8,
        variant="full",
        pos_emb_type="learned_1d",
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        use_gradient_checkpointing=False,
        use_fourier_noise_embedding=False,
        external_cond_dropout=0.1,
    )

    # Create dummy inputs
    # x has shape (B, T, C, H, W)
    x = torch.randn(B, T, C, H, W)
    noise_levels = torch.randn(B, T)       # (B, T, noise_level_dim)
    external_cond = torch.randn(B, T, external_cond_dim)          # (B, T, external_cond_dim)

    # Forward pass
    with torch.no_grad():
        output = model(x, noise_levels, external_cond)

    # The shape should be (B, T, C, H, W)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(0)

if __name__ == "__main__":
    quick_test_dit3d()
