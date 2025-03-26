from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from abc import abstractmethod, ABC
from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from src.models.components.embedding.embeddings import (
    StochasticTimeEmbedding,
    RandomDropoutCondEmbedding,
)
from src.models.components.transformer.ditv2 import DiTBase

# ---- BaseBackbone without config ----


class BaseBackbone(ABC, nn.Module):
    def __init__(
        self,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        noise_level_emb_dim: int,
        use_fourier_noise_embedding: bool = False,
        external_cond_dropout: float = 0.0,
        use_causal_mask: bool = True,
    ):
        super().__init__()

        self.external_cond_dim = external_cond_dim
        self.use_causal_mask = use_causal_mask
        self.x_shape = x_shape
        self.max_tokens = max_tokens

        # Store the embedding dimension for noise levels
        self._noise_level_emb_dim = noise_level_emb_dim

        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            dim=self.noise_level_dim,
            time_embed_dim=self.noise_level_emb_dim,
            use_fourier=use_fourier_noise_embedding,
        )

        self.external_cond_embedding = (
            RandomDropoutCondEmbedding(
                cond_dim=self.external_cond_dim,
                cond_emb_dim=self.external_cond_emb_dim,
                dropout_prob=external_cond_dropout,
            )
            if self.external_cond_dim
            else None
        )

    @property
    def noise_level_dim(self) -> int:
        # Example way to scale
        return max(self.noise_level_emb_dim // 4, 32)

    @property
    def noise_level_emb_dim(self) -> int:
        return self._noise_level_emb_dim

    @property
    @abstractmethod
    def external_cond_emb_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# ---- DiT3D without config ----
class DiT3D(BaseBackbone):
    def __init__(
        self,
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

        self.dit_base = DiTBase(
            num_patches=self.num_patches,
            max_temporal_length=max_tokens,
            out_channels=out_channels,
            variant=variant,
            pos_emb_type=pos_emb_type,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            learn_sigma=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.initialize_weights()

    @property
    def in_channels(self) -> int:
        return self.x_shape[0]

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
        x = self.dit_base(x, emb)  # (B, N, C)

        # Unpatchify
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B*T, H, W, C)

        # Reshape back to (B, T, ...)
        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )  # (B, T, C, H, W)
        return x
