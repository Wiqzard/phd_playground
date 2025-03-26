import torch
from einops import rearrange

from src.models.components.dit import DiT


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
        learn_sigma=True,
        out_channels=None,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            class_dropout_prob,
            num_classes,
            learn_sigma,
            out_channels,
        )

    def forward(self, x, timestep, cond=None):
        bs, t, c, h, w = x.shape

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N*T , S, D), where S = H * W / patch_size ** 2
        x = rearrange(x, "(b t) p c -> b (t p) c", b=bs)  #
        t_emb = self.t_embedder(timestep)  # (N, D)

        if cond is not None:
            y = self.y_embedder(cond, self.training)  # (N, D)
            c = t_emb + y  # (N, D)
        else:
            c = t_emb

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B * T, H, W, C)
        x = rearrange(x, "(b t) h w c -> b t c h w", b=bs)  # (B, T, C, H, W)

        return x
