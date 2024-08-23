import itertools
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.components.yoda.utils import (
    CrossAttnDownBlockSpatioTemporal,
    CrossAttnUpBlockSpatioTemporal,
    DownBlockSpatioTemporal,
    SpatioTemporalResBlock,
    TemporalDownSample,
    TemporalResnetBlock,
    UNetMidBlockSpatioTemporal,
    UpBlockSpatioTemporal,
)
from src.models.components.yoda.utils.all import TimestepEmbedding, Timesteps


def get_down_block(
    down_block_type: str,
    num_layers: int,
    transformer_layers_per_block: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    cross_attention_dim: int,
    num_attention_heads: int,
    resnet_act_fn: str,
    num_in_frames: int = -1,
    num_out_frames: int = -1,
):
    if down_block_type == "DownBlockSpatioTemporal":
        # added for SDV
        return DownBlockSpatioTemporal(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    elif down_block_type == "CrossAttnDownBlockSpatioTemporal":
        # added for SDV
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlockSpatioTemporal"
            )
        return CrossAttnDownBlockSpatioTemporal(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            add_downsample=add_downsample,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            # num_in_frames=num_in_frames,
            # num_out_frames=num_out_frames,
        )

    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    num_attention_heads: int,
    resolution_idx: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = True,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    temporal_num_attention_heads: int = 8,
    temporal_cross_attention_dim: Optional[int] = None,
    temporal_max_seq_length: int = 32,
    transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    dropout: float = 0.0,
    in_frames: int = -1,
    out_frames: int = -1,
):
    if up_block_type == "UpBlockSpatioTemporal":
        # added for SDV
        return UpBlockSpatioTemporal(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            add_upsample=add_upsample,
            in_frames=in_frames,
            out_frames=out_frames,
        )
    elif up_block_type == "CrossAttnUpBlockSpatioTemporal":
        # added for SDV
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlockSpatioTemporal"
            )
        return CrossAttnUpBlockSpatioTemporal(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            add_upsample=add_upsample,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resolution_idx=resolution_idx,
            in_frames=in_frames,
            out_frames=out_frames,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class VectorFieldRegressor(nn.Module):
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (32, 64, 128, 128),
        addition_time_embed_dim: int = 128,  # 256,
        projection_class_embeddings_input_dim: int = 5 * 128,  # 64,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 128,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (2, 4, 8, 8),
        skip_action: bool = False,
        num_frames_in_block: Tuple[int] = [10, 8, 6, 6, 4, 4, 2, 1],
    ):
        super().__init__()

        self.sample_size = sample_size
        self.num_frames_in_block = num_frames_in_block

        # if len(down_block_types) != len(self.num_frames_in_block):
        #    raise ValueError(
        #        f"Must provide the same number of `down_block_types` as `temporal_downsampling`. `down_block_types`: {down_block_types}. `temporal_downsampling`: {self.temporal_downsampling}."
        #    )
        self.cross_attention_dim = cross_attention_dim

        # Check Inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.skip_action = skip_action
        if skip_action:
            self.dummy_action = nn.Parameter(torch.zeros(1, 1, 1, cross_attention_dim))

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(
            projection_class_embeddings_input_dim, time_embed_dim
        )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        self.temporal_downsamples = nn.ModuleList([])
        for i in range(len(self.num_frames_in_block) - 1):
            if i > len(block_out_channels) - 1:
                channels = block_out_channels[len(block_out_channels) - 1 - i]
            else:
                channels = block_out_channels[i]
            if self.num_frames_in_block[i] != self.num_frames_in_block[i + 1]:
                self.temporal_downsamples.append(
                    TemporalDownSample(
                        in_channels=channels,
                        out_channels=channels,
                        temb_channels=time_embed_dim,
                        num_in_frames=self.num_frames_in_block[i],
                        num_out_frames=self.num_frames_in_block[i + 1],
                        downsample_context=True,
                        downsample_residuals=True,
                    )
                )
            else:
                self.temporal_downsamples.append(nn.Identity())
            # in_frames of the residuals for the upsample blocks
            # out_frames num_layers times + 1 except for last layers in each block
            # then group them by number of layers from behind

        blocks_time_embed_dim = time_embed_dim
        # down
        res_out_frames = [[self.num_frames_in_block[0]]]
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.down_blocks.append(down_block)

            res_out_frames.append(
                (layers_per_block[i] + (not is_final_block)) * [self.num_frames_in_block[i]]
            )

        res_out_frames = list(itertools.chain(*res_out_frames))

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            in_frames = list(reversed(res_out_frames[-reversed_layers_per_block[i] - 1 :]))
            out_frames = self.num_frames_in_block[i + len(down_block_types)]
            res_out_frames = res_out_frames[: -reversed_layers_per_block[i] - 1]
            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu",
                in_frames=in_frames,
                out_frames=out_frames,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=32, eps=1e-5
        )
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        skip_action: bool = False,
    ):
        """
        sample: [batch, frames, channels, height, width]
            (k past frames randomly sampled, n-k last frames)
            frames must be self.num_frames_in_block[0]

        timestep: [batch]
        encoder_hidden_states:
            [batch, num_frames, n_tokens, channels] -> keep as is
            [batch, n_tokens, channels] -> repeat for num_frames
            [batch, num_frames, n_tokens, channels, height, width] -> pos_enc and flatten

        added_time_ids: [batch, frames]


        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            if isinstance(timestep, float):
                dtype = torch.float64
            else:
                dtype = torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        sample = sample.flatten(0, 1)

        # Repeat the embeddings num_video_frames times
        emb = emb.repeat_interleave(num_frames, dim=0)
        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )
        if skip_action:
            encoder_hidden_states = self.dummy_action.repeat(batch_size, num_frames, 1, 1)
        encoder_hidden_states = encoder_hidden_states.reshape(-1, *encoder_hidden_states.shape[2:])

        # 2. pre-process
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            image_only_indicator = image_only_indicator[:, : self.num_frames_in_block[i]]

            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )

            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            if not isinstance(self.temporal_downsamples[i], nn.Identity):
                sample, res_samples_, emb, encoder_hidden_states = self.temporal_downsamples[i](
                    sample, None, emb, encoder_hidden_states
                )

            down_block_res_samples += res_samples
            # 4. mid

        image_only_indicator = image_only_indicator[:, : self.num_frames_in_block[i + 1]]
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            image_only_indicator = image_only_indicator[
                :, : self.num_frames_in_block[i + len(self.down_blocks)]
            ]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

            if i + len(self.down_blocks) < len(self.temporal_downsamples):
                if not isinstance(
                    self.temporal_downsamples[i + len(self.down_blocks)], nn.Identity
                ):
                    sample, _, emb, encoder_hidden_states = self.temporal_downsamples[
                        i + len(self.down_blocks)
                    ](sample, None, emb, encoder_hidden_states)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        sample = sample.reshape(batch_size, self.num_frames_in_block[-1], *sample.shape[1:])

        return sample


if __name__ == "__main__":
    device = "cpu"
    bs, t, c, h, w = 2, 15, 4, 32, 32
    model = VectorFieldRegressor(
        sample_size=[32, 32],
        in_channels=4,
        out_channels=4,
        down_block_types=(
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            # "DownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types=(
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            # "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels=(64, 128, 128, 256),
        addition_time_embed_dim=128,  # 256,
        projection_class_embeddings_input_dim=t * 128,  # 64,
        layers_per_block=3,
        cross_attention_dim=128,
        transformer_layers_per_block=1,
        num_attention_heads=(2, 4, 4, 8),
        num_frames_in_block=[15, 10, 8, 5, 4, 4, 2, 1],
        skip_action=True,
        # num_frames_in_block=[10, 8, 5, 5, 4, 4, 2, 1]
    ).to(device)
    model.train()
    # num parameters of model
    print(f"{sum(p.numel() for p in model.parameters()):,}")
    with torch.no_grad():
        input = torch.randn(bs, t, c, h, w).to(device)
        time_step = 100
        encoder_hidden_states = torch.randn(bs, t, 3, 128).to(device)
        additional_time_ids = torch.randn(bs, t).to(device)

        out = model(
            input,
            timestep=time_step,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=additional_time_ids,
            skip_action=False,  # True,  # torch.rand(1, device=device).item() > 0.5,
        )
    print(out)
