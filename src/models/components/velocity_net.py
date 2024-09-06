import os
import sys
from functools import partial, wraps
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torchdiffeq import odeint
from tqdm import tqdm

# from lutils.configuration import Configuration
# from lutils.dict_wrapper import DictWrapper
#

# from model.vqgan.taming.autoencoder import (
#    VQModelInterface,
#    vq_f8_ddconfig,
#    vq_f8_small_ddconfig,
#    vq_f16_ddconfig,
# )


def manage_gpu_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        local_vars = list(locals().keys())
        for var in local_vars:
            if var != "result" and var != "torch":
                del locals()[var]
        torch.cuda.empty_cache()
        return result

    return wrapper


class VideoAutoencoder(nn.Module):
    def __init__(
        self,
        type: str,
        ckpt_path: str,
        hf_token: str = None,
        model_id: str = None,
    ):
        super().__init__()
        self.type = type
        self.ckpt_path = ckpt_path
        self.hf_token = hf_token
        self.model_id = model_id

        if type == "ours":
            raise NotImplementedError("Not implemented yet")
            # self.ae = build_vqvae(config=ae_config, convert_to_sequence=True)
            # self.ae.backbone.load_from_ckpt(ckpt_path)
        elif type == "svd":
            from diffusers import AutoencoderKLTemporalDecoder
            from huggingface_hub import login

            login(token=hf_token)
            self.ae = AutoencoderKLTemporalDecoder.from_pretrained(
                model_id, subfolder="vae", use_safetensors=True
            )
        else:
            raise NotImplementedError("Not implemented yet")
            # if type == "f8":
            #     ae_settings = vq_f8_ddconfig
            # elif type == "f8_small":
            #     ae_settings = vq_f8_small_ddconfig
            # else:
            #     ae_settings = vq_f16_ddconfig
            # self.ae = VQModelInterface(ae_settings, ckpt_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = self._encode_observations(x)
        return latents, self._decode_latents(latents)

    @torch.no_grad()
    def _encode_observations(self, x: torch.Tensor) -> torch.Tensor:
        if self.type == "ours":
            latents = self.ae(x).latents
        else:
            flat_input_frames = rearrange(x, "b n c h w -> (b n) c h w")
            if self.type == "svd":
                posterior = self.ae.encode(flat_input_frames).latent_dist
                flat_latents = posterior.sample()
            else:
                flat_latents = self.ae.encode(flat_input_frames)
            latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=x.size(1))
        return latents

    @manage_gpu_memory
    def _decode_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to decode latents back to image space."""
        b, num_frames = x.shape[:2]
        latents = rearrange(x, "b n c h w -> (b n) c h w")
        if self.type == "ours":
            reconstructed_observations = self.ae.backbone.decode_from_latents(latents)
        elif self.type == "svd":
            reconstructed_observations = self.ae.decode(
                latents, num_frames=num_frames
            ).sample
        else:
            reconstructed_observations = self.ae.decode(latents)

        reconstructed_observations = rearrange(
            reconstructed_observations, "(b n) c h w -> b n c h w", b=b
        )
        return reconstructed_observations


class VelocityNet(nn.Module):
    def __init__(
        self,
        autoencoder: Any,
        flow_network: Any,
        flow_representation_network: Any,
        sparsification_network: Any,
        vector_field_regressor: Any,
        sigma: float,
        skip_prob: float,
        num_ref_frames: int = 5,
        num_cond_frames: int = 5,
        hist_window_size: int = 5,
    ):
        super().__init__()

        self.autoencoder = autoencoder
        self.flow_network = flow_network
        self.flow_representation_network = flow_representation_network
        self.sparsification_network = sparsification_network
        self.vector_field_regressor = vector_field_regressor

        self.sigma = sigma
        self.skip_prob = skip_prob
        self.num_ref_frames = num_ref_frames
        self.num_cond_frames = num_cond_frames
        self.hist_window_size = hist_window_size

        self.autoencoder.eval()

    def load_from_ckpt(
        self, ckpt_path: str, except_keys: List[str] = [], fuzzy_match: bool = False
    ):
        loaded_state = torch.load(ckpt_path, map_location="cpu")
        if except_keys:
            for k in except_keys:
                if k in loaded_state["model"]:
                    del loaded_state["model"][k]
        is_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            state = {
                k.replace("module.", ""): v for k, v in loaded_state["model"].items()
            }
        else:
            state = {f"module.{k}": v for k, v in loaded_state["model"].items()}

        if fuzzy_match:
            model_state_dict = self.state_dict()
            matched_keys = []
            for k, v in state.items():
                for model_key in model_state_dict:
                    if model_key in k:
                        model_state_dict[model_key] = v
                        matched_keys.append(k)
                        break
            state = model_state_dict

        dmodel = (
            self.module
            if isinstance(self, torch.nn.parallel.DistributedDataParallel)
            else self
        )
        dmodel.load_state_dict(state)
        print(f"Loaded weights for keys: {state}")

    def _get_flows(self, X: torch.Tensor, flows: torch.Tensor = None):
        with torch.no_grad():
            if flows is None:
                flows = self.flow_network(X[:, -2:]).squeeze(1)
                # flows = self.flow_network(X[:, -8:]).squeeze()
                # print(flows.shape)
            else:
                flows = flows[:, -1].unsqueeze(1)
        sparse_flows = self.sparsification_network(flows)[0]
        flows = self.flow_representation_network(sparse_flows)
        flows = flows.reshape(*flows.shape[:2], -1).permute(0, 2, 1)
        return flows

    @manage_gpu_memory
    def calculate_sparse_flows(
        self, observations: torch.Tensor, flows: torch.Tensor, num_vectors: int = None
    ):
        """
        :param observations: [b, num_observations, num_channels, height, width]
        :param flows: [b, num_observations - 1, 2, height, width]
        :param num_vectors: number of vectors to select
        :return: [b, num_observations - 1, 3, height, width]
        """

        # Compute flows
        if flows is None:
            flows = self.flow_network(observations)
            flows = flows.reshape(-1, 2, *flows.shape[-2:])

        # Sparsify flows
        sparsification_results = self.sparsification_network(
            flows, num_vectors=num_vectors
        )
        sparse_flows = sparsification_results[0]

        return sparse_flows

    def forward(self, t, input_latents: torch.Tensor, context: torch.Tensor):
        batch_size, n_obs = input_latents.size(0), input_latents.size(1)
        flow_states = context[1]
        index_distances = context[0]
        # index_distances = (
        #    torch.arange(n_obs, device=input_latents.device)
        #    .flip(0)
        #    .unsqueeze(0)
        #    .repeat(batch_size, 1)
        # )
        # Predict vectors using the regressor
        reconstructed_vectors = self.vector_field_regressor(
            sample=input_latents,
            timestep=t.squeeze(),
            encoder_hidden_states=flow_states,
            added_time_ids=index_distances,
            skip_action=torch.rand(1, device=input_latents.device).item()
            < self.skip_prob,
        )
        return reconstructed_vectors

    def get_input_frames(
        self, X: torch.Tensor, training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # last frames
        num_ref_frames = self.num_ref_frames
        # the last frame will be the ground truth
        if training:
            num_ref_frames += 1

        reference_frames = X[:, -num_ref_frames:]

        deterministic = False  # True #False
        if not deterministic:
            # sample within the history window before reference frames
            upper_bound = min(X.shape[1] - num_ref_frames, self.hist_window_size)

            conditioning_frames_idx = (
                torch.sort(torch.randperm(upper_bound)[: self.num_cond_frames])[0]
                .unsqueeze(0)
                .repeat(X.shape[0], 1)
                .flip(1)
            ) + 1
        else:
            conditioning_frames_idx = (
                torch.arange(
                    X.shape[1] - num_ref_frames - self.num_cond_frames,
                    X.shape[1] - num_ref_frames,
                )
                .unsqueeze(0)
                .repeat(X.shape[0], 1)
                + 1
            )

        conditioning_frames = X[:, :-num_ref_frames][
            torch.arange(X.shape[0]).unsqueeze(1), -conditioning_frames_idx
        ]

        time_ids_conditioning = num_ref_frames + conditioning_frames_idx
        # append for the denoising frame if not training
        time_ids_reference = torch.arange(0, num_ref_frames + (not training)).flip(0)

        time_ids = torch.cat(
            [
                time_ids_conditioning,
                time_ids_reference.unsqueeze(0).expand(X.shape[0], -1),
            ],
            dim=1,
        ).to(X.device)

        frames = torch.cat([conditioning_frames, reference_frames], dim=1)

        if training:
            frames = self.autoencoder._encode_observations(frames)

        return frames, time_ids

    @manage_gpu_memory
    def generate_frames(
        self,
        observations: torch.Tensor,
        context: torch.Tensor = None,
        num_frames: int = None,
        steps: int = 100,
        warm_start: float = 0.0,
        skip_past: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generates num_frames frames conditioned on observations.

        :param observations: Tensor of shape [b, 1, num_channels, height, width]
        :param sparse_flows: Tensor of shape [b, n, 3, height, width], None for no actions
        :param num_frames: Number of frames to generate
        :param warm_start: Part of the integration path to jump to
        :param steps: Number of steps for sampling
        :param past_horizon: Number of frames to condition on
        :param skip_past: Whether to consider past or not
        :param verbose: Whether to display loading bar
        :return: Generated frames in tensor format
        """

        # Encode observations to latents
        observations = observations[:, -(self.num_cond_frames + self.num_ref_frames) :]
        with torch.no_grad():
            latents = self.autoencoder._encode_observations(observations)

        b, n, c, h, w = latents.shape
        shape = [b, n, c, h, w]
        sparse_flows = context
        num_actions = sparse_flows.size(1) if sparse_flows is not None else 0

        # Generate future latents
        gen = tqdm(
            range(num_frames),
            desc="Generating frames",
            disable=not verbose,
            leave=False,
        )

        for i in gen:
            print(latents.shape)
            latents = self._generate_next_latent(
                latents,
                sparse_flows,
                num_actions,
                shape,
                warm_start,
                steps,
                i,
            )

        gen.close()

        # Decode to image space
        reconstructed_observations = self.autoencoder._decode_latents(latents)

        return reconstructed_observations

    def _generate_next_latent(
        self,
        latents: torch.Tensor,
        sparse_flow_states: torch.Tensor,
        num_actions: int,
        shape: List[int],
        warm_start: float,
        steps: int,
        i: int,
    ) -> torch.Tensor:
        """Helper function to generate the next latent."""

        def f(t: torch.Tensor, y: torch.Tensor):
            # here are all latents, e.g. 10 gt latents and 1 generated latent
            # sample k conditional latents and cat with the reference latents - 1
            # cat with the initial reference latent y
            # calculate time_ids + flows

            sample_latents, index_distances = self.get_input_frames(
                latents, training=False
            )
            sample_latents = torch.cat([sample_latents, y.unsqueeze(1)], dim=1)

            # Calculate vectors
            vectors = self.vector_field_regressor(
                sample=sample_latents,
                timestep=t * torch.ones(b, device=latents.device),
                encoder_hidden_states=(
                    sparse_flow_states[:, i : i + 1]
                    if i < num_actions
                    else sparse_flow_states[:, -1:]
                ),
                added_time_ids=index_distances,
                skip_action=i >= num_actions,
            )

            return vectors

        b, n, c, h, w = shape
        # Initialize with noise
        noise = torch.randn([b, c, h, w], device=latents.device)
        y0 = (1 - (1 - self.sigma) * warm_start) * noise + warm_start * latents[:, -1]

        # Solve ODE
        next_latents = odeint(
            f,
            y0,
            t=torch.linspace(
                warm_start, 1, int((1 - warm_start) * steps), device=y0.device
            ),
            method="rk4",
        )[-1]

        latents = torch.cat([latents, next_latents.unsqueeze(1)], dim=1)

        return latents
