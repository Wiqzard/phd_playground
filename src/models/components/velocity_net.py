import sys
import os
from typing import Any, List
from functools import partial, wraps

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from torchdiffeq import odeint

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
        num_frames: int = 1,
    ):
        super(VideoAutoencoder, self).__init__()
        self.type = type
        self.ckpt_path = ckpt_path
        self.hf_token = hf_token
        self.model_id = model_id
        self.num_frames = num_frames

        if type == "ours":
            raise NotImplementedError("Not implemented yet")
            self.ae = build_vqvae(config=ae_config, convert_to_sequence=True)
            self.ae.backbone.load_from_ckpt(ckpt_path)
        elif type == "svd":
            from diffusers import AutoencoderKLTemporalDecoder
            from huggingface_hub import login

            login(token=hf_token)
            self.ae = AutoencoderKLTemporalDecoder.from_pretrained(
                model_id, subfolder="vae", use_safetensors=True
            )
        else:
            raise NotImplementedError("Not implemented yet")
            if type == "f8":
                ae_settings = vq_f8_ddconfig
            elif type == "f8_small":
                ae_settings = vq_f8_small_ddconfig
            else:
                ae_settings = vq_f16_ddconfig
            self.ae = VQModelInterface(ae_settings, ckpt_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = self._encode_observations(x)
        return latents, self._decode_latents(latents)

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

    def _decode_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to decode latents back to image space."""
        b, num_frames = x.shape[:1]
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
    ):
        super(VelocityNet, self).__init__()

        self.autoencoder = autoencoder
        self.flow_network = flow_network
        self.flow_representation_network = flow_representation_network
        self.sparsification_network = sparsification_network
        self.vector_field_regressor = vector_field_regressor

        self.sigma = sigma
        self.skip_prob = skip_prob

        # self.config = config
        # self.sigma = config["sigma"]
        # self.ae.eval()
        # if False:
        # ckpt_path = "runs/custom_run-europe_videos/checkpoints/final_step_300000.pth"
        # self.load_from_ckpt(ckpt_path, except_keys=["ae"], fuzzy_match=True)

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
            else:
                flows = flows[:, -1].unsqueeze(1)
        sparse_flows = self.sparsification_network(flows)[0]
        flows = self.flow_representation_network(sparse_flows).unsqueeze(1)
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

        # Sparsify flows
        sparsification_results = self.sparsification_network(
            flows, num_vectors=num_vectors
        )
        sparse_flows = sparsification_results.sparse_output

        return sparse_flows

    def forward(self, t, input_latents: torch.Tensor, flow_states: torch.Tensor):
        batch_size, n_obs = input_latents.size(0), input_latents.size(1)
        index_distances = (
            torch.arange(n_obs, device=input_latents.device)
            .flip(0)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
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

    @manage_gpu_memory
    def generate_frames(
        self,
        observations: torch.Tensor,
        sparse_flows: torch.Tensor = None,
        num_frames: int = None,
        steps: int = 100,
        warm_start: float = 0.0,
        past_horizon: int = -1,
        skip_past: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Generates num_frames frames conditioned on observations.

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
        self.ae.eval()
        with torch.no_grad():
            latents = self._encode_observations(observations)

        b, n, c, h, w = latents.shape
        shape = [b, n, c, h, w]

        # Encode sparse flows
        if sparse_flows is not None:
            sparse_flow_states = self.flow_representation_network(
                sparse_flows
            ).unsqueeze(1)
            num_actions = sparse_flow_states.size(1)
        else:
            sparse_flow_states = torch.zeros(
                [b, 1, self.vector_field_regressor.action_state_size, 16, 16],
                device=device,
            )
            num_actions = 0

        # Generate future latents
        gen = tqdm(
            range(num_frames),
            desc="Generating frames",
            disable=not verbose,
            leave=False,
        )

        for i in gen:
            latents = self._generate_next_latent(
                latents,
                sparse_flow_states,
                num_actions,
                shape,
                warm_start,
                steps,
                i,
            )

        gen.close()

        # Decode to image space
        reconstructed_observations = self._decode_latents(latents, b)

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
            sample_latents = latents[:, -n + 1 :]
            sample_latents = torch.cat([sample_latents, y.unsqueeze(1)], dim=1)
            index_distances = (
                torch.arange(0, n).flip(0).unsqueeze(0).repeat(b, 1).to(latents.device)
            )

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
