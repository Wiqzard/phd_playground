from typing import Optional, Callable, Literal, Dict, Any
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce

from src.models.components.diffusion.noise_schedule import make_beta_schedule


def extract(a, t, x_shape):
    shape = t.shape
    out = a[t]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))


ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)


class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        timesteps: int = 1000,
        sampling_timesteps: int = 50,
        beta_schedule: str = "cosine",
        schedule_fn_kwargs: Dict = {},
        objective: str = "v_pred",
        loss_weighting: Dict = {},
        ddim_sampling_eta: float = 0.0,
        clip_noise: float = 20.0,
        use_causal_mask: bool = False,
        reconstruction_guidance: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = model
        self.x_shape = x_shape
        self.max_tokens = max_tokens
        self.external_cond_dim = external_cond_dim
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.beta_schedule = beta_schedule
        self.schedule_fn_kwargs = schedule_fn_kwargs
        self.objective = objective
        self.loss_weighting = loss_weighting
        self.ddim_sampling_eta = ddim_sampling_eta
        self.clip_noise = clip_noise
        self.use_causal_mask = use_causal_mask
        self.reconstruction_guidance = reconstruction_guidance
        self.is_discrete = True
        if loss_weighting == {}:
            self.loss_weighting = {
                "strategy": "fused_min_snr",
                "snr_clip": 5,
                "cum_snr_decay": 0.9,
            }
        if schedule_fn_kwargs == {}:
            self.schedule_fn_kwargs = {
                "shift": 1.0,
            }
        self._build_buffer()

    def _build_buffer(self):
        betas = make_beta_schedule(
            schedule=self.beta_schedule,
            timesteps=self.timesteps,
            zero_terminal_snr=self.objective != "pred_noise",
            **self.schedule_fn_kwargs,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32), persistent=False
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        # if (
        #     self.objective == "pred_noise"
        #     or self.reconstruction_guidance is not None
        # ):
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer("snr", snr)
        if self.loss_weighting.strategy in {"min_snr", "fused_min_snr"}:
            clipped_snr = snr.clone()
            clipped_snr.clamp_(max=self.loss_weighting.snr_clip)
            register_buffer("clipped_snr", clipped_snr)
        elif self.loss_weighting.strategy == "sigmoid":
            register_buffer("logsnr", torch.log(snr))

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(
        self,
        x,
        k,
        external_cond=None,
        external_cond_mask=None,
        uncond_cond=None,
        uncond_cond_mask=None,
        cfg_scale=1.0,
        **kwargs
    ):
        """
        x:             The current noisy sample (batch, channels, height, width).
        k:             The diffusion timestep (or index).
        external_cond: The conditional embedding for your model (e.g. text embedding).
        external_cond_mask: Any mask needed for external_cond (optional).
        uncond_cond:   The 'null' or unconditional embedding.
        uncond_cond_mask: Any mask needed for uncond_cond (optional).
        cfg_scale:     The guidance scale. cfg_scale=1 means no guidance
                    (only a single forward pass), and >1 amplifies
                    the difference between conditional and unconditional outputs.
        """

        # If no CFG requested (scale=1 or no uncond input), just do a single pass
        if cfg_scale == 1.0 or uncond_cond is None:
            model_output, *other_output = self.model(
                x, k, external_cond, external_cond_mask, **kwargs
            )
        else:
            model_output_cond, *other_output_cond = self.model(
                x, k, external_cond, external_cond_mask, **kwargs
            )

            kwargs.pop("neural_memory_cache", None)
            model_output_uncond, *other_output_uncond = self.model(
                x, k, uncond_cond, uncond_cond_mask, **kwargs
            )
            model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
            other_output = other_output_cond  # (other_output_uncond, other_output_cond)

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)
        return model_pred, other_output

    def predict_start_from_noise(self, x_k, k, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    def predict_noise_from_start(self, x_k, k, x0):
        # return (
        #     extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        # ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (x_k - extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0) / extract(
            self.sqrt_one_minus_alphas_cumprod, k, x_k.shape
        )

    def predict_v(self, x_start, k, noise):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    def predict_noise_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
        )

    def q_mean_variance(self, x_start, k):
        mean = extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, k, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, k, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_k, k):
        posterior_mean = (
            extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        posterior_variance = extract(self.posterior_variance, k, x_k.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, k, x_k.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, k, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )

    def p_mean_variance(self, x, k, external_cond=None, external_cond_mask=None):
        model_pred = self.model_predictions(
            x=x, k=k, external_cond=external_cond, external_cond_mask=external_cond_mask
        )
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_k=x, k=k)

    def compute_loss_weights(
        self,
        k: torch.Tensor,
        strategy: Literal["min_snr", "fused_min_snr", "uniform", "sigmoid"],
    ) -> torch.Tensor:
        if strategy == "uniform":
            return torch.ones_like(k)
        snr = self.snr[k]
        epsilon_weighting = None
        match strategy:
            case "sigmoid":
                logsnr = self.logsnr[k]
                # sigmoid reweighting proposed by https://arxiv.org/abs/2303.00848
                # and adopted by https://arxiv.org/abs/2410.19324
                epsilon_weighting = torch.sigmoid(
                    self.loss_weighting.sigmoid_bias - logsnr
                )
            case "min_snr":
                # min-SNR reweighting proposed by https://arxiv.org/abs/2303.09556
                clipped_snr = self.clipped_snr[k]
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case "fused_min_snr":
                # fused min-SNR reweighting proposed by Diffusion Forcing v1
                # with an additional support for bi-directional Fused min-SNR for non-causal models
                snr_clip, cum_snr_decay = (
                    self.loss_weighting.snr_clip,
                    self.loss_weighting.cum_snr_decay,
                )
                clipped_snr = self.clipped_snr[k]
                normalized_clipped_snr = clipped_snr / snr_clip
                normalized_snr = snr / snr_clip

                def compute_cum_snr(reverse: bool = False):
                    new_normalized_clipped_snr = (
                        normalized_clipped_snr.flip(1)
                        if reverse
                        else normalized_clipped_snr
                    )
                    cum_snr = torch.zeros_like(new_normalized_clipped_snr)
                    for t in range(0, k.shape[1]):
                        if t == 0:
                            cum_snr[:, t] = new_normalized_clipped_snr[:, t]
                        else:
                            cum_snr[:, t] = (
                                cum_snr_decay * cum_snr[:, t - 1]
                                + (1 - cum_snr_decay) * new_normalized_clipped_snr[:, t]
                            )
                    cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
                    return cum_snr.flip(1) if reverse else cum_snr

                if self.use_causal_mask:
                    cum_snr = compute_cum_snr()
                else:
                    # bi-directional cum_snr when not using causal mask
                    cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
                    cum_snr *= 0.5
                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (
                    1 - normalized_clipped_snr
                )
                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
                clipped_snr = clipped_fused_snr * snr_clip
                snr = fused_snr * snr_clip
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case _:
                raise ValueError(f"unknown loss weighting strategy {strategy}")

        match self.objective:
            case "pred_noise":
                return epsilon_weighting
            case "pred_x0":
                return epsilon_weighting * snr
            case "pred_v":
                return epsilon_weighting * snr / (snr + 1)
            case _:
                raise ValueError(f"unknown objective {self.objective}")

    def diffusion_loss_for_noise_level(
        self,
        x: torch.Tensor,
        noise_level: torch.Tensor,  # shape [B, T], or broadcastable to [B, T]
        conditions: Optional[torch.Tensor] = None,
        # guidance_fn: Optional[Callable] = None,
        context_frame_mask: Optional[torch.Tensor] = None,
        sliding_context_len: Optional[int] = None,
        return_per_token_loss: bool = False,
        **kwargs,
    ):
        """
        Compute a single-step diffusion loss at the given `noise_level`,
        by sliding over the entire sequence in windows of up to `self.max_tokens`.

        Args:
            x: shape [B, T, ...], the ground-truth sequence of length T.
            noise_level: shape [B, T] (or broadcastable).
                        The fixed noise-level for each token.
            conditions: optional, shape [B, T, ...], e.g. external conditions.
            guidance_fn: optional, if you do classifier-free or other guidance.
            context_frame_mask: shape [B, T] (0=to-be-predicted, 1=ground-truth context).
            sliding_context_len: how many tokens from the "past" to include as context
                                for each chunk. Defaults to `self.max_tokens - 1` or
                                requires manual setting if T > self.max_tokens.
            return_per_token_loss: if True, return the full loss tensor [B,T,...].
            ...

        Returns:
            total_loss: scalar if return_per_token_loss=False, else the full [B,T,...].
        """
        device = x.device
        B, T, *x_shape = x.shape

        # -----------------------------------------------------
        # Handle the length & sliding context
        # -----------------------------------------------------
        if sliding_context_len is None:
            if T > self.max_tokens:
                raise ValueError(
                    "When T > max_tokens, you must specify sliding_context_len."
                )
            else:
                sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        # -----------------------------------------------------
        # Create the noised input at this single noise_level
        # -----------------------------------------------------
        # You presumably have a q_sample(...) that does: x_noised = sqrt(α)*x + sqrt(1-α)*noise, etc.
        # Or you can replicate your approach in forward_1(...)
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_noised = self.q_sample(x_start=x, k=noise_level, noise=noise)  # shape [B, T]
        n_context_frames = (1 - context_frame_mask.int()).sum(-1)[0].item()
        context = x[:, :n_context_frames].clone()
        context = torch.cat(
            [
                context,
                torch.zeros(B, T - n_context_frames, *x_shape).to(device, x.dtype),
            ],
            dim=1,
        )

        out_pred = torch.zeros_like(x)
        loss_accumulator = torch.zeros_like(x_noised)  # [B, T, ...] same shape as x
        if context_frame_mask is None:
            context_frame_mask = torch.zeros(B, T, dtype=torch.long, device=device)

        # Start "sliding" across the sequence. We'll do a single pass (not multiple noise steps).
        curr_token = 0
        neural_memory_cache = None
        while curr_token < T:
            # chunk size
            c = min(sliding_context_len, curr_token)  # how many frames of context
            h = min(
                T - curr_token, self.max_tokens - c
            )  # how many frames of "new" to process
            start = curr_token - c
            end = start + (c + h)
            print(
                f"curr_token: {curr_token}, start: {start}, end: {end}, curr_token: {curr_token}, h: {h}, c: {c}"
            )

            chunk_slice = slice(start, end)  # the frames [start, end)

            chunk_x = x[:, chunk_slice].clone()  # ground truth
            chunk_x_noised = x_noised[:, chunk_slice].clone()  # noised input
            chunk_noise_level = noise_level[
                :, chunk_slice
            ]  # noise-level for these frames
            context_chunk = context[:, chunk_slice].clone()  # context frames

            chunk_cond = None
            if conditions is not None:
                chunk_cond = conditions[:, chunk_slice].clone()

            if True:
                chunk_x_noised = torch.cat([chunk_x_noised, context_chunk], dim=2)

            model_pred, *aux_output = self.model_predictions(
                x=chunk_x_noised,
                k=chunk_noise_level,
                external_cond=chunk_cond,
                # guidance_fn=guidance_fn,
                neural_memory_cache=neural_memory_cache,
                **kwargs,
            )
            neural_memory_cache, aux_output = aux_output[0]

            pred = model_pred.model_out  # shape [B, chunk_len, D], e.g. predicted noise
            out_x_pred = model_pred.pred_x_start
            if True:
                pred = pred[:, :, : context_chunk.shape[2]]
                out_x_pred = out_x_pred[:, :, : context_chunk.shape[2]]

            if self.objective == "pred_noise":
                target_chunk = noise[:, chunk_slice]
            elif self.objective == "pred_x0":
                target_chunk = chunk_x
            elif self.objective == "pred_v":
                target_chunk = self.predict_v(
                    chunk_x, chunk_noise_level, noise[:, chunk_slice]
                )
            else:
                raise ValueError(f"Unknown objective {self.objective}")

            chunk_loss = F.mse_loss(
                pred, target_chunk.detach(), reduction="none"
            )  # [B, chunk_len, D]

            loss_weight = self.compute_loss_weights(
                chunk_noise_level, self.loss_weighting.strategy
            )
            loss_weight = self.add_shape_channels(loss_weight)
            chunk_loss = chunk_loss * loss_weight

            # Store chunk_loss into the big accumulator
            loss_accumulator[:, chunk_slice] = chunk_loss
            out_pred[:, chunk_slice] = out_x_pred
            curr_token += h

        return out_pred, loss_accumulator, None

    def forward_1(self, x, external_cond, k, context_frame_mask=None, **kwargs):
        batch_size, n_frames, *x_shape = x.shape
        xs_pred = []
        curr_frame = 0

        # context
        n_context_frames = (1 - context_frame_mask.int()).sum(-1)[0].item()
        xs_pred = x[:, :n_context_frames].clone()
        # curr_frame += n_context_frames
        chunk_size = self.max_tokens

        xs_pred_noised = xs_pred.clone()

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        gt_xs_noised = self.q_sample(x_start=x, k=k, noise=noise)
        context = x[:, :n_context_frames].clone()
        context = torch.cat(
            [
                context,
                torch.zeros(batch_size, n_frames - n_context_frames, *x_shape).to(
                    x.device, x.dtype
                ),
            ],
            dim=1,
        )
        xs_in_noised = torch.cat([gt_xs_noised, context], 2)
        x_pred = torch.zeros_like(x)

        total_loss = torch.zeros_like(x)
        neural_memory_cache = None
        while curr_frame < n_frames:
            if chunk_size > 0:
                horizon = min(n_frames - curr_frame, chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= n_frames, "horizon exceeds the number of tokens."

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame + horizon - self.max_tokens)

            print(
                f"start_frame: {start_frame}, curr_frame: {curr_frame}, horizon: {horizon}"
            )

            chunk_x = x[:, start_frame : start_frame + self.max_tokens]
            chunk_noise = noise[:, start_frame : start_frame + self.max_tokens]
            chunk_xs_in_noised = xs_in_noised[
                :, start_frame : start_frame + self.max_tokens
            ]
            chunk_cond = (
                external_cond[:, start_frame : start_frame + self.max_tokens]
                if external_cond is not None
                else None
            )
            chunk_k = k[:, start_frame : start_frame + self.max_tokens]

            model_pred, *aux_output = self.model_predictions(
                x=chunk_xs_in_noised,
                k=chunk_k,
                external_cond=chunk_cond,
                neural_memory_cache=neural_memory_cache,
                **kwargs,
            )
            neural_memory_cache, aux_output = aux_output[0]

            out_pred = model_pred.model_out
            out_x_pred = model_pred.pred_x_start

            if context_frame_mask is not None:
                out_pred = out_pred[:, :, :3]
                out_x_pred = out_x_pred[:, :, :3]

            pred = out_pred
            x_pred[:, start_frame : start_frame + horizon] = out_x_pred[:, :horizon]

            if self.objective == "pred_noise":
                target = noise
            elif self.objective == "pred_x0":
                target = x
            elif self.objective == "pred_v":
                target = self.predict_v(chunk_x, chunk_k, chunk_noise)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            loss = F.mse_loss(pred, target.detach(), reduction="none")

            loss_weight = self.compute_loss_weights(
                chunk_k, self.loss_weighting.strategy
            )
            loss_weight = self.add_shape_channels(loss_weight)
            loss = loss * loss_weight
            total_loss[:, start_frame : start_frame + horizon] = loss[:, :horizon]

            curr_frame += horizon

        return x_pred, total_loss, None
        # return x_pred[:, :n_frames], total_loss[:, :n_frames], aux_output

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        context_frame_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self.q_sample(x_start=x, k=k, noise=noise)
        if context_frame_mask is not None:
            context_frames = x * (1 - context_frame_mask[..., None, None, None].int())
            #            if torch.rand(1) < 0.2:
            #                context_frames = torch.zeros_like(context_frames)
            #
            noised_x = torch.cat([noised_x, context_frames], dim=2)
        model_pred, *aux_output = self.model_predictions(
            x=noised_x, k=k, external_cond=external_cond, **kwargs
        )

        pred = model_pred.model_out
        x_pred = model_pred.pred_x_start
        if context_frame_mask is not None:
            pred = pred[:, :, :3]
            x_pred = x_pred[:, :, :3]

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, k, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred, target.detach(), reduction="none")

        loss_weight = self.compute_loss_weights(k, self.loss_weighting.strategy)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss, None

    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
        shape = indices.shape
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)

    def sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        cfg_scale: float = 1.0,
        **kwargs,
    ):
        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                guidance_fn=guidance_fn,
                cfg_scale=cfg_scale,
                **kwargs,
            )

        # FIXME: temporary code for checking ddpm sampling
        assert torch.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            curr_noise_level=curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            guidance_fn=guidance_fn,
        )

    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
    ):
        if guidance_fn is not None:
            raise NotImplementedError("guidance_fn is not yet implmented for ddpm.")

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            k=clipped_curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
        )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred)

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        cfg_scale: float = 1.0,
        **kwargs,
    ):

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                    aux_input=aux_input,
                )

                guidance_loss = guidance_fn(
                    xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                )

                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]
                grad = torch.nan_to_num(grad, nan=0.0)

                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = torch.where(
                    alpha > 0,  # to avoid NaN from zero terminal SNR
                    self.predict_start_from_noise(
                        x, clipped_curr_noise_level, pred_noise
                    ),
                    model_pred.pred_x_start,
                )

        else:
            model_pred, *aux_output = self.model_predictions(
                x=x,
                k=clipped_curr_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                cfg_scale=cfg_scale,
                **kwargs,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            x,
            x_pred,
        )

        return x_pred, aux_output

    def estimate_noise_level(self, x, mu=None):
        # x ~ ( B, T, C, ...)
        if mu is None:
            mu = torch.zeros_like(x)
        x = x - mu
        mse = reduce(x**2, "b t ... -> b t", "mean")
        ll_except_c = -self.log_one_minus_alphas_cumprod[None, None] - mse[
            ..., None
        ] * self.alphas_cumprod[None, None] / (1 - self.alphas_cumprod[None, None])
        k = torch.argmax(ll_except_c, -1)
        return k
