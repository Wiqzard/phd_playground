import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional, Dict
from collections import namedtuple

from .discrete_diffusion import DiscreteDiffusion, ModelPrediction  # or wherever you keep these


class CosineNoiseSchedule(nn.Module):
    """
    A minimal cosine noise schedule for continuous-time diffusion,
    parameterized by logSNR-min, logSNR-max, and a shift factor.
    """

    def __init__(
        self,
        logsnr_min: float = -15.0,
        logsnr_max: float = 15.0,
        shift: float = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            "t_min",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "t_max",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float32))),
            persistent=False,
        )
        # shift is applied as logSNR += log(shift^2)
        self.register_buffer(
            "shift",
            2.0 * torch.log(torch.tensor(shift, dtype=torch.float32)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t in [0, 1], output logSNR(t).
        """
        return -2.0 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min))) + self.shift


class ContinuousDiffusion(DiscreteDiffusion):
    """
    A minimal continuous-time diffusion class that reuses the DiscreteDiffusion
    interface but overrides scheduling/buffering logic. By design:

    - Only supports objective = 'pred_v'
    - Only supports loss_weighting.strategy = 'sigmoid'
    """

    def __init__(
        self,
        model: nn.Module,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        timesteps: int = 1000,
        sampling_timesteps: int = 50,
        beta_schedule: str = "cosine",
        # Noise schedule (continuous) parameters:
        training_schedule: str = "cosine",
        schedule_logsnr_min: float = -15.0,
        schedule_logsnr_max: float = 15.0,
        training_schedule_shift: float = 1.0,
        # Diffusion objective & weighting:
        objective: str = "pred_v",
        loss_weighting: Dict = None,  # e.g. {"strategy": "sigmoid", "sigmoid_bias": 0.0}
        # Other:
        precond_scale: float = 1.0,
        clip_noise: float = 20.0,
        ddim_sampling_eta: float = 0.0,
        use_causal_mask: bool = False,
        schedule_fn_kwargs: Dict = {},
    ):
        """
        Args:
            model: The neural network model used for predicting noise/v/etc.
            x_shape: The shape of each data sample (e.g., (channels, height, width)).
            max_tokens: The maximum chunk size or tokens processed at once.
            external_cond_dim: Dimension of external conditioning (if used).
            timesteps: Total number of discrete “timesteps” (still used for sampling).
            sampling_timesteps: Number of timesteps to use at sampling (DDIM or DDPM).
            schedule_name: Which continuous schedule to use, e.g., 'cosine'.
            schedule_logsnr_min: Minimum logSNR for the schedule.
            schedule_logsnr_max: Maximum logSNR for the schedule.
            schedule_shift: Shift factor in log-space for the schedule.
            objective: Must be 'pred_v' for this continuous version.
            loss_weighting: Must have strategy='sigmoid'. E.g. {"strategy": "sigmoid", "sigmoid_bias": 0.0}
            precond_scale: Scale factor applied to logSNR inside the model for v-pred.
            clip_noise: Clamping range for random noise.
            ddim_sampling_eta: DDIM sampling hyper-parameter (as in DiscreteDiffusion).
            use_causal_mask: Whether to use a causal strategy in weighting (not typically used here).
        """
        if loss_weighting is None:
            loss_weighting = {"strategy": "sigmoid", "sigmoid_bias": 0.0}

        # Validate required settings for continuous
        if objective != "pred_v":
            raise ValueError("ContinuousDiffusion only supports objective='pred_v'.")
        if loss_weighting.get("strategy", "") != "sigmoid":
            raise ValueError(
                "ContinuousDiffusion only supports loss_weighting.strategy='sigmoid'."
            )

        # We call DiscreteDiffusion.__init__ to reuse its chunking, sampling, etc.
        super().__init__(
            model=model,
            x_shape=x_shape,
            max_tokens=max_tokens,
            external_cond_dim=external_cond_dim,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            beta_schedule=beta_schedule,  # This won't be used for training, but is used for sampling steps
            schedule_fn_kwargs=schedule_fn_kwargs,  # We won't rely on them, but keep to satisfy DiscreteDiffusion's init
            objective=objective,
            loss_weighting=loss_weighting,
            ddim_sampling_eta=ddim_sampling_eta,
            clip_noise=clip_noise,
            use_causal_mask=use_causal_mask,
        )
        # Overwrite discrete flag:
        self.is_discrete = False

        # Additional continuous-diffusion hyperparams
        self.precond_scale = precond_scale
        self.sigmoid_bias = loss_weighting.get("sigmoid_bias", 0.0)

        # Build the chosen continuous schedule
        # You can make it swappable if you need more than just "cosine"
        if training_schedule == "cosine":
            self.training_schedule = CosineNoiseSchedule(
                logsnr_min=schedule_logsnr_min,
                logsnr_max=schedule_logsnr_max,
                shift=training_schedule_shift,
            )
        else:
            raise ValueError(f"Unknown continuous schedule '{training_schedule}'.")

    def model_predictions(
        self,
        x,
        k,
        external_cond=None,
        external_cond_mask=None,
        uncond_cond=None,
        uncond_cond_mask=None,
        cfg_scale=1,
        **kwargs,
    ):

        if cfg_scale == 1.0 or uncond_cond is None:
            model_output, *other_output = self.model(
                x, self.precond_scale * self.logsnr[k], external_cond, external_cond_mask, **kwargs
            )
        else:
            model_output_cond, *other_output_cond = self.model(
                x, self.precond_scale * self.logsnr[k], external_cond, external_cond_mask, **kwargs
            )

            if uncond_cond is None:
                uncond_cond = torch.zeros_like(external_cond)
                uncond_cond_mask = torch.zeros_like(external_cond_mask)
            kwargs.pop("neural_memory_cache", None)
            model_output_uncond, *other_output_uncond = self.model(
                x, self.precond_scale * self.logsnr[k], uncond_cond, uncond_cond_mask, **kwargs
            )
            model_output = model_output_uncond + cfg_scale * (
                model_output_cond - model_output_uncond
            )
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

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        **kwargs,
    ):
        """
        Forward pass for training. Interprets `k` as a continuous time t ∈ [0,1].
        1. We compute the logSNR(t).
        2. Create a noised input x_t = α(t)*x + σ(t)*noise.
        3. Model predicts v(x_t).
        4. We derive the noise prediction and x_0 prediction from v.
        5. MSE loss vs. the real noise (with a sigmoid weighting).
        """
        # Convert continuous time t in [0,1] to logSNR
        logsnr = self.training_schedule(k)  # shape = [B, T, ...]

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())

        # Noised input
        x_t = alpha_t * x + sigma_t * noise

        # Model does v-pred
        # (If the model needs shape [B, T, D], ensure it matches your architecture.)
        v_pred = self.model(
            x_t,
            self.precond_scale * logsnr,  # e.g. condition on scaled logSNR
            external_cond,
            **kwargs,
        )

        # Post-process to noise_pred & x_0 pred:
        noise_pred = alpha_t * v_pred + sigma_t * x_t  # => predicted noise
        x_pred = alpha_t * x_t - sigma_t * v_pred  # => predicted x_0

        # MSE loss wrt real noise
        loss = F.mse_loss(noise_pred, noise, reduction="none")

        # Sigmoid weighting: shape-match with add_shape_channels
        bias = self.sigmoid_bias
        loss_weight = torch.sigmoid(bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss, None

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
                raise ValueError("When T > max_tokens, you must specify sliding_context_len.")
            else:
                sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        # -----------------------------------------------------
        # Create the noised input at this single noise_level
        # -----------------------------------------------------
        # You presumably have a q_sample(...) that does: x_noised = sqrt(α)*x + sqrt(1-α)*noise, etc.
        # Or you can replicate your approach in forward_1(...)
        logsnr = self.training_schedule(noise_level)
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
        x_noised = alpha_t * x + sigma_t * noise

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
            h = min(T - curr_token, self.max_tokens - c)  # how many frames of "new" to process
            start = curr_token - c
            end = start + (c + h)
            print(
                f"curr_token: {curr_token}, start: {start}, end: {end}, curr_token: {curr_token}, h: {h}, c: {c}"
            )

            chunk_slice = slice(start, end)  # the frames [start, end)

            # chunk_x = x[:, chunk_slice].clone()  # ground truth
            chunk_noise = noise[:, chunk_slice].clone()  # noise
            chunk_x_noised = x_noised[:, chunk_slice].clone()  # noised input
            chunk_noise_level = noise_level[:, chunk_slice]  # noise-level for these frames
            context_chunk = context[:, chunk_slice].clone()  # context frames
            chunk_logsnr = logsnr[:, chunk_slice].clone()
            chunk_alpha_t = alpha_t[:, chunk_slice].clone()
            chunk_sigma_t = sigma_t[:, chunk_slice].clone()

            chunk_cond = None
            if conditions is not None:
                chunk_cond = conditions[:, chunk_slice].clone()

            if True:
                chunk_x_noised = torch.cat([chunk_x_noised, context_chunk], dim=2)

            v_pred, *aux_output = self.model(
                x=chunk_x_noised,
                noise_levels=self.precond_scale * chunk_logsnr,
                external_cond=chunk_cond,
                neural_memory_cache=neural_memory_cache,
                **kwargs,
            )

            neural_memory_cache, aux_output = aux_output  # [0]

            noise_pred = chunk_alpha_t * v_pred + chunk_sigma_t * chunk_x_noised
            x_pred = chunk_alpha_t * chunk_x_noised - chunk_sigma_t * v_pred
            if True:
                noise_pred = noise_pred[:, :, : context_chunk.shape[2]]
                x_pred = x_pred[:, :, : context_chunk.shape[2]]
                # v_pred = v_pred[:, :, : context_chunk.shape[2]]

            chunk_loss = F.mse_loss(
                noise_pred, chunk_noise.detach(), reduction="none"
            )  # [B, chunk_len, D]
            loss_weight = torch.sigmoid(self.sigmoid_bias - chunk_logsnr)
            loss_weight = self.add_shape_channels(loss_weight)
            chunk_loss = chunk_loss * loss_weight

            # Store chunk_loss into the big accumulator
            loss_accumulator[:, chunk_slice] = chunk_loss
            out_pred[:, chunk_slice] = x_pred
            curr_token += h

        return out_pred, loss_accumulator, None
