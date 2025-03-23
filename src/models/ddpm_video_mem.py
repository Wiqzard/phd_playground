from typing import Any, Dict, Optional, Union, Sequence, Tuple, Callable, Literal
from functools import partial
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
import wandb
from tqdm import tqdm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.optimizer import Optimizer
from lightning.pytorch.utilities import grad_norm

from einops import rearrange, repeat, reduce

# Import the DDPM scheduler from diffusers
# from src.models.components.moe_lora import inject_lora, disable_all_adapters, enable_all_adapters, set_lora_trainability, reset_all_lora_parameters, get_lora_adapter_parameters, get_lora_adapter_parameters, set_global_trainability , get_global_parameters
from diffusers import DDPMScheduler
from src.models.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from src.models.common import BaseLightningTrainer
from src.models.components.diffusion import DiscreteDiffusion, ContinuousDiffusion

from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.logging_utils import log_video
from utils.print_utils import cyan
from utils.torch_utils import freeze_model, bernoulli_tensor

###############################################################################
# Helper Function: Video Preprocessing for Logging (Optional)
###############################################################################


def preprocess_and_format_video(x: torch.Tensor) -> any:
    """
    Preprocess a video tensor and format it for logging.
    Expected input shape: (B, C, T, H, W).

    Returns:
        numpy.ndarray: Video data in shape (B, T, H, W, C) scaled to [0, 255].
    """
    # Clamp to [-1, 1], then rescale to [0, 1]
    x = x.clamp(-1.0, 1.0)
    x = x / 2.0 + 0.5
    # If batch dimension is missing, add one (should not happen here)
    if x.dim() == 4:
        x = x.unsqueeze(0)
    # Rearrange from (B, C, T, H, W) to (B, T, C, H, W)
    video_to_log = x.permute(0, 2, 1, 3, 4)
    video_to_log = (video_to_log * 255).to(torch.uint8).cpu().numpy()
    return video_to_log


###############################################################################
# Diffusion Model Trainer using Diffusers DDPM Scheduler
###############################################################################


# LightningModule
class DiffusionModelTrainer(BaseLightningTrainer):
    def __init__(
        self,
        #model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        scheduler: Optional[Any] = None,
        compile: bool = False,
        num_gen_steps: int = 10,
        meta_learning: bool = False,
        num_inner_steps: int = 10,
        inner_lr: float = 1e-4,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        is_latent_diffusion: bool = False,
        is_latent_online: bool = False,
        latent_downsampling_factor: Tuple[int, int, int] = (1, 1),
        x_shape: Tuple[int, int, int] = (3, 64, 64),
        diffusion_model: Optional[Union[DiscreteDiffusion, ContinuousDiffusion]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the diffusion model trainer.

        Args:
            model: The neural network (a UNet3D) that predicts the noise given a noisy input and timestep.
            optimizer: The optimizer class (e.g., torch.optim.Adam).
            num_train_timesteps: Number of diffusion timesteps for training.
            beta_schedule: Type of beta schedule ("linear", "cosine", etc.).
            compile: Whether to compile the model (requires PyTorch 2.x).
            lr: Learning rate.
            num_inference_steps: Number of steps to use during sampling in validation.
            kwargs: Additional hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=("model", "diffusion_model", "optimizer", "lr_scheduler", "scheduler"))
        #self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.compile_model = compile
        self.diffusion_model = diffusion_model

        # Initialize the DDPM scheduler for training
        if scheduler is not None:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule
            )
        else:
            self.scheduler = scheduler

        self.temporal_downsampling_factor = latent_downsampling_factor[0]

        self.tasks = [task for task in self.hparams.tasks]

        self.num_logged_videos = 0

        # Metrics for logging
        self.train_loss = MeanMetric()
        self.recon_loss = MeanMetric()
        self.diffusion_loss = MeanMetric()

        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    # ---------------------------------------------------------------------
    # Prepare Model, Optimizer, and Metrics
    # ---------------------------------------------------------------------

    def _load_vae(self) -> None:
        """
        PUT THIS IN THE CONFIG
        Load the pretrained VAE model.

        """
        vae_cls = VideoVAE if self.is_latent_video_vae else ImageVAE
        self.vae = vae_cls.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            torch_dtype=(
                torch.float16 if self.cfg.vae.use_fp16 else torch.float32
            ),  # only for Diffuser's ImageVAE
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)

        freeze_model(self.vae)

    def _metrics(
        self,
        task: Literal["prediction", "interpolation"],
    ) -> Optional[VideoMetric]:
        """
        Get the appropriate metrics object for the given task.
        """
        return getattr(self, f"metrics_{task}", None)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        self.diffusion_model = self.diffusion_model(
            x_shape=self.hparams.x_shape,
            max_tokens=self.max_tokens,
            external_cond_dim=self.hparams.external_cond_dim,
        )

        if self.hparams.ckpt_path:
            self.load_state_dict(
                torch.load(self.hparams.ckpt_path, weights_only=False)["state_dict"]
            )

        if self.hparams.lora_finetune:
            for module in self.model.modules():
                for param in module.parameters():
                    param.requires_grad = False

            # for module in self.model.memory_layers.modules():
            #    for param in module.parameters():
            #        param.requires_grad = True

            from peft import LoraConfig, TaskType, get_peft_model

            # Create a LoRA configuration
            lora_config = LoraConfig(
                r=8,  # rank
                lora_alpha=32,  # alpha scaling
                lora_dropout=0.00,  # dropout, can be 0.0 as well
                bias="none",
                target_modules=["qkv", "proj"],
            )

            ## Wrap your DiT model with LoRA
            self.model = get_peft_model(self.model, lora_config)

            self.print_trainable_parameters()
            for module in self.model.memory_layers.modules():
                for param in module.parameters():
                    param.requires_grad = True

        if self.compile_model and stage == "fit":
            if self.compile_model == "true_without_ddp_optimizer":
                # NOTE: `cfg.compile` should be set to this value when using `torch.compile` with DDP & Gradient Checkpointing
                # Otherwise, torch.compile will raise an error.
                # Reference: https://github.com/pytorch/pytorch/issues/104674
                # pylint: disable=protected-access
                torch._dynamo.config.optimize_ddp = False

        #self.diffusion_model = torch.compile(
        #    self.diffusion_model ,
        #    disable=not self.compile_model,
        #)
        self.register_data_mean_std(self.hparams.data_mean, self.hparams.data_std)

        # 2. VAE model
        if self.hparams.is_latent_diffusion and self.hparams.is_latent_online:
            self._load_vae()
        else:
            self.vae = None

        # 3. Metrics
        registry = SharedVideoMetricModelRegistry()
        metric_types = self.hparams.metrics
        for task in self.tasks:
            match task:
                case "prediction":
                    self.metrics_prediction = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.hparams.metrics_batch_size,
                    )
                case "interpolation":
                    assert (
                        not self.hparams.use_causal_mask
                        # and not self.hparams.is_full_sequence
                        and self.max_tokens > 2
                    ), "To execute interpolation, the model must be non-causal, not full sequence, and be able to process more than 2 tokens."
                    self.metrics_interpolation = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.hparams.metrics_batch_size,
                    )

    # ---------------------------------------------------------------------
    # Length-related Properties and Utils
    # NOTE: "Frame" and "Token" should be distinguished carefully.
    # "Frame" refers to original unit of data loaded from dataset.
    # "Token" refers to the unit of data processed by the diffusion model.
    # The two differ when using a VAE for latent diffusion.
    # ---------------------------------------------------------------------

    def _n_frames_to_n_tokens(self, n_frames: int) -> int:
        """
        Converts the number of frames to the number of tokens.
        - Chunk-wise VideoVAE: 1st frame -> 1st token, then every self.temporal_downsampling_factor frames -> next token.
        - ImageVAE or Non-latent Diffusion: 1 token per frame.
        """
        return (n_frames - 1) // self.temporal_downsampling_factor + 1

    def _n_tokens_to_n_frames(self, n_tokens: int) -> int:
        """
        Converts the number of tokens to the number of frames.
        """
        return (n_tokens - 1) * self.temporal_downsampling_factor + 1

    # ---------------------------------------------------------------------
    # NOTE: max_{frames, tokens} indicates the maximum number of frames/tokens
    # that the model can process within a single forward pass.
    # ---------------------------------------------------------------------

    @property
    def max_frames(self) -> int:
        return self.hparams.max_frames

    @property
    def max_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.max_frames)

    # ---------------------------------------------------------------------
    # NOTE: n_{frames, tokens} indicates the number of frames/tokens
    # that the model actually processes during training/validation.
    # During validation, it may be different from max_{frames, tokens},
    # ---------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return self.max_frames if self.trainer.training else self.hparams.n_frames

    @property
    def n_context_frames(self) -> int:
        return self.hparams.context_frames

    @property
    def n_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_frames)

    @property
    def n_context_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_context_frames)

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------

    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        """
        Preprocess the batch before training/validation.

        Args:
            batch (Dict): The batch of data. Contains "videos" or "latents", (optional) "conditions", and "masks".
            dataloader_idx (int): The index of the dataloader.
        Returns:
            xs (Tensor, "B n_tokens *x_shape"): Tokens to be processed by the model.
            conditions (Optional[Tensor], "B n_tokens d"): External conditions for the tokens.
            masks (Tensor, "B n_tokens"): Masks for the tokens.
            gt_videos (Optional[Tensor], "B n_frames *x_shape"): Optional ground truth videos, used for validation in latent diffusion.
        """
        # 1. Tokenize the videos and optionally prepare the ground truth videos
        gt_videos = None
        if self.hparams.is_latent_diffusion:
            xs = self._encode(batch["videos"]) if self.is_latent_online else batch["latents"]
            if "videos" in batch:
                gt_videos = batch["videos"]
        else:
            xs = batch["videos"]
        xs = self._normalize_x(xs)

        # 2. Prepare external conditions
        conditions = batch.get("conds", None)

        # 3. Prepare the masks
        if "masks" in batch:
            assert (
                not self.is_latent_video_vae
            ), "Masks should not be provided from the dataset when using VideoVAE."
        else:
            masks = torch.ones(*xs.shape[:2]).bool().to(self.device)

        return xs, conditions, masks, gt_videos

    # ---------------------------------------------------------------------
    # Logging (Metrics, Videos)
    # ---------------------------------------------------------------------

    def _update_metrics(self, all_videos: Dict[str, Tensor]) -> None:
        """Update all metrics during validation/test step."""
        if (
            self.hparams.n_metrics_frames is not None
        ):  # only consider the first n_metrics_frames for evaluation
            all_videos = {k: v[:, : self.hparams.n_metrics_frames] for k, v in all_videos.items()}

        gt_videos = all_videos["gt"]
        for task in self.tasks:
            metric = self._metrics(task)
            videos = all_videos[task]
            context_mask = torch.zeros(self.n_frames).bool().to(self.device)
            match task:
                case "prediction":
                    context_mask[: self.n_context_frames] = True
                case "interpolation":
                    context_mask[[0, -1]] = True
            if self.hparams.n_metrics_frames is not None:
                context_mask = context_mask[: self.hparams.n_metrics_frames]
            metric(videos, gt_videos, context_mask=context_mask)

    def _log_videos(self, all_videos: Dict[str, Tensor], namespace: str) -> None:
        """Log videos during validation/test step."""
        all_videos = self.gather_data(all_videos)
        batch_size, n_frames = all_videos["gt"].shape[:2]

        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.hparams.log_max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.hparams.log_max_num_videos - self.num_logged_videos,
            batch_size,
        )
        cut_videos = lambda x: x[:num_videos_to_log]

        for task in self.tasks:
            log_video(
                cut_videos(all_videos[task]),
                cut_videos(all_videos["gt"]),
                step=None if namespace == "test" else self.global_step,
                namespace=f"{task}_vis",
                logger=self.logger.experiment,
                indent=self.num_logged_videos,
                raw_dir=self.trainer.log_dir,
                context_frames=(
                    self.n_context_frames
                    if task == "prediction"
                    else torch.tensor([0, n_frames - 1], device=self.device, dtype=torch.long)
                ),
                captions=f"{task} | gt",
            )

        self.num_logged_videos += batch_size

    # ---------------------------------------------------------------------
    # Data Preprocessing Utils
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _process_conditions(
        self,
        conditions: Optional[Tensor],
        noise_levels: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Post-process the conditions before feeding them to the model.
        For example, conditions that should be computed relatively (e.g. relative poses)
        should be processed here instead of the dataset.

        Args:
            conditions (Optional[Tensor], "B T ..."): The external conditions for the video.
            noise_levels (Optional[Tensor], "B T"): Current noise levels for each token during sampling
        """

        if conditions is None:
            return conditions
        match self.hparams.external_cond_processing:
            case "mask_first":
                mask = torch.ones_like(conditions)
                mask[:, :1, : self.hparams.external_cond_dim] = 0
                return conditions * mask
            case _:
                raise NotImplementedError(
                    f"External condition processing {self.cfg.external_cond_processing} is not implemented."
                )

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    def forward(self, x, conditions=None, timesteps=None, loss=True):
        b, t, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        noise = torch.randn_like(x)
        batch_size = x.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (batch_size,), device=x.device
            ).long()
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)
        y = torch.zeros((batch_size,), device=x.device, dtype=torch.long)

        video = []
        memory_states = None

        timesteps = timesteps.unsqueeze(1)  # .repeat(1, t)
        for i in range(t):
            x_noisy_i = x_noisy[:, :, i].unsqueeze(2)
            padding = torch.zeros_like(x_noisy_i)
            if i == 0:
                # cat first frame in channel dimension
                frame_padding = torch.zeros_like(x_noisy_i)
                frame_padding[:, :, 0] = x[:, :, 0]
                x_noisy_i = torch.cat([x_noisy_i, frame_padding], dim=1)
            else:
                # cat padding in channel dimension
                x_noisy_i = torch.cat([x_noisy_i, padding], dim=1)
            x_noisy_i = x_noisy_i.permute(0, 2, 1, 3, 4)

            if False:
                noise_pred, memory_states, suprises = self.model.forward(
                    x_noisy_i, timesteps, cond=y, memory_states=memory_states, return_memory=True
                )
            else:
                # noise_pred = self.model.forward(x_noisy_i, timesteps, cond=y, cache_params=memory_states, use_cache=False)
                y = None

                noise_pred, memory_states, aux_output = self.model.forward(
                    x_noisy_i, timesteps, external_cond=y, neural_memory_cache=memory_states
                )
                noise_pred = noise_pred[:, :, :3]
                # noise_pred, memory_states = self.model.forward(x_noisy_i, timesteps, cond=y, cache_params=memory_states, use_cache=True, run=i)
                suprises = torch.zeros_like(noise_pred)
            # b, t, c, h, w
            x_pred = self.scheduler.get_velocity(
                sample=noise_pred, noise=x_noisy_i[:, :, :3], timesteps=timesteps.squeeze(-1)
            )
            video.append(x_pred)

        x_pred = torch.cat(video, dim=1)
        # x_pred = x_pred.permute(0, 2, 1, 3, 4)
        if not loss:
            return x_pred, timesteps

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)

        while len(weights.shape) < len(x_pred.shape):
            weights = weights.unsqueeze(-1)
        diffusion_loss = self.loss(x_pred.permute(0, 2, 1, 3, 4), x, weights)  # adjust as needed
        return x_pred, diffusion_loss, None  # , #suprises

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, namespace: str = "training"
    ) -> STEP_OUTPUT:
        """Training step"""
        xs, conditions, masks, *_ = batch

        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        xs_pred, loss, aux_loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
            context_frame_mask=masks,
        )
        loss = self._reweight_loss(loss, masks)
        if batch_idx % self.trainer.log_every_n_steps == 0:  # cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                prog_bar=True,
            )
            if aux_loss is not None:
                for key, value in aux_loss.items():
                    self.log(
                        f"{namespace}/{key}",
                        value,
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                        prog_bar=True,
                    )

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

        #xs, conditions, masks, *_ = batch

        ## ----- Diffusion branch: Compute diffusion loss -----
        #timesteps = None
        #xs_pred, diffusion_loss, aux_loss = self.forward(
        #    xs, conditions=self._process_conditions(conditions), loss=True, timesteps=timesteps
        #)
        ## surprises = surprises[-1].mean()

        ## ----- Combine losses -----
        ## You might want to weight the autoencoder loss differently (using lambda_ae)
        ### Update metrics and log losses
        ## self.train_loss.update(total_loss)
        ## self.recon_loss.update(recon_loss)
        ## self.diffusion_loss.update(diffusion_loss)
        ## self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        ## self.log("train/surprises", surprises, on_step=True, on_epoch=False, prog_bar=True)
        ## self.log("train/diffusion_loss", diffusion_loss, on_step=True, on_epoch=False, prog_bar=True)

        #if batch_idx % self.trainer.log_every_n_steps == 0:  # cfg.logging.loss_freq == 0:
        #    self.log(
        #        f"{namespace}/loss",
        #        diffusion_loss,
        #        on_step=namespace == "training",
        #        on_epoch=namespace != "training",
        #        sync_dist=True,
        #        prog_bar=True,
        #    )
        #    if aux_loss is not None:
        #        for key, value in aux_loss.items():
        #            self.log(
        #                f"{namespace}/{key}",
        #                value,
        #                on_step=namespace == "training",
        #                on_epoch=namespace != "training",
        #                sync_dist=True,
        #                prog_bar=True,
        #            )

        #xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))
        #output_dict = {
        #    "loss": diffusion_loss,
        #    "xs_pred": xs_pred,
        #    "xs": xs,
        #}
        #return output_dict

    def loss(self, pred, target, weight=None):
        if weight is None:
            weight = torch.ones_like(pred)
        return torch.mean((weight * (pred - target) ** 2).reshape(pred.shape[0], -1))

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if (
            self.hparams.log_grad_norm_freq
            and self.global_step % self.hparams.log_grad_norm_freq == 0
        ):
            norms = grad_norm(self.model, norm_type=2)
            # NOTE: `norms` need not be gathered, as they are already uniform across all devices
            self.log_dict(norms)

    


    # ---------------------------------------------------------------------
    # Validation & Test
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """Validation step"""
        # 1. If running validation while training a model, directly evaluate
        # the denoising performance to detect overfitting, etc.
        # Logs the "denoising_vis" visualization as well as "validation/loss" metric.
        if self.trainer.state.fn == "fit":
            self._eval_denoising(batch, batch_idx, namespace=namespace)

        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics.
        if not (self.trainer.sanity_checking and not self.hparams.log_sanity_generation):
            all_videos = self._sample_all_videos(batch, batch_idx, namespace)
            self._update_metrics(all_videos)
            self._log_videos(all_videos, namespace)

    def on_validation_epoch_start(self) -> None:
        if self.hparams.log_deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.global_rank + self.trainer.world_size * self.hparams.log_deterministic
            )
        if self.hparams.is_latent_diffusion and not self.hparams.is_latent_online:
            self._load_vae()

    def on_validation_epoch_end(self, namespace="validation") -> None:
        self.generator = None
        if self.hparams.is_latent_diffusion and not self.hparams.is_latent_online:
            self.vae = None
        self.num_logged_videos = 0

        if self.trainer.sanity_checking and not self.hparams.log_sanity_generation:
            return

        for task in self.tasks:
            self.log_dict(
                self._metrics(task).log(task),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    # ---------------------------------------------------------------------
    # Denoising Evaluation
    # ---------------------------------------------------------------------

    def _eval_denoising(self, batch, batch_idx, namespace="training") -> None:
        """Evaluate the denoising performance during training."""
        xs, conditions, masks, gt_videos = batch

        xs = xs[:, : self.max_tokens]
        if conditions is not None:
            conditions = conditions[:, : self.max_tokens]
        masks = masks[:, : self.max_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.max_frames]

        batch = (xs, conditions, masks, gt_videos)
        output = self.training_step(batch, batch_idx, namespace=namespace)

        gt_videos = gt_videos if self.hparams.is_latent_diffusion else output["xs"]
        recons = output["xs_pred"]
        if self.hparams.is_latent_diffusion:
            recons = self._decode(recons)

        if recons.shape[1] < gt_videos.shape[1]:  # recons.ndim is 5
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos.shape[1] - recons.shape[1], 0, 0),
            )

        gt_videos, recons = self.gather_data((gt_videos, recons))

        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.hparams.log_max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.hparams.log_max_num_videos - self.num_logged_videos,
            gt_videos.shape[0],
        )
        log_video(
            recons[:num_videos_to_log],
            gt_videos[:num_videos_to_log],
            step=self.global_step,
            namespace="denoising_vis",
            logger=self.logger.experiment,
            indent=self.num_logged_videos,
            captions="denoised | gt",
        )

        ## log validation loss
        # loss = output["loss"]
        # self.log(f"{namespace}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation"
    ) -> Optional[Dict[str, torch.Tensor]]:
        xs, conditions, *_, gt_videos = batch
        all_videos: Dict[str, torch.Tensor] = {"gt": xs}

        for task in self.tasks:
            sample_fn = self._predict_videos if task == "prediction" else self._interpolate_videos
            all_videos[task] = sample_fn(xs, conditions=conditions)

        # remove None values
        all_videos = {k: v for k, v in all_videos.items() if v is not None}
        # unnormalize/detach the videos
        all_videos = {k: self._unnormalize_x(v).detach() for k, v in all_videos.items()}
        # decode latents if using latents
        if self.hparams.is_latent_diffusion:
            all_videos = {
                k: self._decode(v) if k != "gt" else gt_videos for k, v in all_videos.items()
            }

        # replace the context frames of video predictions with the ground truth
        if "prediction" in all_videos:
            all_videos["prediction"][:, : self.n_context_frames] = all_videos["gt"][
                :, : self.n_context_frames
            ]
        return all_videos

    def _predict_videos(
        self, xs: torch.Tensor, conditions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict the videos with the given context.
        Optionally do rolling/sliding windows if the sequence is large.
        """
        xs_pred = xs.clone()  # [b, t, c, h, w]

        # Basic single-pass call to _predict_sequence
        xs_pred, _ = self._predict_sequence(
            context=xs_pred[:, : self.n_context_tokens],
            length=xs_pred.shape[1],
            conditions=conditions,
            reconstruction_guidance=self.hparams.reconstruction_guidance,
            sliding_context_len=self.hparams.sliding_context_len # or (self.max_tokens // 2),
        )
        return xs_pred

    
    # ---------------------------------------------------------------------
    # Training Utils
    # ---------------------------------------------------------------------

    def _get_training_noise_levels(
        self, xs: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """Generate random noise levels for training."""
        batch_size, n_tokens, *_ = xs.shape

        # random function different for continuous and discrete diffusion
        rand_fn = partial(
            *(
                (torch.rand,)
                if not self.diffusion_model.is_discrete # self.cfg.diffusion.is_continuous
                else (torch.randint, 0, self.diffusion_model.timesteps)
            ),
            device=xs.device,
            generator=self.generator,
        )

        # baseline training (SD: fixed_context, BD: variable_context)
        context_mask = None

        #if self.cfg.variable_context.enabled:
        #    assert (
        #        not self.cfg.fixed_context.enabled
        #    ), "Cannot use both fixed and variable context"
        #    context_mask = bernoulli_tensor(
        #        (batch_size, n_tokens),
        #        self.cfg.variable_context.prob,
        #        device=self.device,
        #        generator=self.generator,
        #    ).bool()
        #elif self.cfg.fixed_context.enabled:
        context_indices = list(range(self.n_context_tokens))
        context_mask = torch.zeros(
            (batch_size, n_tokens), dtype=torch.bool, device=xs.device
        )
        context_mask[:, context_indices] = True

        match self.hparams.noise_level:
            case "random_independent":  # independent noise levels (Diffusion Forcing)
                noise_levels = rand_fn((batch_size, n_tokens))
            case "random_uniform":  # uniform noise levels (Typical Video Diffusion)
                noise_levels = rand_fn((batch_size, 1)).repeat(1, n_tokens)

        #if self.cfg.uniform_future.enabled:  # simplified training (Appendix A.5)
        #    noise_levels[:, self.n_context_tokens :] = rand_fn((batch_size, 1)).repeat(
        #        1, n_tokens - self.n_context_tokens
        #    )

        ## treat frames that are not available as "full noise"
        #noise_levels = torch.where(
        #    reduce(masks.bool(), "b t ... -> b t", torch.any),
        #    noise_levels,
        #    torch.full_like(
        #        noise_levels,
        #        1 if not self.diffusion_model.is_discrete else self.timesteps - 1,
        #    ),
        #)

        if context_mask is not None:
            # binary dropout training to enable guidance
            dropout = (
                #(
                #    self.cfg.variable_context
                #    if self.cfg.variable_context.enabled
                #    else self.cfg.fixed_context
                #).dropout
                self.hparams.context_dropout
                if self.trainer.training
                else 0.0
            )
            context_noise_levels = bernoulli_tensor(
                (batch_size, 1),
                dropout,
                device=xs.device,
                generator=self.generator,
            )
            if  self.diffusion_model.is_discrete:
                context_noise_levels = context_noise_levels.long() * (
                    self.diffusion_model.timesteps - 1
                )

            if not self.hparams.cat_context_in_c_dim:
                noise_levels = torch.where(context_mask, context_noise_levels, noise_levels)

            # modify masks to exclude context frames from loss computation
            context_mask = rearrange(
                context_mask, "b t -> b t" + " 1" * len(masks.shape[2:])
            )
            masks = torch.where(context_mask, False, masks)

        return noise_levels, masks

    def _reweight_loss(self, loss, weight=None):
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "... -> ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss.mean()



    # ---------------------------------------------------------------------
    # Sampling Utilities
    # ---------------------------------------------------------------------

    def _generate_scheduling_matrix(
        self,
        horizon: int,
        padding: int = 0,
    ):
        """
        Generates a scheduling matrix based on the self.hparams.scheduling_matrix parameter.
        Each row represents a noise-level index (or “timestep”).
        """

        match self.hparams.scheduling_matrix:
            case "full_sequence":
                # Each column has the same countdown from sampling_timesteps -> 0
                scheduling_matrix = np.arange(self.hparams.sampling_timesteps, -1, -1)[:, None]
                scheduling_matrix = np.repeat(scheduling_matrix, horizon, axis=1)

            case "autoregressive":
                # Example pyramid / autoregressive schedule
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    horizon=horizon,
                    timesteps=self.hparams.sampling_timesteps,
                )

            case other:
                raise ValueError(f"Unknown scheduling_matrix type: {other}")

        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

        # Optionally convert from “timestep index” to actual noise levels:
        scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(scheduling_matrix)

        # If we want to pad the extra tokens as pure noise, do so here:
        scheduling_matrix = F.pad(
            scheduling_matrix,
            (0, padding, 0, 0),
            value=self.diffusion_model.timesteps - 1,  # or your “max noise index”
        )

        return scheduling_matrix

    def _predict_sequence(
        self,
        context: torch.Tensor,   # (batch_size, self.n_context_tokens, *x_shape) just gt frames
        length: Optional[int] = None, # self.n_tokens frames  (determined over xs_pred)
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        sliding_context_len: Optional[int] = None, # self.hparams.sliding_context_len 
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence given initial context frames, possibly in a
        rolling/sliding window approach if length > self.max_tokens.
        """
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        batch_size, gt_len, *_ = context.shape
        
        curr_token = gt_len
        xs_pred = context
        record = None

        # How many total diffusion steps for the entire rolling process?
        # The factor accounts for how many “chunks” we will sample

        #if sliding_context_len == 0:
        #    # we have in the first iteration   
        #else:
        #number_of_chunks = 1 + max(0, (length - self.max_tokens)) // (self.max_tokens - sliding_context_len)
        number_of_chunks = math.ceil((length - self.max_tokens) / (self.max_tokens - sliding_context_len)) + 1 

            #number_of_chunks = 1 + max(0, (length - sliding_context_len - 1)) // max(1, (self.max_tokens - sliding_context_len))
        total_passes = self.hparams.sampling_timesteps * number_of_chunks

        pbar = tqdm(
            total=total_passes,
            initial=0,
            desc="Predicting (vanilla diffusion)",
            leave=False,
        )

        # Rolling from left to right until we generate 'length' frames:
        iteration = 0
        while curr_token < length:
            # If storing all steps, forbid sliding windows for simplicity
            if return_all:
                raise ValueError("return_all is not supported with sliding window.")

            # Decide how many frames of context vs. how many frames to generate:
            if sliding_context_len == 0 and iteration == 0:
                c = self.n_context_tokens    
            else: 
                c = min(sliding_context_len, curr_token)


            h =  self.max_tokens - c
            l = c + h # total length fed to the model

            if not self.hparams.generate_in_noise_dim:
                l = (self.n_tokens // self.max_tokens) * self.max_tokens  # total frames to be generated
                h = l - c
                if h < 0:
                    raise ValueError("Context length is larger than the total number of tokens.")
                

            # Prepare next chunk (context chunk + blank frames)
            pad = torch.zeros((batch_size, h, *self.hparams.x_shape), device=self.device)

            context_mask = torch.cat(
                [
                    # Mark c tokens as “context” (1)
                    torch.ones((batch_size, c), dtype=torch.long, device=self.device),
                    # Mark h tokens as “to be generated” (0)
                    torch.zeros((batch_size, h), dtype=torch.long, device=self.device),
                ],
                dim=1,
            )

            context_chunk = torch.cat(
                [xs_pred[:, -c:] if c > 0 else torch.empty_like(pad[:, :0]), pad], dim=1
            )

            if conditions is not None and self.n_tokens > self.max_tokens:  #self.hparams.use_causal_mask:
                cond_slice = conditions[:, curr_token - c : curr_token - c + l]
            else:
                cond_slice= conditions

            # -----------------------------------
            new_pred, record = self._sample_sequence(
                batch_size,
                length=l,
                context=context_chunk,
                context_mask=context_mask,
                conditions=cond_slice,
                guidance_fn=guidance_fn,
                #reconstruction_guidance=reconstruction_guidance,
                #history_guidance=None,
                return_all=return_all,
                number_of_chunks=number_of_chunks,
                pbar=pbar,
            )
            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
            curr_token = xs_pred.shape[1]
            iteration += 1
        pbar.close()
        if xs_pred.shape[1] > length:
            xs_pred = xs_pred[:, :length]

        return xs_pred, record

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        return_all: bool = False,
        number_of_chunks: int = 1,
        pbar: Optional[tqdm] = None,
        guidance_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x_shape = self.hparams.x_shape
        padding = self.max_tokens - length

        scheduling_matrix = self._generate_scheduling_matrix(
            length,
            0,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        # Prepare the record list if we want all intermediate steps
        record = [] if return_all else None

        # Initial random noise
        xs_pred = torch.randn(
            (batch_size, self.max_tokens, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise)

        # Create a single progress bar if none is given
        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )

        if not self.hparams.generate_in_noise_dim:
            xs_pred, record = self._sample_sequence_in_time_dimension(
                batch_size,
                length,
                context,
                context_mask,
                conditions,
                return_all,
                number_of_chunks,
                pbar,
                guidance_fn=guidance_fn,
            )
        else:

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels_full = scheduling_matrix[m]     # shape: [b, t]
                to_noise_levels_full = scheduling_matrix[m + 1]   # shape: [b, t]

                # Decide how many times we loop over chunks
                if self.hparams.generate_in_noise_dim:
                    number_of_generations = number_of_chunks
                else:
                    number_of_generations = 1

                # Prepare final container if you want to piecewise-generate
                final_xs_pred = torch.zeros(
                    (batch_size, self.max_tokens * number_of_generations, *x_shape),
                    device=self.device
                )
                from_noise_levels = from_noise_levels_full
                to_noise_levels   = to_noise_levels_full

                if context is None:
                    context = torch.zeros_like(xs_pred)
                    context_mask = torch.zeros(
                        (batch_size, self.max_tokens),
                        dtype=torch.long,
                        device=self.device
                    )

                # If we do NOT concatenate context into the diffusion model's channels,
                # we literally replace the context frames in xs_pred with the given context
                if not self.hparams.cat_context_in_c_dim:
                    xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

                # Make a backup for context tokens so we can revert them after diffusion
                xs_pred_prev = xs_pred.clone()
                if return_all:
                    record.append(xs_pred.clone())

                # If we DO concatenate context in c-dim, cat them here
                if self.hparams.cat_context_in_c_dim:
                    xs_pred = torch.cat([xs_pred, context], dim=2)  # depends on your exact shape

                # Single diffusion step from one noise level to another
                xs_pred, aux_output = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(conditions, from_noise_levels),
                    #conditions_mask=None,
                    guidance_fn=guidance_fn,
                )

                # If we concatenated context channels, revert the shape
                if self.hparams.cat_context_in_c_dim:
                    # removing last channels
                    xs_pred = xs_pred[:, :, : x_shape[0]]

                # Revert context tokens to their original values
                xs_pred = torch.where(
                    self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
                )
                final_xs_pred = xs_pred

                pbar.update(1)
            # End of chunk loop
            # Overwrite xs_pred with the final chunk results if desired
            xs_pred = final_xs_pred

            # Increment progress bar once per main iteration

        # Finished the main loop
        if return_all:
            # Append final state
            record.append(xs_pred.clone())

        # Stack record outside the loop
        if return_all:
            record = torch.stack(record, dim=0)  # shape: [steps, b, tokens, ...]

        # Remove any padding from the final predictions / record
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            if return_all:
                record = record[..., :-padding, :]

        return xs_pred, record

# algorithm
    def _sample_sequence_in_time_dimension(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        return_all: bool = False,
        number_of_chunks: int = 1,
        guidance_fn: Optional[Callable] = None,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x_shape = self.hparams.x_shape
        number_of_tokens = self.max_tokens * number_of_chunks
        padding = number_of_tokens - length
        # pad context and context_mask
        if context is not None:
            context = torch.cat(
                [
                    context,
                    torch.zeros((batch_size, padding, *x_shape), device=self.device, dtype=context.dtype),
                ],
                dim=1,
            )
        if context_mask is not None:
            context_mask = F.pad(context_mask, (0, padding))


        scheduling_matrix = self._generate_scheduling_matrix(
            #length,
            number_of_tokens,
            0,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        # Prepare the record list if we want all intermediate steps
        record = [] if return_all else None

        # Initial random noise
        xs_pred = torch.randn(
            (batch_size, number_of_chunks * self.max_tokens, *x_shape),
            #(batch_size, length, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise)
        sliding_window_context_length = self.hparams.sliding_context_len # the the overlap between the context and the generated frames
        h = self.max_tokens - sliding_window_context_length
        l = sliding_window_context_length + h

        # Create a single progress bar if none is given
        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels_full = scheduling_matrix[m]     # shape: [b, t]
            to_noise_levels_full = scheduling_matrix[m + 1]   # shape: [b, t]
            number_of_generations = number_of_chunks

            if return_all:
                record.append(xs_pred.clone())

            for i in range(number_of_generations):
                # in sliding window fashion
                idx = self.max_tokens - sliding_window_context_length if i > 0 else self.max_tokens
                from_noise_levels = from_noise_levels_full[:, i*idx:(i+1)*idx]
                to_noise_levels   = to_noise_levels_full[:, i*idx:(i+1)*idx]
                context_in = context[:, i*idx:(i+1)*idx]
                context_mask_in = context_mask[:, i*idx:(i+1)*idx]
                xs_pred_in = xs_pred[:, i*idx:(i+1)*idx]
                conditions_in = conditions[:, i*idx:(i+1)*idx] if conditions is not None else None
                
                # If context is None, create a dummy context + mask, also if we are not in the first iteration
                if context is None:
                    context_in = torch.zeros_like(xs_pred)
                    context_mask_in = torch.zeros(
                        (batch_size, self.max_tokens),
                        dtype=torch.long,
                        device=self.device
                    )
                    # context mask is 1 for sliding_window_context_length frames and 0 for the rest
                context_mask_in = torch.cat(
                    [
                        torch.ones((batch_size, sliding_window_context_length), dtype=torch.long, device=self.device),
                        torch.zeros((batch_size, h), dtype=torch.long, device=self.device),
                    ],
                    dim=1,
                )

                xs_pred_prev_in = xs_pred_in.clone()
                # If we do NOT concatenate context into the diffusion model's channels,
                # we literally replace the context frames in xs_pred with the given context
                if not self.hparams.cat_context_in_c_dim or sliding_window_context_length > 0:
                    xs_pred_in = torch.where(self._extend_x_dim(context_mask_in) >= 1, context_in, xs_pred_in)

                # Make a backup for context tokens so we can revert them after diffusion

                # If we DO concatenate context in c-dim, cat them here
                if self.hparams.cat_context_in_c_dim:
                    xs_pred_in = torch.cat([xs_pred_in, context_in], dim=2)  # depends on your exact shape

                # Single diffusion step from one noise level to another
                xs_pred_in, aux_output = self.diffusion_model.sample_step(
                    xs_pred_in,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(conditions_in, from_noise_levels),
                    #conditions_mask=None,
                    guidance_fn=guidance_fn,
                )

                # If we concatenated context channels, revert the shape
                if self.hparams.cat_context_in_c_dim:
                    # removing last channels
                    xs_pred_in = xs_pred_in[:, :, : x_shape[0]]

                # Revert context tokens to their original values
                xs_pred_in = torch.where(
                    self._extend_x_dim(context_mask_in) == 0, xs_pred_in, xs_pred_prev_in
                )
                xs_pred[:, i*idx:(i+1)*idx] = xs_pred_in

        return xs_pred[:, :length], record


    def _extend_x_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extend the 2D [batch, tokens] mask to match the shape of your x_shape (i.e., [batch, tokens, c, h, w]).
        """
        return rearrange(
            x,
            "... -> ... "
            + "1 " * len(self.hparams.x_shape),  # e.g. " -> ... 1 1 1" if x_shape=[3,64,64]
        )

    # ---------------------------------------------------------------------
    # Example pyramid scheduling (if your "autoregressive" needs it)
    # ---------------------------------------------------------------------
    def _generate_pyramid_scheduling_matrix(self, horizon: int, timesteps: int) -> np.ndarray:
        """
        Example “autoregressive” or “pyramid” scheduling logic:
        Decreasing timesteps on smaller chunks, etc.
        Feel free to replace with your own logic.
        """
        # Just a toy example: each column has a line from timesteps->0,
        # but we skip more steps as we move right, forming a “pyramid” shape.
        matrix = []
        for col in range(horizon):
            steps = np.linspace(timesteps, 0, timesteps - col if (timesteps - col) > 0 else 1)
            matrix_col = np.round(steps).astype(int)
            # pad the top so all columns have the same #rows:
            row_pad = timesteps + 1 - len(matrix_col)
            matrix_col = np.pad(matrix_col, (row_pad, 0), constant_values=-1)
            if len(matrix) == 0:
                matrix = matrix_col[:, None]
            else:
                matrix = np.concatenate([matrix, matrix_col[:, None]], axis=1)
        # Replace negative with 0 at the top:
        matrix = np.where(matrix < 0, 0, matrix)
        return matrix

    

    # ---------------------------------------------------------------------
    # Latent & Normalization Utils
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _run_vae(
        self,
        x: Tensor,
        shape: str,
        vae_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """
        Helper function to run the VAE, either for encoding or decoding.
        - Requires shape to be a permutation of b, t, c, h, w.
        - Reshapes the input tensor to the required shape for the VAE, and reshapes the output back.
            - x: `shape` shape.
            - VideoVAE requires (b, c, t, h, w) shape.
            - ImageVAE requires (b, c, h, w) shape.
        - Split the input tensor into chunks of size cfg.vae.batch_size, to avoid memory errors.
        """
        x = rearrange(x, f"{shape} -> b c t h w")
        batch_size = x.shape[0]
        vae_batch_size = self.cfg.vae.batch_size
        # chunk the input tensor by vae_batch_size
        chunks = torch.chunk(x, (batch_size + vae_batch_size - 1) // vae_batch_size, 0)
        outputs = []
        for chunk in chunks:
            b = chunk.shape[0]
            if not self.is_latent_video_vae:
                chunk = rearrange(chunk, "b c t h w -> (b t) c h w")
            output = vae_fn(chunk)
            if not self.is_latent_video_vae:
                output = rearrange(output, "(b t) c h w -> b c t h w", b=b)
            outputs.append(output)
        return rearrange(torch.cat(outputs, 0), f"b c t h w -> {shape}")

    def _encode(self, x: Tensor, shape: str = "b t c h w") -> Tensor:
        return self._run_vae(x, shape, lambda y: self.vae.encode(2.0 * y - 1.0).sample())

    def _decode(self, latents: Tensor, shape: str = "b t c h w") -> Tensor:
        return self._run_vae(
            latents,
            shape,
            lambda y: (
                self.vae.decode(y, self._n_tokens_to_n_frames(latents.shape[1]))
                if self.is_latent_video_vae
                else self.vae.decode(y)
            )
            * 0.5
            + 0.5,
        )

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    # ---------------------------------------------------------------------
    # Checkpoint Utils
    # ---------------------------------------------------------------------

    def _uncompile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict if self.diffusion_model is compiled, to uncompiled."""
        if self.compile:
            checkpoint["state_dict"] = {
                k.replace("diffusion_model._orig_mod.", "diffusion_model."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _compile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict to the format expected by the compiled model."""
        if self.compile:
            checkpoint["state_dict"] = {
                k.replace("diffusion_model.", "diffusion_model._orig_mod."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return key.startswith("diffusion_model.model") or key.startswith(
            "diffusion_model._orig_mod.model"
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) uncompile the model's state_dict before saving
        self._uncompile_checkpoint(checkpoint)
        # 2. Only save the meaningful keys defined by self._should_include_in_checkpoint
        # by default, only the model's state_dict is saved and metrics & registered buffes (e.g. diffusion schedule) are not discarded
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if not self._should_include_in_checkpoint(key):
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) compile the model's state_dict before loading
        self._compile_checkpoint(checkpoint)
        # 2. (Optionally) swap the state_dict of the model with the EMA weights for inference
        super().on_load_checkpoint(checkpoint)
        # 3. (Optionally) reset the optimizer states - for fresh finetuning or resuming training
        if self.cfg.checkpoint.reset_optimizer:
            checkpoint["optimizer_states"] = []

        # 4. Rewrite the state_dict of the checkpoint, only leaving meaningful keys
        # defined by self._should_include_in_checkpoint
        # also print out warnings when the checkpoint does not exactly match the expected format

        new_state_dict = {}
        for key, value in self.state_dict().items():
            if (
                self._should_include_in_checkpoint(key)
                and key in checkpoint["state_dict"]
            ):
                new_state_dict[key] = checkpoint["state_dict"][key]
            else:
                new_state_dict[key] = value

        # print keys that are ignored from the checkpoint
        ignored_keys = [
            key
            for key in checkpoint["state_dict"].keys()
            if not self._should_include_in_checkpoint(key)
        ]
        if ignored_keys:
            rank_zero_print(
                cyan("The following keys are ignored from the checkpoint:"),
                ignored_keys,
            )
        # print keys that are not found in the checkpoint
        missing_keys = [
            key
            for key in self.state_dict().keys()
            if self._should_include_in_checkpoint(key)
            and key not in checkpoint["state_dict"]
        ]
        if missing_keys:
            rank_zero_print(
                cyan("The following keys are not found in the checkpoint:"),
                missing_keys,
            )
            if self.cfg.checkpoint.strict:
                raise ValueError(
                    "Thus, the checkpoint cannot be loaded. To ignore this error, turn off strict checkpoint loading by setting `algorithm.checkpoint.strict=False`."
                )
            else:
                rank_zero_print(
                    cyan(
                        "Strict checkpoint loading is turned off, so using the initialized value for the missing keys."
                    )
                )
        checkpoint["state_dict"] = new_state_dict

    def _load_ema_weights_to_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        if (
            checkpoint.get("pretrained_ema", False)
            and len(checkpoint["optimizer_states"]) == 0
        ):
            # NOTE: for lightweight EMA-only ckpts for releasing pretrained models,
            # we already have EMA weights in the state_dict
            return
        ema_weights = checkpoint["optimizer_states"][0]["ema"]
        parameter_keys = [
            "diffusion_model." + k for k, _ in self.diffusion_model.named_parameters()
        ]
        assert len(parameter_keys) == len(
            ema_weights
        ), "Number of original weights and EMA weights do not match."
        for key, weight in zip(parameter_keys, ema_weights):
            checkpoint["state_dict"][key] = weight



###############################################################################
# Example Usage (Stand-Alone)
###############################################################################

if __name__ == "__main__":
    # Create a dummy 3D UNet model.
    # model = DummyUNet3D(in_channels=1, out_channels=1, base_channels=32)

    from src.models.components.dit import DiT_S_8
    from src.models.components.autoencoder.simple_autoencoder import AutoEncoder

    autoencoder = AutoEncoder(in_channels=3, latent_dim=4, hidden_size=64, downsampling_factor=4)
    model = DiT_S_8(
        in_channels=8, input_size=16, out_channels=4, num_classes=10, autoencoder=autoencoder
    )
    optimizer = torch.optim.Adam

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="v_prediction"
    )

    # Instantiate the diffusion trainer.
    diffusion_trainer = DiffusionModelTrainer(
        model=model,
        optimizer=optimizer,
        num_train_timesteps=1000,
        scheduler=scheduler,
        compile=False,
        lr=1e-4,
        num_inference_steps=20,
        num_gen_steps=15,
    )
    from src.data.memory_maze import MemoryMazeDataset

    dataset = MemoryMazeDataset(
        base_folder="/data/cvg/sebastian/memory_maze/memory-maze-9x9/eval",
        num_frames=2,
        transform=None,
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, num_workers=8, shuffle=True)
    batch = next(iter(loader))

    # loss = diffusion_trainer.training_step(batch, batch_idx=0)
    val = diffusion_trainer.validation_step(batch, batch_idx=0)
    from lightning import Trainer

    trainer = Trainer(devices=1, num_sanity_val_steps=1, val_check_interval=1)
    trainer.fit(
        diffusion_trainer,
        loader,
    )

    # Create a dummy video batch.
    # For example, batch_size=16, channels=1, frames=16, height=64, width=64.
    dummy_batch = {
        "video": torch.rand(16, 3, 64, 64) * 2 - 1,
        "cond": {"img": torch.rand(16, 3, 64, 64) * 2 - 1, "y": torch.randint(0, 10, (16,))},
    }  # values in [-1, 1]

    # Simulate one training step.
    print(0)
