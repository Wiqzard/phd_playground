from typing import Any, Dict, Optional, Union, Sequence, Tuple, Callable, Literal

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
#from src.models.components.moe_lora import inject_lora, disable_all_adapters, enable_all_adapters, set_lora_trainability, reset_all_lora_parameters, get_lora_adapter_parameters, get_lora_adapter_parameters, set_global_trainability , get_global_parameters
from diffusers import DDPMScheduler
from src.models.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from src.models.common import BaseLightningTrainer

from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.logging_utils import log_video
from utils.print_utils import cyan
from utils.torch_utils import freeze_model







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

#LightningModule
class DiffusionModelTrainer(BaseLightningTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        scheduler: Optional[Any] = None,
        compile: bool = False,
        num_inference_steps: int = 50,  # fewer steps for faster inference
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
        self.save_hyperparameters(ignore=("model", "optimizer", "lr_scheduler", "scheduler"))
        self.model = model
        self.optimizer= optimizer
        self.lr_scheduler = lr_scheduler
        self.compile_model = compile

        # Initialize the DDPM scheduler for training
        if scheduler is not None:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule
            )
        else:
            self.scheduler = scheduler
        self.timesteps = num_train_timesteps


        self.temporal_downsampling_factor = latent_downsampling_factor[0]

        
        self.tasks = [
            task
            for task in self.hparams.tasks
        ]

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
        if self.hparams.ckpt_path:
            self.load_state_dict(torch.load(self.hparams.ckpt_path, weights_only=False)["state_dict"])

        if self.hparams.lora_finetune:
            for module in self.model.modules():
                for param in module.parameters():
                    param.requires_grad = False

            #for module in self.model.memory_layers.modules():
            #    for param in module.parameters():
            #        param.requires_grad = True

            from peft import LoraConfig, TaskType, get_peft_model
            # Create a LoRA configuration
            lora_config = LoraConfig(
                r=8,                  # rank
                lora_alpha=32,        # alpha scaling
                lora_dropout=0.00,    # dropout, can be 0.0 as well
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

        self.model = torch.compile(
            self.model,
            disable=not self.compile_model,
        )
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
                        #and not self.hparams.is_full_sequence
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
            xs = (
                self._encode(batch["videos"])
                if self.is_latent_online
                else batch["latents"]
            )
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
            self.logging.n_metrics_frames is not None
        ):  # only consider the first n_metrics_frames for evaluation
            all_videos = {
                k: v[:, : self.logging.n_metrics_frames] for k, v in all_videos.items()
            }

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
            if self.logging.n_metrics_frames is not None:
                context_mask = context_mask[: self.logging.n_metrics_frames]
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
                    else torch.tensor(
                        [0, n_frames - 1], device=self.device, dtype=torch.long
                    )
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
                noise_pred, memory_states, suprises = self.model.forward(x_noisy_i, timesteps, cond=y, memory_states=memory_states, return_memory=True)
            else:
                #noise_pred = self.model.forward(x_noisy_i, timesteps, cond=y, cache_params=memory_states, use_cache=False)
                noise_pred, memory_states = self.model.forward(x_noisy_i, timesteps, cond=y, cache_params=memory_states, use_cache=True, run=i)
                suprises = torch.zeros_like(noise_pred)
            # b, t, c, h, w
            x_pred = self.scheduler.get_velocity(sample=noise_pred, noise=x_noisy_i[:, : ,:3], timesteps=timesteps)
            video.append(x_pred)
        
        x_pred = torch.cat(video, dim=1)
        #x_pred = x_pred.permute(0, 2, 1, 3, 4)
        if not loss:
            return x_pred, timesteps

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        
        while len(weights.shape) < len(x_pred.shape):
            weights = weights.unsqueeze(-1)
        diffusion_loss = self.loss(x_pred.permute(0, 2, 1, 3, 4), x, weights)  # adjust as needed
        return  x_pred, diffusion_loss, None #, #suprises



    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, namespace: str ="training") -> STEP_OUTPUT:
        xs, conditions, masks, *_ = batch
        
        # ----- Diffusion branch: Compute diffusion loss -----
        timesteps = None
        xs_pred, diffusion_loss, aux_loss = self.forward(xs, conditions=self._process_conditions(conditions), loss=True, timesteps=timesteps)
        #surprises = surprises[-1].mean()
        
        # ----- Combine losses -----
        # You might want to weight the autoencoder loss differently (using lambda_ae)
        ## Update metrics and log losses
        #self.train_loss.update(total_loss)
        #self.recon_loss.update(recon_loss)
        #self.diffusion_loss.update(diffusion_loss)
        #self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        #self.log("train/surprises", surprises, on_step=True, on_epoch=False, prog_bar=True)
        #self.log("train/diffusion_loss", diffusion_loss, on_step=True, on_epoch=False, prog_bar=True)

        if batch_idx % self.trainer.log_every_n_steps == 0: # cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss",
                diffusion_loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )
            if aux_loss is not None:
                for key, value in aux_loss.items():
                    self.log(
                        f"{namespace}/{key}",
                        value,
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))
        output_dict = {
            "loss": diffusion_loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }
        return output_dict 

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
        if not (
            self.trainer.sanity_checking and not self.hparams.log_sanity_generation
        ):
            all_videos = self._sample_all_videos(batch, batch_idx, namespace)
            self._update_metrics(all_videos)
            self._log_videos(all_videos, namespace)


    def on_validation_epoch_start(self) -> None:
        if self.hparams.log_deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.global_rank
                + self.trainer.world_size * self.hparams.log_deterministic
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

        #xs = xs[:, : self.max_tokens]
        #if conditions is not None:
            #conditions = conditions[:, : self.max_tokens]
        #masks = masks[:, : self.max_tokens]
        #if gt_videos is not None:
        #    gt_videos = gt_videos[:, : self.max_frames]
#
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
        #loss = output["loss"]
        #self.log(f"{namespace}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)


    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation"
    ) -> Optional[Dict[str, torch.Tensor]]:
        xs, conditions, *_, gt_videos = batch
        all_videos: Dict[str, torch.Tensor] = {"gt": xs}

        for task in self.tasks:
            sample_fn = (
                self._predict_videos
                if task == "prediction"
                else self._interpolate_videos
            )
            all_videos[task] = sample_fn(xs, conditions=conditions)

        # remove None values
        all_videos = {k: v for k, v in all_videos.items() if v is not None}
        # unnormalize/detach the videos
        all_videos = {k: self._unnormalize_x(v).detach() for k, v in all_videos.items()}
        # decode latents if using latents
        if self.hparams.is_latent_diffusion:
            all_videos = {
                k: self._decode(v) if k != "gt" else gt_videos
                for k, v in all_videos.items()
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
        Predict the videos with the given context, using sliding window rollouts if necessary.
        Optionally, if cfg.tasks.prediction.keyframe_density < 1, predict the keyframes first,
        then interpolate the missing frames.
        """
        xs_pred = xs.clone()  # b, t, c, h, w

        # -- Example keyframe logic (if needed)
        # density = self.cfg.tasks.prediction.keyframe_density or 1
        # if density > 1:
        #     raise ValueError("tasks.prediction.keyframe_density must be <= 1")
        # keyframe_indices = (
        #     torch.linspace(0, xs_pred.shape[1] - 1, round(density * xs_pred.shape[1]))
        #     .round()
        #     .long()
        # )
        # # force context frames to be keyframes
        # keyframe_indices = torch.cat(
        #     [torch.arange(self.n_context_tokens), keyframe_indices]
        # ).unique()
        # key_conditions = conditions[:, keyframe_indices] if conditions is not None else None
        # # 1. Predict keyframes
        # xs_pred_key, *_ = self._predict_sequence(
        #     xs_pred[:, : self.n_context_tokens],
        #     length=len(keyframe_indices),
        #     conditions=key_conditions,
        #     reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
        #     sliding_context_len=self.cfg.tasks.prediction.sliding_context_len
        #         or self.max_tokens // 2,
        # )
        # xs_pred[:, keyframe_indices] = xs_pred_key

        # -- If you skip the keyframe logic entirely, just do one pass:
        xs_pred, *_ = self._predict_sequence(
            context=xs_pred[:, : self.n_context_tokens],
            length=xs_pred.shape[1],
            conditions=conditions,
            reconstruction_guidance=self.hparams.reconstruction_guidance,
            sliding_context_len=self.hparams.sliding_context_len
                or self.max_tokens // 2,
        )

        # 2. (Optional) Interpolate the intermediate frames if you used keyframes
        # (If not using keyframes, you can remove this chunk.)
        # if len(keyframe_indices) < xs_pred.shape[1]:
        #     context_mask = torch.zeros(xs_pred.shape[:2], device=self.device).bool()
        #     context_mask[:, keyframe_indices] = True
        #     xs_pred = self._interpolate_videos(
        #         context=xs_pred,
        #         context_mask=context_mask,
        #         conditions=conditions,
        #     )

        return xs_pred

    # ---------------------------------------------------------------------
    # Sampling Utils
    # ---------------------------------------------------------------------

    def _generate_scheduling_matrix(
        self,
        horizon: int,
        padding: int = 0,
    ):
        match self.hparams.scheduling_matrix:
            case "full_sequence":
                scheduling_matrix = np.arange(self.hparams.sampling_timesteps, -1, -1)[
                    :, None
                ].repeat(horizon, axis=1)
            case "autoregressive":
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    horizon, self.hparams.sampling_timesteps
                )

        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

        #scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(
        #    scheduling_matrix
        #)

        # paded entries are labeled as pure noise
        scheduling_matrix = F.pad(
            scheduling_matrix, (0, padding, 0, 0), value=self.timesteps - 1
        )

        return scheduling_matrix



    def _predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        sliding_context_len: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence given context tokens at the beginning, possibly using sliding-window
        if length > self.max_tokens. This version removes any HistoryGuidance usage.
        """
        if length is None:
            length = self.max_tokens
        if sliding_context_len is None:
            if self.max_tokens < length:
                raise ValueError(
                    "when length > max_tokens, sliding_context_len must be specified."
                )
            else:
                sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        batch_size, gt_len, *_ = context.shape

        if sliding_context_len < gt_len:
            raise ValueError(
                "sliding_context_len is expected to be >= length of initial context."
            )

        chunk_size = self.hparams.chunk_size if self.hparams.use_causal_mask else self.max_tokens

        curr_token = gt_len
        xs_pred = context
        record = None
        pbar = tqdm(
            total=self.hparams.sampling_timesteps
            * (
                1
                + (length - sliding_context_len - 1)
                // (self.max_tokens - sliding_context_len)
            ),
            initial=0,
            desc="Predicting (vanilla diffusion)",
            leave=False,
        )

        
        # put everything here


        while curr_token < length:
            # If storing all steps, forbid sliding windows for simplicity
            if return_all:
                raise ValueError("return_all is not supported with sliding window.")
            # actual context depends on whether it's during sliding window or not
            c = min(sliding_context_len, curr_token)
            h = min(length - curr_token, self.max_tokens - c)
            if chunk_size > 0:
                h = min(h, chunk_size)
            l = c + h

            # Prepare the next chunk’s context
            pad = torch.zeros((batch_size, h, *self.hparams.x_shape), device=self.device)
            context_mask = torch.cat([
                # Mark the existing c tokens (some might be GT, some generated)
                torch.ones((batch_size, c), dtype=torch.long, device=self.device),
                # Mark the h new tokens as to-be-generated (0)
                torch.zeros((batch_size, h), dtype=torch.long, device=self.device),
            ], dim=1)

            # If you want to distinguish GT vs generated context, do so here
            # but for “vanilla” sampling, we’ll just treat them as context=1

            context_chunk = torch.cat([
                xs_pred[:, -c:] if c > 0 else torch.empty(0),
                pad
            ], dim=1)
            # Run a standard diffusion sampling on this chunk
            new_pred, _ = self._sample_sequence(
                batch_size=batch_size,
                length=l,
                context=context_chunk,
                context_mask=context_mask,
                conditions=conditions[:, curr_token - c : curr_token - c + l]
                    if (conditions is not None and self.hparams.use_causal_mask)
                    else conditions,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                return_all=False,
                pbar=pbar,
            )
            # Take only the newly generated portion
            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], dim=1)
            curr_token = xs_pred.shape[1]

        pbar.close()
        return xs_pred, record

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        A simpler (vanilla) sampling method for up to self.max_tokens in length. 
        We remove any references to HistoryGuidance or fancy scheduling. 
        We just do a standard diffusion reverse pass from self.sampling_timesteps down to 1.
        """
        x_shape = self.hparams.x_shape

        if length is None:
            length = self.max_tokens if context is None else context.shape[1]
        if length > self.max_tokens:
            raise ValueError(
                f"length is expected to <= self.max_tokens, but got {length}."
            )

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")
        else:
            # If no context is given, create dummy zero context
            context = torch.zeros(
                (batch_size, length, *x_shape), device=self.device
            )
            context_mask = torch.zeros(
                (batch_size, length), dtype=torch.long, device=self.device
            )

        # For non-causal, conditions should always have shape [b, self.max_tokens, ...]
        # For causal, conditions must match the length dimension
        if conditions is not None:
            if self.hparams.use_causal_mask and conditions.shape[1] != length:
                raise ValueError(
                    f"For causal models, conditions length must be {length}, "
                    f"but got {conditions.shape[1]}."
                )
            elif not self.hparams.use_causal_mask and conditions.shape[1] != self.max_tokens:
                raise ValueError(
                    f"For noncausal models, conditions length must be {self.max_tokens}, "
                    f"but got {conditions.shape[1]}."
                )

        horizon = length if self.hparams.use_causal_mask else self.max_tokens
        padding = horizon - length

        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            horizon - padding,
            padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
        # fill context tokens' noise levels as -1 in scheduling matrix
        #if not self.is_full_sequence:
        if False:
            scheduling_matrix = torch.where(
                context_mask[None] >= 1, -1, scheduling_matrix
            )

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling: ",
                leave=False,
            )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1),
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            #with history_guidance(context_mask) as history_guidance_manager:
            if True:
                xs_pred_in = torch.cat([context, xs_pred], dim=2) # b, t, c, h, w
                model_output, _ = self.model(xs_pred_in, from_noise_levels,  cond=conditions, use_cache=True)
                


                nfe = history_guidance_manager.nfe
                pbar.set_postfix(NFE=nfe)

                xs_pred, from_noise_levels, to_noise_levels, conditions_mask = (
                    history_guidance_manager.prepare(
                        xs_pred,
                        from_noise_levels,
                        to_noise_levels,
                        replacement_fn=self.diffusion_model.q_sample,
                        replacement_only=self.is_full_sequence,
                    )
                )

                # update xs_pred by DDIM or DDPM sampling
                xs_pred = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(
                        (
                            repeat(
                                conditions,
                                "b ... -> (b nfe) ...",
                                nfe=nfe,
                            ).clone()
                            if conditions is not None
                            else None
                        ),
                        from_noise_levels,
                    ),
                    conditions_mask,
                    guidance_fn=composed_guidance_fn,
                )

                xs_pred = history_guidance_manager.compose(xs_pred)

            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(
                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
            )
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record










        # Initialize from random noise (vanilla approach)
        xs_pred = torch.randn(
            (batch_size, length, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise)

        # Where context_mask >= 1, we clamp to provided context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        record = [] if return_all else None

        # We do a typical reverse-diffusion loop: self.sampling_timesteps steps.
        # For each step, we call self.diffusion_model.sample_step
        # Possibly with an optional reconstruction guidance or other guidance function.
        timesteps = list(range(self.hparams.sampling_timesteps))
        for t in timesteps:
            xs_pred_prev = xs_pred.clone()
            # Single step of your diffusion model’s reverse pass
            #xs_pred = 


            xs_pred = self.diffusion_model.sample_step(
                x=xs_pred,
                # In a standard diffusion, you pass integer "t" or "t -> t_next" (DDIM, etc.).
                from_noise_level=t,
                to_noise_level=(t + 1),
                conditions=self._process_conditions(conditions, t),
                conditions_mask=None,
                guidance_fn=curr_guidance_fn,
            )

            # Re-clamp the known context frames so we do not overwrite them
            xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

            if return_all:
                record.append(xs_pred.clone())
            if pbar is not None:
                pbar.update(1)

        if return_all:
            record = torch.stack(record, dim=0)

        return xs_pred, record


    def _extend_x_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Extend the tensor by adding dimensions at the end to match x_stacked_shape."""
        return rearrange(x, "... -> ..." + " 1" * len(self.hparams.x_shape))



#    def _sample_all_videos(
#        self, batch, batch_idx, namespace="validation"
#    ) -> Optional[Dict[str, Tensor]]:
#        xs, conditions, *_, gt_videos = batch
#        all_videos: Dict[str, Tensor] = {"gt": xs}
#
#        for task in self.tasks:
#            sample_fn = (
#                self._predict_videos
#                if task == "prediction"
#                else self._interpolate_videos
#            )
#            all_videos[task] = sample_fn(xs, conditions=conditions)
#
#        # remove None values
#        all_videos = {k: v for k, v in all_videos.items() if v is not None}
#        # rearrange/unnormalize/detach the videos
#        all_videos = {k: self._unnormalize_x(v).detach() for k, v in all_videos.items()}
#        # decode latents if using latents
#        if self.hparams.is_latent_diffusion:
#            all_videos = {
#                k: self._decode(v) if k != "gt" else gt_videos
#                for k, v in all_videos.items()
#            }
#
#        # # replace the context frames of video predictions with the ground truth
#        if "prediction" in all_videos:
#            all_videos["prediction"][:, : self.n_context_frames] = all_videos["gt"][
#                :, : self.n_context_frames
#            ]
#        return all_videos
#    
#    def _predict_videos(
#        self, xs: Tensor, conditions: Optional[Tensor] = None
#    ) -> Tensor:
#        """
#        Predict the videos with the given context, using sliding window rollouts if necessary.
#        Optionally, if cfg.tasks.prediction.keyframe_density < 1, predict the keyframes first,
#        then interpolate the missing intermediate frames.
#        """
#        xs_pred = xs.clone() # b, t, c, h, w
#
#        #history_guidance = HistoryGuidance.from_config(
#        #    config=self.cfg.tasks.prediction.history_guidance,
#        #    timesteps=self.timesteps,
#        #)
#
#        #density = self.cfg.tasks.prediction.keyframe_density or 1
#        #if density > 1:
#        #    raise ValueError("tasks.prediction.keyframe_density must be <= 1")
#        #keyframe_indices = (
#        #    torch.linspace(0, xs_pred.shape[1] - 1, round(density * xs_pred.shape[1]))
#        #    .round()
#        #    .long()
#        #)
#        #keyframe_indices = torch.cat(
#        #    [torch.arange(self.n_context_tokens), keyframe_indices]
#        #).unique()  # context frames are always keyframes
#        #key_conditions = (
#        #    conditions[:, keyframe_indices] if conditions is not None else None
#        #)
#
#        # 1. Predict the keyframes
#        xs_pred_key, *_ = self._predict_sequence(
#            xs_pred[:, : self.n_context_tokens],
#            length=len(keyframe_indices),
#            conditions=key_conditions,
#            history_guidance=history_guidance,
#            reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
#            sliding_context_len=self.cfg.tasks.prediction.sliding_context_len
#            or self.max_tokens // 2,
#        )
#        xs_pred[:, keyframe_indices] = xs_pred_key
#        # if is_rank_zero: # uncomment to visualize history guidance
#        #     history_guidance.log(logger=self.logger)
#
#        # 2. (Optional) Interpolate the intermediate frames
#        if len(keyframe_indices) < xs_pred.shape[1]:
#            context_mask = torch.zeros(xs_pred.shape[:2], device=self.device).bool()
#            context_mask[:, keyframe_indices] = True
#            xs_pred = self._interpolate_videos(
#                context=xs_pred,
#                context_mask=context_mask,
#                conditions=conditions,
#            )
#
#        return xs_pred
#
#    def _predict_sequence(
#        self,
#        context: torch.Tensor,
#        length: Optional[int] = None,
#        conditions: Optional[torch.Tensor] = None,
#        guidance_fn: Optional[Callable] = None,
#        reconstruction_guidance: float = 0.0,
#        history_guidance = None,
#        #history_guidance: Optional[HistoryGuidance] = None,
#        sliding_context_len: Optional[int] = None,
#        return_all: bool = False,
#    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#        """
#        Predict a sequence given context tokens at the beginning, using sliding window if necessary.
#        Args
#        ----
#        context: torch.Tensor, Shape (batch_size, init_context_len, *self.x_shape)
#            Initial context tokens to condition on
#        length: Optional[int]
#            Desired number of tokens in sampled sequence.
#            If None, fall back to to self.max_tokens, and
#            If bigger than self.max_tokens, sliding window sampling will be used.
#        conditions: Optional[torch.Tensor], Shape (batch_size, conditions_len, ...)
#            Unprocessed external conditions for sampling, e.g. action or text, optional
#        guidance_fn: Optional[Callable]
#            Guidance function for sampling
#        reconstruction_guidance: float
#            Scale of reconstruction guidance (from Video Diffusion Models Ho. et al.)
#        history_guidance: Optional[HistoryGuidance]
#            History guidance object that handles compositional generation
#        sliding_context_len: Optional[int]
#            Max context length when using sliding window. -1 to use max_tokens - 1.
#            Has no influence when length <= self.max_tokens as no sliding window is needed.
#        return_all: bool
#            Whether to return all steps of the sampling process.
#
#        Returns
#        -------
#        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
#            Predicted sequence with both context and generated tokens
#        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
#            Record of all steps of the sampling process
#        """
#        if length is None:
#            length = self.max_tokens
#        if sliding_context_len is None:
#            if self.max_tokens < length:
#                raise ValueError(
#                    "when length > max_tokens, sliding_context_len must be specified."
#                )
#            else:
#                sliding_context_len = self.max_tokens - 1
#        if sliding_context_len == -1:
#            sliding_context_len = self.max_tokens - 1
#
#        batch_size, gt_len, *_ = context.shape
#
#        if sliding_context_len < gt_len:
#            raise ValueError(
#                "sliding_context_len is expected to be >= length of initial context,"
#                f"got {sliding_context_len}. If you are trying to use max context, "
#                "consider specifying sliding_context_len=-1."
#            )
#
#        chunk_size = self.chunk_size if self.use_causal_mask else self.max_tokens
#
#        curr_token = gt_len
#        xs_pred = context
#        x_shape = self.x_shape
#        record = None
#        pbar = tqdm(
#            total=self.sampling_timesteps
#            * (
#                1
#                + (length - sliding_context_len - 1)
#                // (self.max_tokens - sliding_context_len)
#            ),
#            initial=0,
#            desc="Predicting with DFoT",
#            leave=False,
#        )
#        while curr_token < length:
#            if record is not None:
#                raise ValueError("return_all is not supported if using sliding window.")
#            # actual context depends on whether it's during sliding window or not
#            # corner case at the beginning
#            c = min(sliding_context_len, curr_token)
#            # try biggest prediction chunk size
#            h = min(length - curr_token, self.max_tokens - c)
#            # chunk_size caps how many future tokens are diffused at once to save compute for causal model
#            h = min(h, chunk_size) if chunk_size > 0 else h
#            l = c + h
#            pad = torch.zeros((batch_size, h, *x_shape))
#            # context is last c tokens out of the sequence of generated/gt tokens
#            # pad to length that's required by _sample_sequence
#            context = torch.cat([xs_pred[:, -c:], pad.to(self.device)], 1)
#            # calculate number of model generated tokens (not GT context tokens)
#            generated_len = curr_token - max(curr_token - c, gt_len)
#            # make context mask
#            context_mask = torch.ones((batch_size, c), dtype=torch.long)
#            if generated_len > 0:
#                context_mask[:, -generated_len:] = 2
#            pad = torch.zeros((batch_size, h), dtype=torch.long)
#            context_mask = torch.cat([context_mask, pad.long()], 1).to(context.device)
#
#            cond_len = l if self.use_causal_mask else self.max_tokens
#            cond_slice = None
#            if conditions is not None:
#                cond_slice = conditions[:, curr_token - c : curr_token - c + cond_len]
#
#            new_pred, record = self._sample_sequence(
#                batch_size,
#                length=l,
#                context=context,
#                context_mask=context_mask,
#                conditions=cond_slice,
#                guidance_fn=guidance_fn,
#                reconstruction_guidance=reconstruction_guidance,
#                history_guidance=history_guidance,
#                return_all=return_all,
#                pbar=pbar,
#            )
#            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
#            curr_token = xs_pred.shape[1]
#        pbar.close()
#        return xs_pred, record
#
#    def _extend_x_dim(self, x: torch.Tensor) -> torch.Tensor:
#        """Extend the tensor by adding dimensions at the end to match x_stacked_shape."""
#        return rearrange(x, "... -> ..." + " 1" * len(self.x_shape))
#
#    def _sample_sequence(
#        self,
#        batch_size: int,
#        length: Optional[int] = None,
#        context: Optional[torch.Tensor] = None,
#        context_mask: Optional[torch.Tensor] = None,
#        conditions: Optional[torch.Tensor] = None,
#        guidance_fn: Optional[Callable] = None,
#        reconstruction_guidance: float = 0.0,
#        #history_guidance: Optional[HistoryGuidance] = None,
#        history_guidance = None,
#        return_all: bool = False,
#        pbar: Optional[tqdm] = None,
#    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#        """
#        The unified sampling method, with length up to maximum token size.
#        context of length can be provided along with a mask to achieve conditioning.
#
#        Args
#        ----
#        batch_size: int
#            Batch size of the sampling process
#        length: Optional[int]
#            Number of frames in sampled sequence
#            If None, fall back to length of context, and then fall back to `self.max_tokens`
#        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
#            Context tokens to condition on. Assumed to be same across batch.
#            Tokens that are specified as context by `context_mask` will be used for conditioning,
#            and the rest will be discarded.
#        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
#            Mask for context
#            0 = To be generated, 1 = Ground truth context, 2 = Generated context
#            Some sampling logic may discriminate between ground truth and generated context.
#        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.max_tokens (noncausal), ...)
#            Unprocessed external conditions for sampling
#        guidance_fn: Optional[Callable]
#            Guidance function for sampling
#        history_guidance: Optional[HistoryGuidance]
#            History guidance object that handles compositional generation
#        return_all: bool
#            Whether to return all steps of the sampling process
#        Returns
#        -------
#        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
#            Complete sequence containing context and generated tokens
#        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
#            All recorded intermediate results during the sampling process
#        """
#        x_shape = self.x_shape
#
#        if length is None:
#            length = self.max_tokens if context is None else context.shape[1]
#        if length > self.max_tokens:
#            raise ValueError(
#                f"length is expected to <={self.max_tokens}, got {length}."
#            )
#
#        if context is not None:
#            if context_mask is None:
#                raise ValueError("context_mask must be provided if context is given.")
#            if context.shape[0] != batch_size:
#                raise ValueError(
#                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
#                )
#            if context.shape[1] != length:
#                raise ValueError(
#                    f"context length is expected to be {length} but got {context.shape[1]}."
#                )
#            if tuple(context.shape[2:]) != tuple(x_shape):
#                raise ValueError(
#                    f"context shape not compatible with x_stacked_shape {x_shape}."
#                )
#
#        if context_mask is not None:
#            if context is None:
#                raise ValueError("context must be provided if context_mask is given. ")
#            if context.shape[:2] != context_mask.shape:
#                raise ValueError("context and context_mask must have the same shape.")
#
#        if conditions is not None:
#            if self.use_causal_mask and conditions.shape[1] != length:
#                raise ValueError(
#                    f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
#                )
#            elif not self.use_causal_mask and conditions.shape[1] != self.max_tokens:
#                raise ValueError(
#                    f"for noncausal models, conditions length is expected to be {self.max_tokens}, got {conditions.shape[1]}."
#                )
#
#        horizon = length if self.use_causal_mask else self.max_tokens
#        padding = horizon - length
#        # create initial xs_pred with noise
#        xs_pred = torch.randn(
#            (batch_size, horizon, *x_shape),
#            device=self.device,
#            generator=self.generator,
#        )
#        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)
#
#        if context is None:
#            # create empty context and zero context mask
#            context = torch.zeros_like(xs_pred)
#            context_mask = torch.zeros_like(
#                (batch_size, horizon), dtype=torch.long, device=self.device
#            )
#        elif padding > 0:
#            # pad context and context mask to reach horizon
#            context_pad = torch.zeros(
#                (batch_size, padding, *x_shape), device=self.device
#            )
#            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
#            context_mask_pad = -torch.ones(
#                (batch_size, padding), dtype=torch.long, device=self.device
#            )
#            context = torch.cat([context, context_pad], 1)
#            context_mask = torch.cat([context_mask, context_mask_pad], 1)
#
#        if history_guidance is None:
#            # by default, use conditional sampling
#            history_guidance = HistoryGuidance.conditional(
#                timesteps=self.timesteps,
#            )
#
#        # replace xs_pred's context frames with context
#        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)
#
#        # generate scheduling matrix
#        scheduling_matrix = self._generate_scheduling_matrix(
#            horizon - padding,
#            padding,
#        )
#        scheduling_matrix = scheduling_matrix.to(self.device)
#        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
#        # fill context tokens' noise levels as -1 in scheduling matrix
#        if not self.is_full_sequence:
#            scheduling_matrix = torch.where(
#                context_mask[None] >= 1, -1, scheduling_matrix
#            )
#
#        # prune scheduling matrix to remove identical adjacent rows
#        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
#        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
#        scheduling_matrix = scheduling_matrix[skip:]
#
#        record = [] if return_all else None
#
#        if pbar is None:
#            pbar = tqdm(
#                total=scheduling_matrix.shape[0] - 1,
#                initial=0,
#                desc="Sampling with DFoT",
#                leave=False,
#            )
#
#        for m in range(scheduling_matrix.shape[0] - 1):
#            from_noise_levels = scheduling_matrix[m]
#            to_noise_levels = scheduling_matrix[m + 1]
#
#            # update context mask by changing 0 -> 2 for fully generated tokens
#            context_mask = torch.where(
#                torch.logical_and(context_mask == 0, from_noise_levels == -1),
#                2,
#                context_mask,
#            )
#
#            # create a backup with all context tokens unmodified
#            xs_pred_prev = xs_pred.clone()
#            if return_all:
#                record.append(xs_pred.clone())
#
#            conditions_mask = None
#            with history_guidance(context_mask) as history_guidance_manager:
#                nfe = history_guidance_manager.nfe
#                pbar.set_postfix(NFE=nfe)
#                xs_pred, from_noise_levels, to_noise_levels, conditions_mask = (
#                    history_guidance_manager.prepare(
#                        xs_pred,
#                        from_noise_levels,
#                        to_noise_levels,
#                        replacement_fn=self.diffusion_model.q_sample,
#                        replacement_only=self.is_full_sequence,
#                    )
#                )
#
#                if reconstruction_guidance > 0:
#
#                    def composed_guidance_fn(
#                        xk: torch.Tensor,
#                        pred_x0: torch.Tensor,
#                        alpha_cumprod: torch.Tensor,
#                    ) -> torch.Tensor:
#                        loss = (
#                            F.mse_loss(pred_x0, context, reduction="none")
#                            * alpha_cumprod.sqrt()
#                        )
#                        _context_mask = rearrange(
#                            context_mask.bool(),
#                            "b t -> b t" + " 1" * len(x_shape),
#                        )
#                        # scale inversely proportional to the number of context frames
#                        loss = torch.sum(
#                            loss
#                            * _context_mask
#                            / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
#                        )
#                        likelihood = -reconstruction_guidance * 0.5 * loss
#                        return likelihood
#
#                else:
#                    composed_guidance_fn = guidance_fn
#
#                # update xs_pred by DDIM or DDPM sampling
#                xs_pred = self.diffusion_model.sample_step(
#                    xs_pred,
#                    from_noise_levels,
#                    to_noise_levels,
#                    self._process_conditions(
#                        (
#                            repeat(
#                                conditions,
#                                "b ... -> (b nfe) ...",
#                                nfe=nfe,
#                            ).clone()
#                            if conditions is not None
#                            else None
#                        ),
#                        from_noise_levels,
#                    ),
#                    conditions_mask,
#                    guidance_fn=composed_guidance_fn,
#                )
#
#                xs_pred = history_guidance_manager.compose(xs_pred)
#
#            # only replace the tokens being generated (revert context tokens)
#            xs_pred = torch.where(
#                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
#            )
#            pbar.update(1)
#
#        if return_all:
#            record.append(xs_pred.clone())
#            record = torch.stack(record)
#        if padding > 0:
#            xs_pred = xs_pred[:, :-padding]
#            record = record[:, :, :-padding] if return_all else None
#
#        return xs_pred, record




    @torch.no_grad()
    def validation_step2(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        # -------------------------
        # 1) Compute the validation loss on every batch
        # -------------------------
        x = batch["videos"].to(self.device)  # shape: (B, T, C, H, W)
        x = x * 2 - 1
        x_gt = x.clone()  # keep a copy

        # Diffusion loss (or any other training-like loss)
        _, diffusion_loss, _ = self.forward(x_gt, loss=True)
        
        # Update your running metric and log once per step or once per epoch
        self.val_loss.update(diffusion_loss)
        self.log("val/loss", diffusion_loss, on_step=False, on_epoch=True, prog_bar=True)

        # -------------------------
        # 2) Generate a sample video on the first batch
        # -------------------------

        if batch_idx == 0:
            self._generate_and_log_video(x_gt)  # e.g. put your generation code into a helper
    
    def on_validation_end(self):
        avg_loss = self.val_loss.compute()
        #self.log("val/loss", avg_loss, prog_bar=True)
        self.logger.experiment.log({"val/loss": avg_loss, "global_step": self.global_step})

        self.val_loss.reset()
            

    def _generate_and_log_video(self, x):
        b, t, c, h, w = x.shape
        y = torch.zeros((b,), device=x.device, dtype=torch.long)

        # Prepare the scheduler for inference
        self.scheduler.set_timesteps(self.hparams.num_inference_steps)
        timesteps = self.scheduler.timesteps  # e.g. something like torch.arange()

        # We'll store generated frames here and then concat along time dimension
        generated_frames = []
        # shape (B, C, 1, H, W)
        x_gen = torch.randn(b, c, 1, h, w, device=x.device)
        in_gen_dict = {i: x_gen for i in range(t)}
        for step_t in timesteps:
            memory_states = None
            # Diffusion reverse loop (for each inference step)
            for frame_idx in range(t):
                # Prepare the second chunk (ground-truth only for the first frame)
                x_gen = in_gen_dict[frame_idx]
                if frame_idx == 0:
                    # Mimic training: cat the real first frame in the second chunk
                    # shape: (B, c, 1, H, W)
                    first_frame = x[:, 0].unsqueeze(2)  # ground-truth frame 0
                    # => (B, C, 1, H, W)
                    x_in = torch.cat([x_gen, first_frame], dim=1)  # => (B, 2*C, 1, H, W)
                else:
                    # For subsequent frames, we cat all zeros
                    zeros_ = torch.zeros_like(x_gen)  # (B, C, 1, H, W)
                    x_in = torch.cat([x_gen, zeros_], dim=1)      # => (B, 2*C, 1, H, W)

                # Permute to (B, 1, 2*C, H, W), matching your model's forward usage
                x_in = x_in.permute(0, 2, 1, 3, 4)

                # Model forward
                t_tensor = torch.full((b,), step_t, device=x.device, dtype=torch.long)
                model_output, memory_states = self.model.forward(x_in, t_tensor, cond=y, cache_params=memory_states, use_cache=True, run=frame_idx)

                # Take one step in reverse diffusion
                model_output = model_output.permute(0, 2, 1, 3, 4)
                out = self.scheduler.step(model_output, step_t, x_gen)
                x_gen = out.prev_sample  # shape still (B, C, 1, H, W)
                in_gen_dict[frame_idx] = x_gen

            # Now we have the final reversed sample for this frame
            #generated_frames.append(x_gen)
        generated_frames = [in_gen_dict[i] for i in range(t)]
        # Concatenate all frames along time dimension => (B, C, T, H, W)
        generated_video = torch.cat(generated_frames, dim=2)

        # (Optional) Log or visualize the generated video
        # For logging, many prefer the shape (B, T, C, H, W),
        # so permute back: (B, T, C, H, W)
        generated_video_for_logging = generated_video #.permute(0, 2, 1, 3, 4)

        # Example logging with W&B or another logger:
        for i in range(min(generated_video_for_logging.shape[0], 32)):  # up to 4 examples
            vid = preprocess_and_format_video(generated_video_for_logging[i])
            if self.logger is not None:
                self.logger.experiment.log(
                    {f"val/generated_video_{i}": wandb.Video(vid, fps=1, format="gif")}
                )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Test step: Similar to training/validation.
        """
        x = batch["video"].to(self.device)
        noise = torch.randn_like(x)
        batch_size = x.shape[0]
        timesteps = torch.randint(
            0, self.ddpm_scheduler.num_train_timesteps, (batch_size,), device=x.device
        ).long()
        x_noisy = self.ddpm_scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.forward(x_noisy, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        self.test_loss.update(loss)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    


    def log_gradient_stats(self):
        """Log gradient statistics such as the mean or std of norm."""

        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
                    gpr.append(torch.norm(param.grad) / torch.norm(param))
            if len(grad_norms) == 0:
                return
            grad_norms = torch.tensor(grad_norms)
            gpr = torch.tensor(gpr)
            self.log_dict(
                {
                    "train/grad_norm/min": grad_norms.min(),
                    "train/grad_norm/max": grad_norms.max(),
                    "train/grad_norm/std": grad_norms.std(),
                    "train/grad_norm/mean": grad_norms.mean(),
                    "train/grad_norm/median": torch.median(grad_norms),
                    "train/gpr/min": gpr.min(),
                    "train/gpr/max": gpr.max(),
                    "train/gpr/std": gpr.std(),
                    "train/gpr/mean": gpr.mean(),
                    "train/gpr/median": torch.median(gpr),
                }
            )


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
        return self._run_vae(
            x, shape, lambda y: self.vae.encode(2.0 * y - 1.0).sample()
        )

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
        """Converts the state_dict if self.model is compiled, to uncompiled."""
        if self.compile_model:
            checkpoint["state_dict"] = {
                k.replace("model._orig_mod.", "model."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _compile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict to the format expected by the compiled model."""
        if self.compile_model:
            checkpoint["state_dict"] = {
                k.replace("model.", "model._orig_mod."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return key.startswith("model.model") or key.startswith(
            "model._orig_mod.model"
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
            if self.hparams.strict_load:
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
            "model." + k for k, _ in self.model.named_parameters()
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
    model = DiT_S_8(in_channels=8, input_size=16, out_channels=4, num_classes=10, autoencoder=autoencoder)
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

    #loss = diffusion_trainer.training_step(batch, batch_idx=0)
    val = diffusion_trainer.validation_step(batch, batch_idx=0)
    from lightning import Trainer

    trainer = Trainer(devices=1, num_sanity_val_steps=1, val_check_interval=1)
    trainer.fit(diffusion_trainer, loader,)

    # Create a dummy video batch.
    # For example, batch_size=16, channels=1, frames=16, height=64, width=64.
    dummy_batch = {
        "video": torch.rand(16, 3, 64, 64) * 2 - 1,
        "cond": {"img": torch.rand(16, 3, 64, 64) * 2 - 1, "y": torch.randint(0, 10, (16,))},
    }  # values in [-1, 1]

    # Simulate one training step.
    print(0)
