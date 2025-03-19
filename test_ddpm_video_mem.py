from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
import wandb
from tqdm import tqdm

# Import the DDPM scheduler from diffusers
#from src.models.components.moe_lora import inject_lora, disable_all_adapters, enable_all_adapters, set_lora_trainability, reset_all_lora_parameters, get_lora_adapter_parameters, get_lora_adapter_parameters, set_global_trainability , get_global_parameters
from diffusers import DDPMScheduler



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


class DiffusionModelTrainer(LightningModule):
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
        data_mean: float = 0.0,
        data_std: float = 1.0,
        meta_learning: bool = False,
        num_inner_steps: int = 10,
        inner_lr: float = 1e-4,
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
        self.automatic_optimization = not self.hparams.meta_learning 
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

        # Metrics for logging
        self.train_loss = MeanMetric()
        self.recon_loss = MeanMetric()
        self.diffusion_loss = MeanMetric()

        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()


    def inner_loop(self, in_frame, num_inner_steps):
        losses = []
        step_inner_loss = 0
        _, meta_optimizer = self.optimizers()
        for i in range(num_inner_steps):
            #with self.model.no_sync():  # Prevent gradient synchronization
                _, inner_loss = self.forward(in_frame)
                meta_optimizer.zero_grad()
                #self.manual_backward(inner_loss)
                # maybe this does not sync the gradients
                inner_loss.backward()
                meta_optimizer.step()
                step_inner_loss += inner_loss

        step_inner_loss = step_inner_loss / num_inner_steps
        return step_inner_loss

    def outer_loop(self, x, y):
        pass

    def meta_learning_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Meta-learning step that computes the outer loop loss.
        """
        x = batch["videos"].to(self.device)
        if not hasattr(self, "x"):
            self.x = x
        x = x * 2 - 1
        bs, t, c, h ,w = x.shape
        in_frame = x[:, 0].squeeze(1).requires_grad_(True)
        next_frame = x[:, 1].squeeze(1).requires_grad_(True)


        model_optimizer, meta_optimizer = self.optimizers()

        set_global_trainability(self.model, False)
        reset_all_lora_parameters(self.model)
        set_lora_trainability(self.model, True)
        num_inner_steps = self.hparams.num_inner_steps
        step_inner_loss = self.inner_loop(in_frame, num_inner_steps)

        set_global_trainability(self.model, True)
        set_lora_trainability(self.model, False)

        _, outer_loss = self.forward(next_frame) 
        model_optimizer.zero_grad()
        self.manual_backward(outer_loss)
        model_optimizer.step()

        self.train_loss.update(step_inner_loss)
        self.diffusion_loss.update(outer_loss)
        self.log("train/inner_loss", step_inner_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/outer_loss", outer_loss, on_step=True, on_epoch=False, prog_bar=True)
        return outer_loss




    def forward(self, x, timesteps=None, loss=True):
        noise = torch.randn_like(x)
        batch_size = x.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (batch_size,), device=x.device
            ).long()
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)
        y = torch.zeros((batch_size,), device=x.device, dtype=torch.long)
        noise_pred = self.model.forward(x_noisy, timesteps, y=y)
        x_pred = self.scheduler.get_velocity(sample=noise_pred, noise=x_noisy, timesteps=timesteps)

        if not loss:
            return x_pred, timesteps

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        
        while len(weights.shape) < len(x_pred.shape):
            weights = weights.unsqueeze(-1)
        diffusion_loss = self.loss(x_pred, x, weights)  # adjust as needed
        return  x_pred, diffusion_loss


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step that jointly trains the autoencoder and the diffusion process.
        It computes:
        1. An autoencoder reconstruction loss.
        2. A diffusion loss.
        The total loss is a weighted sum of these two terms.
        """
        # Get the input video and select a frame (e.g., the second frame)
        x = batch["videos"].to(self.device)  # shape: (B, C, T, H, W)
        if not hasattr(self, "x"):
            self.x = x
        x = self.x
        x = x * 2 - 1
        
        # ----- Diffusion branch: Compute diffusion loss -----
        #num_timesteps = self.scheduler.config.num_train_timesteps
        #u = torch.rand(bs, device=x.device)
        #timesteps = (torch.pow(u, 0.25) * (num_timesteps - 1)).long()
        timesteps = None
        x_pred, diffusion_loss = self.forward(x, loss=True, timesteps=timesteps)
        
        # ----- Combine losses -----
        # You might want to weight the autoencoder loss differently (using lambda_ae)
        lambda_ae = 1.0  # adjust this hyperparameter based on your needs
        recon_loss = 0
        total_loss = diffusion_loss + lambda_ae * recon_loss
        
        # Update metrics and log losses
        self.train_loss.update(total_loss)
        self.recon_loss.update(recon_loss)
        self.diffusion_loss.update(diffusion_loss)
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/diffusion_loss", diffusion_loss, on_step=True, on_epoch=False, prog_bar=True)
        return total_loss

    #def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        #"""
        #Training step that jointly trains the autoencoder and the diffusion process.
        #It computes:
        #1. An autoencoder reconstruction loss.
        #2. A diffusion loss.
        #The total loss is a weighted sum of these two terms.
        #"""
        ## Get the input video and select a frame (e.g., the second frame)
        #x = batch["videos"].to(self.device)  # shape: (B, C, T, H, W)
        #x = x * 2 - 1
        #bs, t, c, h ,w = x.shape
        #if t > 1:
            #x = x[:, 0].squeeze(1)

        ## ----- Diffusion branch: Compute diffusion loss -----
        #noise = torch.randn_like(x)
        #batch_size = x.shape[0]
        #timesteps = torch.randint(
            #0, self.scheduler.config.num_train_timesteps, (batch_size,), device=x.device
        #).long()
        #x_noisy = self.scheduler.add_noise(x, noise, timesteps)
        #y = torch.zeros((bs,), device=x.device, dtype=torch.long)
        #noise_pred = self.model.forward(x_noisy, timesteps, y=y)

        ## Get the predicted latent velocity (or denoised latent) from the scheduler
        #x_pred = self.scheduler.get_velocity(sample=noise_pred, noise=x_noisy, timesteps=timesteps)

        #alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        #weights = 1 / (1 - alphas_cumprod)
        
        #while len(weights.shape) < len(x_pred.shape):
            #weights = weights.unsqueeze(-1)
        #diffusion_loss = self.loss(x_pred, x, weights)  # adjust as needed

        ## ----- Combine losses -----
        ## You might want to weight the autoencoder loss differently (using lambda_ae)
        #lambda_ae = 1.0  # adjust this hyperparameter based on your needs
        #recon_loss = 0
        #total_loss = diffusion_loss + lambda_ae * recon_loss

        ## Update metrics and log losses
        #self.train_loss.update(total_loss)
        #self.recon_loss.update(recon_loss)
        #self.diffusion_loss.update(diffusion_loss)
        #self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        #self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=False, prog_bar=True)
        #self.log("train/diffusion_loss", diffusion_loss, on_step=True, on_epoch=False, prog_bar=True)
        #return total_loss

    def loss(self, pred, target, weight=None):
        if weight is None:
            weight = torch.ones_like(pred)
        return torch.mean((weight * (pred - target) ** 2).reshape(pred.shape[0], -1))

    
    def meta_validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        torch.set_grad_enabled(True)
        self.model.train()
        x = batch["videos"].to(self.device)  # shape: (B,  T,C H, W)
        if hasattr(self, "x"):
            x = self.x
        x = x * 2 - 1
        bs, s, c, h, w = x.shape
        in_frame = x[:, 0].squeeze(1).requires_grad_(True)
        next_frame = x[:, 1].squeeze(1).requires_grad_(True)


        model_optimizer, meta_optimizer = self.optimizers()

        set_global_trainability(self.model, False)
        reset_all_lora_parameters(self.model)
        set_lora_trainability(self.model, True)
        # print parameters that require grad
        num_inner_steps = 10 
        #loss = self.inner_loop(in_frame, num_inner_steps)

        y = torch.zeros((bs,), device=x.device, dtype=torch.long)
        self.scheduler.set_timesteps(self.hparams.num_inference_steps)
        num_runs = self.hparams.num_gen_steps  

        video = [in_frame]
        for run in range(2):
            loss = self.inner_loop(in_frame, num_inner_steps)
            x_gen = torch.randn_like(in_frame, requires_grad=True)
            for t in self.scheduler.timesteps: 
                t_tensor = torch.full((bs,), t, device=self.device, dtype=torch.long)
                model_output = self.model.forward(x_gen, t_tensor, y=y)
                step_output = self.scheduler.step(model_output, t, x_gen)
                x_gen = step_output.prev_sample
            in_frame = x_gen.detach().requires_grad_(True)
            video.append(x_gen)

        torch.set_grad_enabled(False)

        # (Optional) Log the generated video
        video = torch.stack(video, dim=2)
        for i in range(video.shape[0]):
            generated_video = preprocess_and_format_video(video[i])
            if self.logger is not None:
                self.logger.experiment.log(
                    {f"val/generated_video_{i}": wandb.Video(generated_video, fps=4, format="gif")}
                )

        mse_loss = self.loss(video[:, 1], next_frame)        
        self.val_loss.update(mse_loss)
        self.log("val/loss", mse_loss, on_step=True, on_epoch=False, prog_bar=True)

    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step:
        1. Generate a sample from pure noise via the reverse diffusion process.
        2. Log the generated sample.
        3. Also compute the standard diffusion loss (matching the training_step logic).
        """
        if self.hparams.meta_learning:
            self.meta_validation_step(batch, batch_idx)
        else:
            x = batch["videos"].to(self.device)  # shape: (B,  T,C H, W)
            x = self.x if hasattr(self, "x") else x
            x = x * 2 - 1
            x_gt = x.clone()
            if x.shape[2] > 1:
                x = x[:, 0].squeeze(1)
                sample_shape = x.shape

            y = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
            self.scheduler.set_timesteps(self.hparams.num_inference_steps)
            num_runs = self.hparams.num_gen_steps  

            video = [x]
            for run in range(num_runs):
                x_gen = torch.randn_like(x)

                for t in self.scheduler.timesteps: 
                    t_tensor = torch.full((sample_shape[0],), t, device=self.device, dtype=torch.long)
                    model_output = self.model.forward(x_gen, t_tensor, y=y)
                    step_output = self.scheduler.step(model_output, t, x_gen)
                    x_gen = step_output.prev_sample

                video.append(x_gen)

            # (Optional) Log the generated video
            video = torch.stack(video, dim=2)
            for i in range(video.shape[0]):
                generated_video = preprocess_and_format_video(video[i])
                if self.logger is not None:
                    self.logger.experiment.log(
                        {f"val/generated_video_{i}": wandb.Video(generated_video, fps=4, format="gif")}
                    )
            x = x_gt[:, 0].clone()  # shape: (B, C, H, W)

            bs, c, h, w = x.shape

            noise = torch.randn_like(x)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bs,), device=x.device
            ).long()

            x_noisy = self.scheduler.add_noise(x, noise, timesteps)

            y = torch.zeros((bs,), device=x.device, dtype=torch.long)  # or your real labels if you have them
            noise_pred = self.model.forward(x_noisy, timesteps, y=y)

            # Get predicted velocity
            x_pred = self.scheduler.get_velocity(sample=noise_pred, noise=x_noisy, timesteps=timesteps)

            # Weighting factor as in training_step
            alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]  # shape: (bs,)
            weights = 1 / (1 - alphas_cumprod)
            while len(weights.shape) < len(x_pred.shape):
                weights = weights.unsqueeze(-1)

            # Use the same custom MSE with weights
            diffusion_loss = self.loss(x_pred, x, weights)

            # Update and log
            self.val_loss.update(diffusion_loss)
            self.log("val/loss", diffusion_loss, on_step=True, on_epoch=False, prog_bar=True)


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
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        
        if self.hparams.meta_learning:
            model_params = get_global_parameters(self.model, exclude_lora=True)
            optimizer = self.optimizer(params=model_params)
            optimizers = [optimizer]

            lora_params = get_lora_adapter_parameters(self.model)
            meta_optimizer = torch.optim.SGD(lora_params, lr=self.hparams.inner_lr)
            optimizers.append(meta_optimizer)
            return optimizers
        else:
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





        if False:
            self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

            # 2. VAE model
            if self.is_latent_diffusion and self.is_latent_online:
                self._load_vae()
            else:
                self.vae = None

            # 3. Metrics
            registry = SharedVideoMetricModelRegistry()
            metric_types = self.logging.metrics
            self.metrics_prediction = VideoMetric(
                registry,
                metric_types,
                split_batch_size=self.logging.metrics_batch_size,
            )



###############################################################################
# Example Usage (Stand-Alone)
###############################################################################

if __name__ == "__main__":
    # Create a dummy 3D UNet model.
    # model = DummyUNet3D(in_channels=1, out_channels=1, base_channels=32)

    #from src.models.components.dit import DiT_S_8
    #from src.models.components.autoencoder.simple_autoencoder import AutoEncoder
    from insert_memory import DiT3D
    model = DiT3D(in_channels=3, input_size=16, out_channels=4, num_classes=10)
    #autoencoder = AutoEncoder(in_channels=3, latent_dim=4, hidden_size=64, downsampling_factor=4)
    #model = DiT_S_8(in_channels=8, input_size=16, out_channels=4, num_classes=10, autoencoder=autoencoder)
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
