from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
import wandb
from tqdm import tqdm

# Import the DDPM scheduler from diffusers
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
        self.save_hyperparameters(
            ignore=("model", "optimizer", "lr_scheduler", "scheduler")
        )
        self.model = model

        self.optimizer = optimizer
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
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step that jointly trains the autoencoder and the diffusion process.
        It computes:
        1. An autoencoder reconstruction loss.
        2. A diffusion loss.
        The total loss is a weighted sum of these two terms.
        """
        # Get the input video and select a frame (e.g., the second frame)
        x = batch["video"].to(self.device)  # shape: (B, C, T, H, W)
        x_frame = x[:, :, 1].squeeze(2)  # shape: (B, C, H, W)

        # ----- Autoencoder branch: Compute reconstruction loss -----
        # Encode the image into the latent space
        latent = self.model.autoencoder.encode(x_frame)
        # Decode the latent representation back to image space
        x_recon = self.model.autoencoder.decode(latent)
        # Compute a reconstruction loss (e.g., MSE) between the original frame and its reconstruction
        recon_loss = F.mse_loss(x_recon, x_frame)

        latent = (
            latent.detach()
        )  # detach the latent representation for the diffusion branch

        # ----- Diffusion branch: Compute diffusion loss -----
        # Sample noise and add it to the latent representation
        noise = torch.randn_like(latent)
        batch_size = latent.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latent.device,
        ).long()
        latent_noisy = self.scheduler.add_noise(latent, noise, timesteps)

        # Optionally, incorporate a conditioning image
        img = batch["first_frame"].to(self.device)  # conditioning image

        img_latent = self.model.autoencoder.encode(img)
        # Concatenate the noisy latent with the conditioning image along the channel dimension
        # noisy_cond = torch.cat([latent_noisy, img], dim=1)

        noisy_cond = torch.cat([latent_noisy, img_latent], dim=1)

        # Compute the diffusion branch prediction
        # (Here, you can decide whether to use classifier-free guidance or other conditioning techniques)
        y = torch.argmax(batch["action"].squeeze(1), dim=1) + 1
        noise_pred = self.model.forward(noisy_cond, timesteps, y=y)

        # Optionally, if your scheduler requires only a subset of channels, slice them
        latent_noisy_slice = noisy_cond[:, : latent.shape[1]]
        # Get the predicted latent velocity (or denoised latent) from the scheduler
        latent_pred = self.scheduler.get_velocity(
            sample=noise_pred, noise=latent_noisy_slice, timesteps=timesteps
        )

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)
        diffusion_loss = self.loss(latent_pred, latent, weights)  # adjust as needed

        # ----- Combine losses -----
        # You might want to weight the autoencoder loss differently (using lambda_ae)
        lambda_ae = 1.0  # adjust this hyperparameter based on your needs
        total_loss = diffusion_loss + lambda_ae * recon_loss

        # Update metrics and log losses
        self.train_loss.update(total_loss)
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=False)
        self.log("train/diffusion_loss", diffusion_loss, on_step=True, on_epoch=False)
        return total_loss

    # def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
    #    """
    #    Training step:
    #      1. Sample noise and random timesteps.
    #      2. Add noise using the DDPM scheduler.
    #      3. Predict noise and compute MSE loss.
    #    """
    #    # Assume batch is a dictionary with key "video"
    #    x = batch["video"].to(self.device)  # shape: (B, C, T, H, W)
    #    x = x[:, :, 1].squeeze(2)
    #    noise = torch.randn_like(x)
    #    batch_size = x.shape[0]

    #    # Sample random diffusion timesteps for each sample
    #    timesteps = torch.randint(
    #        0, self.scheduler.config.num_train_timesteps, (batch_size,), device=x.device
    #    ).long()

    #    # Add noise to the original sample using the forward diffusion process
    #    x_noisy = self.scheduler.add_noise(x, noise, timesteps)

    #    img = batch["first_frame"]
    #    x_noisy = torch.cat([x_noisy, img], dim=1)

    #    # Get the model’s prediction of the noise
    #    uc = torch.rand(1) < 0.2
    #    cond = batch.get("cond", None)
    #    # make one hot to label
    #    y = torch.argmax(batch["action"].squeeze(1), dim=1) + 1

    #    noise_pred = self.model.forward(x_noisy, timesteps, y=y)
    #    x_noisy = x_noisy[:, :3]
    #    x_pred = self.scheduler.get_velocity(sample=noise_pred, noise=x_noisy, timesteps=timesteps)

    #    alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
    #    weights = 1 / (1 - alphas_cumprod)
    #    while len(weights.shape) < len(x_pred.shape):
    #        weights = weights.unsqueeze(-1)

    #    loss = self.loss(x_pred, x, weights)
    #    # Compute MSE loss between the predicted and true noise
    #    # loss = weights *  F.mse_loss(noise_pred, noise)
    #    self.train_loss.update(loss)
    #    self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
    #    return loss

    def loss(self, pred, target, weight=None):
        if weight is None:
            weight = torch.ones_like(pred)
        return torch.mean((weight * (pred - target) ** 2).reshape(pred.shape[0], -1))

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step:
          1. Generate a sample from pure noise via the reverse diffusion process.
          2. Log the generated sample.
          3. Also compute the standard diffusion loss.
        """
        # ---------
        # Reverse Diffusion: Generate a sample from noise.
        # ---------
        x = batch["video"].to(self.device)  # shape: (B, C, T, H, W)
        x = x[:, :, 1].squeeze(2)  # now shape: (B, C, H, W)

        # Use the batch's first frame as the initial conditioning frame
        img = batch["first_frame"]
        x = self.model.autoencoder.encode(x)
        img = self.model.autoencoder.encode(img)

        sample_shape = x.shape  # will be used to initialize noise

        # Prepare the action labels
        y = torch.argmax(batch["action"].squeeze(1), dim=1) + 1

        # Set the scheduler with the desired number of timesteps for inference
        self.scheduler.set_timesteps(self.hparams.num_inference_steps)

        # Define the number of runs (each run uses the previous run's generated frame as conditioning)
        num_runs = self.hparams.num_gen_steps  # Or use a different variable if needed

        video = []
        # for run in tqdm(range(num_runs), desc="Generating Samples"):
        for run in range(num_runs):
            # for run in tqdm(range(num_runs), desc="Generating Samples"):
            # Initialize a fresh noise sample for this run
            x_gen = torch.randn(sample_shape, device=self.device)
            # Concatenate the conditioning frame along the channel dimension.
            # (Assuming x_gen is (B, 3, H, W) and img is (B, 3, H, W), the result is (B, 6, H, W))
            x_gen = torch.cat([x_gen, img], dim=1)

            # Reverse diffusion loop for this run
            for t in self.scheduler.timesteps:
                # Create a tensor for the current timestep for all samples in the batch
                t_tensor = torch.full(
                    (sample_shape[0],), t, device=self.device, dtype=torch.long
                )

                # Predict the noise residual using your model
                model_output = self.model.forward(x_gen, t_tensor, y=y)

                # Update the sample using the scheduler’s step function.
                # Here, we assume that only the first 3 channels of x_gen are updated.
                step_output = self.scheduler.step(model_output, t, x_gen[:, :4])
                x_gen = step_output.prev_sample
                video.append(self.model.autoencoder.decode(x_gen))

                # Re-attach the conditioning frame so that it remains part of x_gen.
                x_gen = torch.cat([x_gen, img], dim=1)

            # After finishing the diffusion process for this run, update the conditioning frame.
            # We assume that the first 3 channels of x_gen represent the generated video frame.
            img = x_gen[:, :4]

            # Optionally: Save or process the generated video for this run.
            # For example: save_video(x_gen[:, :3], run)

        video = torch.stack(video, dim=1)
        # Log the generated sample (e.g., as a video)
        generated_video = preprocess_and_format_video(video)
        if self.logger is not None:
            self.logger.experiment.log(
                {
                    "val/generated_video": wandb.Video(
                        generated_video, fps=4, format="gif"
                    )
                }
            )
        ## ---------
        ## Also compute the diffusion loss on the batch
        ## ---------
        if True:
            noise = torch.randn_like(x)
            batch_size = x.shape[0]
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=x.device,
            ).long()
            x_noisy = self.scheduler.add_noise(x, noise, timesteps)
            x_noisy = torch.cat([x_noisy, img], dim=1)
            noise_pred = self.model.forward(x_noisy, timesteps, y=y)
            x_pred = self.scheduler.get_velocity(
                sample=noise_pred, noise=x_noisy[:, :4], timesteps=timesteps
            )

            loss = F.mse_loss(x_pred, x)
            self.val_loss.update(loss)
            self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

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
        if self.compile_model and stage == "fit":
            self.model = torch.compile(self.model)


###############################################################################
# Example Usage (Stand-Alone)
###############################################################################

if __name__ == "__main__":
    # Create a dummy 3D UNet model.
    # model = DummyUNet3D(in_channels=1, out_channels=1, base_channels=32)

    from src.models.components.dit import DiT_S_8
    from src.models.components.autoencoder.simple_autoencoder import AutoEncoder

    autoencoder = AutoEncoder(
        in_channels=3, latent_dim=4, hidden_size=64, downsampling_factor=4
    )
    model = DiT_S_8(
        in_channels=8,
        input_size=16,
        out_channels=4,
        num_classes=10,
        autoencoder=autoencoder,
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
        "cond": {
            "img": torch.rand(16, 3, 64, 64) * 2 - 1,
            "y": torch.randint(0, 10, (16,)),
        },
    }  # values in [-1, 1]

    # Simulate one training step.
    print(0)
