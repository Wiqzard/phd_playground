from typing import Any, Dict, Tuple

import cv2
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from torchmetrics import MeanMetric

###############################################################################
# Helper Function: Langevin Dynamics
###############################################################################

import math

def get_cosine_sigma(step, max_steps, sigma_min, sigma_max):
    """
    Returns a sigma value following a cosine schedule from sigma_max down to sigma_min.
    step: current training step
    max_steps: total number of training steps
    sigma_min: lower bound on sigma
    sigma_max: upper bound on sigma
    """
    # Make sure step is in [0, max_steps]
    fraction = step / float(max_steps)
    # Cosine decay from sigma_max down to sigma_min
    sigma_t = sigma_min + 0.5 * (sigma_max - sigma_min) * (1 + math.cos(math.pi * fraction))
    return sigma_t


def preprocess_and_format_video(x_neg):
    """
    Preprocesses the input tensor and formats it for logging.

    Args:
        x_neg (torch.Tensor): Input tensor of shape (B, C, T, H, W) or (C, T, H, W).

    Returns:
        numpy.ndarray: Preprocessed video formatted as (B, T, H, W, C) and scaled to [0, 255].
    """
    # Ensure the tensor is normalized to [0, 1]
    # x_neg = (x_neg - x_neg.min()) / (x_neg.max() - x_neg.min())

    x_neg = x_neg.clamp(-1.0, 1.0)
    x_neg = x_neg / 2.0 + 0.5

    # Check if there's a batch dimension
    if x_neg.dim() == 4:  # No batch dimension
        x_neg = x_neg.unsqueeze(0)  # Add batch dimension: (B=1, T, C, H, W)

    # Rearrange dimensions: (B, C, T, H, W) -> (B, T, C, H, W)
    video_to_log = x_neg.permute(0, 2, 1, 3, 4)

    # (B, C, T, H, W)
    # Convert to numpy and scale to [0, 255]
    video_to_log = (video_to_log * 255).to(torch.uint8).cpu().numpy()

    # Convert to RGB

    return video_to_log


def sample_langevin(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    K: int,
    step_size: float,
    sigma: float,
) -> torch.Tensor:
    """
    Runs K steps of Langevin dynamics to produce negative samples.

    Args:
        model     : Energy model E_\theta(x) that outputs a scalar energy per sample.
        x_init    : Initial samples (shape: [batch_size, ...]).
        K         : Number of Langevin steps.
        step_size : Step size for gradient-based updates.
        sigma     : Std of Gaussian noise at each step.

    Returns:
        x_neg (torch.Tensor): Negative samples after Langevin dynamics.
    """

    x_neg = x_init.requires_grad_(True)
    mask = torch.ones_like(x_neg)  # Create a mask of ones
    # mask[:, :, 0, :, :] = 0        # Set the first frame to zero in the mask

    for k in range(K):
        model(x_neg).sum().backward()
        with torch.no_grad():
            #x_neg.grad.clamp_(-0.01, 0.01)
            # set first frame to zero
            update = (
                -step_size * x_neg.grad
            )   + sigma * torch.randn(*x_neg.shape).to(x_neg.device)
            update *= mask  # Apply the mask to zero out updates for the first frame
            x_neg += update
            x_neg.clamp_(-1, 1)
        x_neg.grad.zero_()
    return x_neg.requires_grad_(False)
    import math

def get_cosine_value(current_step, total_steps, min_value, max_value):
    """
    Returns a value following a (half) cosine schedule 
    from max_value down to min_value.
    
    current_step: which iteration we are in [0, total_steps-1]
    total_steps : total number of Langevin steps (K)
    min_value   : lower bound
    max_value   : upper bound
    """
    # fraction goes from 0 to 1 across total_steps
    fraction = current_step / float(total_steps - 1)  # -1 to ensure fraction=1 at last step
    return min_value + 0.5 * (max_value - min_value) * (1 + math.cos(math.pi * fraction))


    # x_neg = x_init.clone().detach().requires_grad_(True)  # True
    # for i in range(K):
    #    # Forward pass: compute energy
    #    energy = model(x_neg)
    #    # gradient of the sum of energies w.r.t. x_neg
    #    grad = torch.autograd.grad(energy.sum(), x_neg, retain_graph=False)[0]
    #    # Update rule: x <- x - step_size * grad + noise
    #    x_neg = x_neg - step_size * grad + sigma * torch.randn_like(x_neg)

    #    #with torch.no_grad:
    #    #    print(i)
    #    #    print(grad.mean(), grad.std(), grad.min(), grad.max())
    #    #    print(x_neg.mean(), x_neg.std(), x_neg.min(), x_neg.max())
    #    # Optionally clamp to valid range
    #    # x_neg = x_neg.clamp(-1.0, 1.0)
    #    x_neg.requires_grad_(True)

    return x_neg.detach()


class EnergyBasedTrainer(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,  # your energy network E_\theta
        optimizer: torch.optim.Optimizer,  # optimizer class (e.g., Adam)
        scheduler: torch.optim.lr_scheduler,  # optional LR scheduler class
        compile: bool,
        # EBM hyperparameters:
        K: int = 10,  # number of Langevin steps
        step_size: float = 0.05,  # Langevin step size
        sigma: float = 0.01,  # Langevin noise scale
        alpha: float = 0.1,  # weight for L2 energy term
        buffer_size: int = 500,  # replay buffer size
        replay_prob: float = 0.95,  # probability to sample from replay buffer
    ) -> None:
        """Initialize the EBM Lightning Module.

        :param net: The energy model E_\theta to train.
        :param optimizer: The optimizer class to use for training.
        :param scheduler: (Optional) The learning rate scheduler class.
        :param compile: If True, use `torch.compile` for the net (PyTorch 2.x).
        :param K: Number of Langevin steps during training.
        :param step_size: Step size for Langevin updates.
        :param sigma: Stddev of Gaussian noise in Langevin updates.
        :param alpha: Weight for L2 energy term in the EBM loss.
        :param buffer_size: Maximum capacity of the replay buffer.
        :param replay_prob: Probability of sampling from the replay buffer vs. random init.
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=("net", "optimizer", "scheduler"))

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Replay buffer: store negative samples on CPU (list of torch.Tensors)
        self.replay_buffer = []

        # We'll track the training, val, test losses using MeanMetric
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    ###########################################################################
    # Forward: compute energy of input x
    ###########################################################################
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the energy network.
        x -> E_\theta(x). The output is a scalar energy per sample.

        :param x: A batch of data (tensor).
        :return: Energies (shape: [batch_size]).
        """
        # Ensure output is (B,) for a batch size B
        print(2000 * "ones?")
        y, t = torch.zeros(x.shape[0], device=x.device), torch.zeros(x.shape[0], device=x.device)
        return self.net(x, y, t).squeeze(-1)

    ###########################################################################
    # Utility: on_train_start
    ###########################################################################
    def on_train_start(self) -> None:
        """Lightning hook called when training begins."""
        # Reset metrics
        self.train_loss.reset()
        self.val_loss.reset()

    ###########################################################################
    # EBM Step: Sample x^- and compute EBM loss
    ###########################################################################
    def ebm_step(self, x_pos: torch.Tensor) -> torch.Tensor:
        """
        Perform the EBM step:
          1) Initialize x^0 from replay buffer or random noise
          2) Run Langevin dynamics to get x^-
          3) Compute E_\theta(x^+) and E_\theta(x^-)
          4) Compute combined EBM loss
          5) Optionally update replay buffer

        :param x_pos: Positive samples (from data) shape [B, ...].
        :return: A scalar loss for this batch.
        """
        # ----------------------------
        # 1) Initialize negative samples
        # ----------------------------
        batch_size = x_pos.shape[0]
        device = x_pos.device

        x_init_list = []
        for k in range(batch_size):
            # with probability replay_prob, sample from replay buffer
            if len(self.replay_buffer) > 0 and torch.rand(1).item() < self.hparams.replay_prob:
                # pick a random negative sample from buffer
                buf_sample = self.replay_buffer[
                    torch.randint(len(self.replay_buffer), (1,)).item()
                ]
                # move to GPU if needed
                x_init_list.append(buf_sample.to(device))
            else:
                # sample random noise in [-1, 1]
                # x_pos_in = x_pos[k : k + 1].clone()  # 1, C, T, H, W
                ##noise = torch.randn_like(x_pos_in[:, :, 1:, :, :])  * self.hparams.sigma
                # noise = torch.rand_like(x_pos_in[:, :, 1:, :, :])  * 2 - 1
                # x_pos_in[:, :, 1:, :, :] = noise
                # x_init_list.append(x_pos_in.squeeze(0))

                random_sample = torch.randn_like(x_pos[0])  # * self.hparams.sigma
                x_init_list.append(random_sample)

        # stack them along batch dimension
        x_init = torch.stack(x_init_list, dim=0).to(device)

        # ----------------------------
        # 2) Langevin dynamics
        # ----------------------------
        #x_neg = sample_langevin(
            #model=self,
            #x_init=x_init,
            #K=self.hparams.K,
            #step_size=self.hparams.step_size,
            #sigma=self.hparams.sigma,
        #).detach()

        # ----------------------------
        # 3) Compute energies
        # ----------------------------

        #e_pos = self.forward(x_pos)  # E_\theta(x^+)
        #e_neg = self.forward(x_neg)  # E_\theta(x^-)

        #if torch.randn(1).item() < 1:  # 0.05:
        #    self.log_negative_training_samples(x_neg[:1,])

        # ----------------------------
        # 4) Compute EBM loss
        #    alpha * (E^2) + (E_pos - E_neg)
        # ----------------------------
        # reconstruction loss
        # MSE(gradient E(x+noise), x + noise)
        
        step = torch.randint(10000, (1,)).item()
        sigma = get_cosine_sigma(step, 10000, 0.01, 1)
        
        # 2. Sample noise ε ~ N(0, I)
        noise = torch.randn_like(x_pos)
        
        # 3. Create noisy input: x_t = x_pos + sigma * noise
        x_t = x_pos + sigma * noise
        x_t.requires_grad_(True)
        # 4. Forward pass: f_θ(x_t). 
        #    If your model returns scalar energy per sample, 
        #    you might want to sum over batch dimension => .sum(dim=-1)
        e_xt = self.forward(x_t).sum(dim=-1)
        if torch.randn(1).item() < 1:  # 0.05:
            self.log_negative_training_samples(x_t[:1,])

        # 5. Compute gradient wrt x_t:  ∇_x f_θ(x_t)
        grad_energy = torch.autograd.grad(
            outputs=e_xt,
            inputs=x_t,
            grad_outputs=torch.ones_like(e_xt),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 6. Compute the loss:  ( (ε / σ) + ∇ f_θ )^2
        #    i.e.  || (ε / σ) + grad_energy ||^2
        #    *Important:*  we want (grad_energy + noise/sigma).
        diff = grad_energy + (noise / sigma)
        loss = (diff ** 2).mean() 



        loss_l2 = 0 #(e_pos**2 + e_neg**2).mean()
        loss_ml = 0 #(e_pos - e_neg).mean()
        loss = self.hparams.alpha * loss_l2 + loss_ml + loss

        # ----------------------------
        # 5) Update replay buffer
        # ----------------------------
#        x_neg_cpu = x_neg.detach().cpu()
#        self.replay_buffer.extend(list(x_neg_cpu))
#        # Keep only the most recent samples
#        if len(self.replay_buffer) > self.hparams.buffer_size:
#            self.replay_buffer = self.replay_buffer[-self.hparams.buffer_size :]
#
        return loss

    def log_negative_training_samples(self, x_neg: torch.Tensor):
        """
        Logs negative samples generated during training.

        Args:
            x_neg (torch.Tensor): Negative samples generated during training.
        """
        # Preprocess and format the video for logging
        generated_video_to_log = preprocess_and_format_video(x_neg[:1,])
        # Log the video with WandB
        self.logger.experiment.log(
            {
                "train_generated_video": wandb.Video(
                    generated_video_to_log, fps=4, format="gif"  # Adjust FPS as needed
                )
            }
        )

    ###########################################################################
    # TRAINING STEP
    ###########################################################################
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (the "positive" examples).
        :param batch_idx: The index of the current batch.
        :return: The EBM loss for this batch.
        """
        x_pos = batch["video"].to(self.device)  # ensure on correct device
        # x_pos = batch.to(self.device)  # ensure on correct device
        loss = self.ebm_step(x_pos)

        # Update training loss metric and log
        self.train_loss.update(loss)
        # self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook called at the end of the training epoch."""
        pass

    ###########################################################################
    # VALIDATION STEP (optional for EBM)
    ###########################################################################
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Perform a single validation step on a batch of data from the validation set.
        For EBM, you might simply log energy values or attempt reconstructions, etc.
        """
        torch.set_grad_enabled(True)

        # x_pos = batch.to(self.device)
        x_pos = batch["video"].to(self.device)  # BS, C, T, H, W

        # get first frame and pad the rest with noise

        x_pos_in = x_pos[:1, ...].clone()  # 1, C, T, H, W
        noise = torch.randn_like(x_pos_in[:, :, 1:, :, :])  # * self.hparams.sigma
        # noise = torch.rand_like(x_pos_in[:, :, 1:, :, :])  * 2 - 1
        x_pos_in[:, :, 1:, :, :] = noise
        x_pos_in = torch.randn_like(x_pos_in)  # * self.hparams.sigma

        # run langevin dynamics
        x_neg = sample_langevin(
            model=self,
            x_init=x_pos_in.clone(),
            K=self.hparams.K,
            step_size=self.hparams.step_size,
            sigma=self.hparams.sigma,
        )
        print(f"MSE: {F.mse_loss(x_pos_in, x_neg)}")

        ## Compute E_\theta(x) as a simple check or re-use ebm_step logic if desired
        # e_neg = self.forward(x_neg)
        e_pos = self.forward(x_pos)
        # For demonstration, let's do a naive "loss" = average energy
        loss = e_pos.mean()

        # Preprocess and format the video for logging
        generated_video_to_log = preprocess_and_format_video(x_neg)
        ground_truth_in_video_to_log = preprocess_and_format_video(x_pos_in)
        ground_truth_video_to_log = preprocess_and_format_video(x_pos[:1,])

        # Log the video with WandB
        self.logger.experiment.log(
            {
                "generated_video": wandb.Video(
                    generated_video_to_log, fps=4, format="gif"  # Adjust FPS as needed
                )
            }
        )
        self.logger.experiment.log(
            {
                "input_video": wandb.Video(
                    ground_truth_in_video_to_log, fps=4, format="gif"  # Adjust FPS as needed
                )
            }
        )
        self.logger.experiment.log(
            {
                "ground_truth_video": wandb.Video(
                    ground_truth_video_to_log, fps=4, format="gif"  # Adjust FPS as needed
                )
            }
        )

        self.val_loss.update(loss)
        self.log(
            "val/loss", self.val_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    ###########################################################################
    # TEST STEP (optional for EBM)
    ###########################################################################
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data from the test set.
        Similar to validation, EBM test steps can vary widely depending on use-case.
        """
        x_pos = batch.to(self.device)
        e_pos = self.forward(x_pos)
        loss = e_pos.mean()

        self.test_loss.update(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    ###########################################################################
    # COMPILE (PyTorch 2.x) + OPTIMIZERS
    ###########################################################################
    def setup(self, stage: str) -> None:
        """Called at the beginning of fit (train+validate), validate, test, or predict."""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and (optionally) LR schedulers.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


###############################################################################
# Example usage (stand-alone)
###############################################################################
if __name__ == "__main__":
    # Suppose we have an example "EnergyNet" that outputs a scalar per sample.
    class EnergyNet(torch.nn.Module):
        def __init__(self, input_dim=128):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.mlp(x)

    # Minimal example:
    #   - net: an EnergyNet
    #   - Using Adam for optimization
    #   - No scheduler
    net = EnergyNet(input_dim=128)
    optimizer = torch.optim.Adam
    scheduler = None

    # Construct our LightningModule
    lit_model = VideoMNISTLitModule(
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        compile=False,
        K=10,
        step_size=0.05,
        sigma=0.01,
        alpha=0.1,
        buffer_size=500,
        replay_prob=0.95,
    )

    # Dummy: single batch of random data as positive examples
    # In practice, you'd use a DataLoader
    dummy_batch = torch.rand(16, 128) * 2 - 1  # shape [B, 128], in [-1, 1]

    # Simulate one training step
    loss = lit_model.training_step(dummy_batch, batch_idx=0)
    print("EBM training step loss:", loss.item())
