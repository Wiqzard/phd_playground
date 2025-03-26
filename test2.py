from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric


###############################################################################
# Helper Function: Langevin Dynamics
###############################################################################
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
    x_neg = x_init.clone().detach().requires_grad_(False)  # True

    for _ in range(K):
        # Forward pass: compute energy
        energy = model(x_neg)
        # gradient of the sum of energies w.r.t. x_neg
        grad = torch.autograd.grad(energy.sum(), x_neg, retain_graph=False)[0]
        # Update rule: x <- x - step_size * grad + noise
        x_neg = x_neg - step_size * grad + sigma * torch.randn_like(x_neg)
        # Optionally clamp to valid range
        # x_neg = x_neg.clamp(-1.0, 1.0)
        x_neg.requires_grad_(False)  # True

    return x_neg.detach()


###############################################################################
# LightningModule: EBM for Video (Demo)
###############################################################################
class VideoMNISTLitModule(LightningModule):
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
        self.save_hyperparameters(
            logger=False, ignore=("net", "optimizer", "scheduler")
        )

        self.net = net
        self.optimzer = optimizer
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
        return self.net(x).squeeze(-1)

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
        for _ in range(batch_size):
            # with probability replay_prob, sample from replay buffer
            if (
                len(self.replay_buffer) > 0
                and torch.rand(1).item() < self.hparams.replay_prob
            ):
                # pick a random negative sample from buffer
                buf_sample = self.replay_buffer[
                    torch.randint(len(self.replay_buffer), (1,)).item()
                ]
                # move to GPU if needed
                x_init_list.append(buf_sample.to(device))
            else:
                # sample random noise in [-1, 1]
                random_sample = 2.0 * torch.rand_like(x_pos[0]) - 1.0
                x_init_list.append(random_sample)

        # stack them along batch dimension
        x_init = torch.stack(x_init_list, dim=0).to(device)

        # ----------------------------
        # 2) Langevin dynamics
        # ----------------------------
        x_neg = sample_langevin(
            model=self,
            x_init=x_init,
            K=self.hparams.K,
            step_size=self.hparams.step_size,
            sigma=self.hparams.sigma,
        )

        # ----------------------------
        # 3) Compute energies
        # ----------------------------
        e_pos = self.forward(x_pos)  # E_\theta(x^+)
        e_neg = self.forward(x_neg)  # E_\theta(x^-)

        # ----------------------------
        # 4) Compute EBM loss
        #    alpha * (E^2) + (E_pos - E_neg)
        # ----------------------------
        loss_l2 = (e_pos**2 + e_neg**2).mean()
        loss_ml = (e_pos - e_neg).mean()
        loss = self.hparams.alpha * loss_l2 + loss_ml

        # ----------------------------
        # 5) Update replay buffer
        # ----------------------------
        x_neg_cpu = x_neg.detach().cpu()
        self.replay_buffer.extend(list(x_neg_cpu))
        # Keep only the most recent samples
        if len(self.replay_buffer) > self.hparams.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.hparams.buffer_size :]

        return loss

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
        x_pos = batch.to(self.device)  # ensure on correct device
        loss = self.ebm_step(x_pos)

        # Update training loss metric and log
        self.train_loss.update(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
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
        x_pos = batch.to(self.device)
        # Compute E_\theta(x) as a simple check or re-use ebm_step logic if desired
        e_pos = self.forward(x_pos)
        # For demonstration, let's do a naive "loss" = average energy
        loss = e_pos.mean()

        self.val_loss.update(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

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
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
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
