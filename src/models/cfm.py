import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchvision import transforms

from .components.distribution_distances import compute_distribution_distances
from .components.optimal_transport import OTPlanSampler
from .components.plotting import (
    plot_samples,
    plot_trajectory,
    store_trajectories,
)
from .components.solver import FlowSolver
from .utils import get_wandb_logger


class CFMLitModule(LightningModule):
    """Conditional Flow Matching Module for training generative models and models over time."""

    def __init__(
        self,
        net: Any,
        optimizer: Any,
        partial_solver: FlowSolver,
        scheduler: Optional[Any] = None,
        neural_ode: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        dim: Any = (3, 32, 32),
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
        test_nfe: int = 100,
        plot: bool = False,
        nice_name: str = "CFM",
    ) -> None:
        """Initialize a conditional flow matching network either as a generative model or for a
        sequence of timepoints.

        Note: DDP does not currently work with NeuralODE objects from torchdyn
        in the init so we initialize them every time we need to do a sampling
        step.

        Args:
            net: torch module representing dx/dt = f(t, x) for t in [1, T] missing dimension.
            optimizer: partial torch.optimizer missing parameters.
            ot_sampler: ot_sampler specified as an object or string. If none then no OT is used in minibatch.
            sigma_min: sigma_min determines the width of the Gaussian smoothing of the data and interpolations.
            plot: if true, log intermediate plots during validation
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "partial_solver",
            ],
            logger=False,
        )
        self.net = net(dim=self.dim)
        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            self.ot_sampler = OTPlanSampler(method=ot_sampler, reg=2 * sigma_min**2)
        self.criterion = torch.nn.MSELoss()

    def forward_integrate(self, batch: Any, t_span: torch.Tensor):
        """Forward pass with integration over t_span intervals.

        (t, x, t_span) -> [x_t_span].
        """
        X = self.unpack_batch(batch)
        X_start = X[:, t_span[0], :]
        traj = self.node.trajectory(X_start, t_span=t_span)
        return traj

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Forward pass (t, x) -> dx/dt."""
        return self.net(t, x)

    def unpack_batch(self, batch):
        """Unpacks a batch of data to a single tensor."""
        if not isinstance(self.dim, int):
            # Assume this is an image classification dataset where we need to strip the targets
            return batch[0]
        return batch

    def preprocess_batch(self, X, training=False):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        t_select = torch.zeros(1, device=X.device)
        x0 = torch.randn_like(X)
        x1 = X
        return x0, x1, t_select

    def average_ut(self, x, t, mu_t, sigma_t, ut):
        pt = torch.exp(-0.5 * (torch.cdist(x, mu_t) ** 2) / (sigma_t**2))
        batch_size = x.shape[0]
        ind = torch.randint(
            batch_size, size=(batch_size, self.hparams.avg_size - 1)
        )  # randomly (non-repreat) sample m-many index
        # always include self
        ind = torch.cat([ind, torch.arange(batch_size)[:, None]], dim=1)
        pt_sub = torch.stack([pt[i, ind[i]] for i in range(batch_size)])
        ut_sub = torch.stack([ut[ind[i]] for i in range(batch_size)])
        p_sum = torch.sum(pt_sub, dim=1, keepdim=True)
        ut = torch.sum(pt_sub[:, :, None] * ut_sub, dim=1) / p_sum
        # Reduce batch size because they are all the same
        return x[:1], ut[:1], t[:1]

    def calc_mu_sigma(self, x0, x1, t):
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.hparams.sigma_min
        return mu_t, sigma_t

    def calc_u(self, x0, x1, x, t, mu_t, sigma_t):
        del x, t, mu_t, sigma_t
        return x1 - x0

    def calc_loc_and_target(self, x0, x1, t, t_select, training):
        """Computes the loss on a batch of data."""

        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t, sigma_t = self.calc_mu_sigma(x0, x1, t_xshape)
        eps_t = torch.randn_like(mu_t)
        x = mu_t + sigma_t * eps_t
        ut = self.calc_u(x0, x1, x, t_xshape, mu_t, sigma_t)

        # if we are starting from right before the leaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.leaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.leaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.leaveout_timepoint] *= 2

        # p is the pair-wise conditional probability matrix. Note that this has to be torch.cdist(x, mu) in that order
        # t that network sees is incremented by first timepoint
        t = t + t_select.reshape(-1, *t.shape[1:])
        return x, ut, t, mu_t, sigma_t, eps_t

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""

        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)
        # Either randomly sample a single T or sample a batch of T's
        if self.hparams.avg_size > 0:
            t = torch.rand(1).repeat(X.shape[0]).type_as(X)
        else:
            t = torch.rand(X.shape[0]).type_as(X)
        # Resample the plan if we are using optimal transport
        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        x, ut, t, mu_t, sigma_t, eps_t = self.calc_loc_and_target(
            x0, x1, t, t_select, training
        )

        if self.hparams.avg_size > 0:
            x, ut, t = self.average_ut(x, t, mu_t, sigma_t, ut)

        vt = self.net(t, x, augmented_input=False)
        return torch.self.criterion(vt, ut)

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch, training=True)
        prefix = "train"
        self.log_dict(
            {f"{prefix}/loss": loss},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def image_eval_step(self, batch: Any, batch_idx: int, prefix: str):
        import os
        from math import prod

        from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
        from torchvision.utils import save_image

        solver = self.partial_solver(self.net, self.dim)
        if isinstance(self.hparams.test_nfe, int):
            t_span = torch.linspace(0, 1, int(self.hparams.test_nfe) + 1)
        elif isinstance(self.hparams.test_nfe, str):
            solver.ode_solver = "tsit5"
            t_span = torch.linspace(0, 1, 2)
        else:
            raise NotImplementedError(f"Unknown test procedure {self.hparams.test_nfe}")
        traj = solver.odeint(
            torch.randn(batch[0].shape[0], *self.dim).type_as(batch[0]), t_span
        )[-1]
        os.makedirs("images", exist_ok=True)
        mean = [-x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [255.0 / x for x in [63.0, 62.1, 66.7]]
        inv_normalize = transforms.Compose(
            [
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=std),
                transforms.Normalize(mean=mean, std=[1.0, 1.0, 1.0]),
            ]
        )
        traj = inv_normalize(traj)
        traj = torch.clip(traj, min=0, max=1.0)
        for i, image in enumerate(traj):
            save_image(image, fp=f"images/{batch_idx}_{i}.png")
        return {"x": batch[0]}

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        if prefix == "test" and self.is_image:
            self.image_eval_step(batch, batch_idx, prefix)
        shapes = [b.shape[0] for b in batch]

        if (
            not self.is_image
            and prefix == "val"
            and shapes.count(shapes[0]) == len(shapes)
        ):
            reg, mse = self.step(batch, training=False)
            loss = mse + reg
            self.log_dict(
                {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return {"loss": loss, "mse": mse, "reg": reg, "x": self.unpack_batch(batch)}

        return {"x": batch}

    def preprocess_epoch_end(self, outputs: List[Any], prefix: str):
        """Preprocess the outputs of the epoch end function."""
        if isinstance(self.dim, int):
            v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
            x = v["x"]
        else:
            x = [d["x"] for d in outputs][0][0][:100]
        # Sample some random points for the plotting function
        rand = torch.randn_like(x)
        # rand = torch.randn_like(x, generator=torch.Generator(device=x.device).manual_seed(42))
        x = torch.stack([rand, x], dim=1)
        ts = x.shape[1]
        x0 = x[:, 0]
        x_rest = x[:, 1:]
        return ts, x, x0, x_rest

    def forward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a trajectory
        t_span = torch.linspace(0, 1, 101)
        aug_dims = self.val_augmentations.aug_dims
        regs = []
        trajs = []
        full_trajs = []
        solver = self.partial_solver(self.net, self.dim)
        nfe = 0
        x0_tmp = x0.clone()

        if self.is_image:
            traj = solver.odeint(x0, t_span)
            full_trajs.append(traj)
            trajs.append(traj[0])
            trajs.append(traj[-1])
            nfe += solver.nfe

        if not self.is_image:
            for i in range(ts - 1):
                traj, aug = solver.odeint(x0_tmp, t_span + i)
                full_trajs.append(traj)
                traj, aug = traj[-1], aug[-1]
                x0_tmp = traj
                regs.append(torch.mean(aug, dim=0).detach().cpu().numpy())
                trajs.append(traj)
                nfe += solver.nfe

        full_trajs = torch.cat(full_trajs)

        if not self.is_image:
            regs = np.stack(regs).mean(axis=0)
            names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
            self.log_dict(dict(zip(names, regs)), sync_dist=True)

            names, dists = compute_distribution_distances(trajs, x_rest)
            names = [f"{prefix}/{name}" for name in names]
            d = dict(zip(names, dists))
            if self.hparams.leaveout_timepoint >= 0:
                to_add = {
                    f"{prefix}/t_out/{key.split('/')[-1]}": val
                    for key, val in d.items()
                    if key.startswith(f"{prefix}/t{self.hparams.leaveout_timepoint}")
                }
                d.update(to_add)
            d[f"{prefix}/nfe"] = nfe

            self.log_dict(d, sync_dist=True)
        return trajs, full_trajs

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        if prefix == "test" and self.is_image:
            os.makedirs("images", exist_ok=True)
            if len(os.listdir("images")) > 0:
                path = "/home/mila/a/alexander.tong/scratch/trajectory-inference/data/fid_stats_cifar10_train.npz"
                from pytorch_fid import fid_score

                fid = fid_score.calculate_fid_given_paths(
                    ["images", path], 256, "cuda", 2048, 0
                )
                self.log(f"{prefix}/fid", fid)

        ts, x, x0, x_rest = self.preprocess_epoch_end(outputs, prefix)
        trajs, full_trajs = self.forward_eval_integrate(ts, x0, x_rest, outputs, prefix)

        if self.hparams.plot:
            if isinstance(self.dim, int):
                plot_trajectory(
                    x,
                    full_trajs,
                    title=f"{self.current_epoch}_ode",
                    key="ode_path",
                    wandb_logger=wandb_logger,
                )
            else:
                plot_samples(
                    trajs[-1],
                    title=f"{self.current_epoch}_samples",
                    wandb_logger=wandb_logger,
                )

        if prefix == "test" and not self.is_image:
            store_trajectories(x, self.net)

    def validation_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs: List[Any]):
        self.eval_epoch_end(outputs, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs: List[Any]):
        self.eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """Pass model parameters to optimizer."""
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)
