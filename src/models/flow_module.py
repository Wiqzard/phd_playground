import os
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from lightning import LightningModule

from utils.logging_utils import get_wandb_logger, plot_samples

from .components.cfm.augmentation import AugmentationModule, AugmentedVectorField
from .components.helpers import to_video
from .components.optimal_transport import OTPlanSampler
from .components.solver import FlowSolver


class YodaLitModule(LightningModule):
    def __init__(
        self,
        net: Any,
        optimizer: Any,
        augmentations: AugmentationModule,
        partial_solver: FlowSolver,
        scheduler: Optional[Any] = None,
        neural_ode: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        num_val_frames: int = 10,
        avg_size: int = -1,
        test_nfe: int = 100,
        path: str = "./",
        plot: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "augmentations",
                "partial_solver",
            ],
            logger=False,
        )

        self.dim = (
            [net.vector_field_regressor.num_frames_in_block[-1]]
            + [net.vector_field_regressor.in_channels]
            + net.vector_field_regressor.sample_size
        )
        self.net = net

        # Freeze the weights of the autoencoder and flow_network
        self.freeze_module(self.net.autoencoder)
        self.freeze_module(self.net.flow_network)
        # self.net.autoencoder.ae.decoder.cpu()
        # self.freeze_module(self.net.flow_representation_network)

        self.augmentations = augmentations
        self.aug_net = AugmentedVectorField(self.net, self.augmentations.regs, self.dim)

        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken for optimal Schrodinger bridge relationship
            self.ot_sampler = OTPlanSampler(method=ot_sampler, reg=2 * sigma_min**2)
        self.criterion = torch.nn.MSELoss()

        self.eval_preds = []
        self.eval_gts = []

        # for averaging loss across batches
        # self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

    def forward(self, t: torch.Tensor, x: torch.Tensor, flows: torch.Tensor):
        """Forward pass (t, x) -> dx/dt."""
        return self.net(t, x, flows)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def calc_mu_sigma(self, x0, x1, t):
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.hparams.sigma_min
        return mu_t, sigma_t

    def calc_u(self, x0, x1, x, t, mu_t, sigma_t):
        del x, t, mu_t, sigma_t
        return x1 - x0

    def calc_loc_and_target(self, x0, x1, t, t_select):
        """Computes the loss on a batch of data."""

        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t, sigma_t = self.calc_mu_sigma(x0, x1, t_xshape)
        eps_t = torch.randn_like(mu_t)
        x = mu_t + sigma_t * eps_t
        ut = self.calc_u(x0, x1, x, t_xshape, mu_t, sigma_t)

        # p is the pair-wise conditional probability matrix. Note that this has to be torch.cdist(x, mu) in that order
        # t that network sees is incremented by first timepoint
        t = t + t_select.reshape(-1, *t.shape[1:])
        return x, ut, t, mu_t, sigma_t, eps_t

    def preprocess_batch(self, X, flows=None, training: bool = False):
        t_select = torch.zeros(1, device=X.device)

        # sample conditioning and reference frames from behind, encode them, return latents and time_ids
        with torch.no_grad():
            latents, time_ids = self.net.get_input_frames(X, training)

        flows = self.net._get_flows(X, flows).unsqueeze(1)  # gets flow from -2 to -1 frame

        context = [time_ids, flows]

        x1 = latents[:, -1]
        x0 = torch.randn_like(x1)
        return latents[:, :-1], context, x0, x1, t_select

    def unpack_batch(self, batch):
        num_frames = self.net.vector_field_regressor.num_frames_in_block[0]
        if isinstance(batch, list) or isinstance(batch, tuple):
            return (
                batch[0][:, num_frames:],
                batch[0][:, -num_frames:],
                batch[1][:, num_frames:],
            )
        else:
            return batch, None  # batch[:, :num_frames], batch[:, num_frames:], None

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        # x0=noise, x1=target_frame, x=input_to_model[x:t,x0]
        X, context = self.unpack_batch(batch)

        # get input_frames, noise, target_frame, t_select
        X, context, x0, x1, t_select = self.preprocess_batch(X, context, training)
        # Either randomly sample a single T or sample a batch of T's
        if self.hparams.avg_size > 0:
            t = torch.rand(1).repeat(X.shape[0]).type_as(X)
        else:
            t = torch.rand(X.shape[0]).type_as(X)
        # Resample the plan if we are using optimal transport
        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        x, ut, t, mu_t, sigma_t, _ = self.calc_loc_and_target(x0, x1, t, t_select)

        if self.hparams.avg_size > 0:
            x, ut, t = self.average_ut(x, t, mu_t, sigma_t, ut)

        x = torch.cat([X, x.unsqueeze(1)], dim=1)
        aug_x = self.aug_net(t, x, context, augmented_input=False)
        reg, vt = self.augmentations(aug_x)

        return torch.mean(reg), self.criterion(vt, ut)

    def training_step(self, batch: Any, batch_idx: int):
        reg, mse = self.step(batch, training=True)
        loss = mse + reg
        prefix = "train"
        self.log_dict(
            {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def video_eval_step(self, batch: Any, batch_idx: int, prefix: str):
        X, flows = self.unpack_batch(batch)
        X, y = (
            X[:, : -self.hparams.num_val_frames],
            X[:, -self.hparams.num_val_frames :],
        )

        X, y = X[0, ...].unsqueeze(0), y[0, ...].unsqueeze(0)

        flows = self.net._get_flows(X, flows).unsqueeze(
            0
        )  # .unsqueeze(1)  # gets flow from -2 to -1 frame
        # latents, time_ids  = self.net.get_input_frames(X, training=False)

        with torch.no_grad():
            frames = self.net.generate_frames(
                observations=X,
                # observations=X[: min(4, X.shape[0]), ...],
                context=flows,
                num_frames=self.hparams.num_val_frames,
                steps=self.hparams.test_nfe,
                warm_start=0,
                # if main rank tTrue
                verbose=self.trainer.is_global_zero,
            )

        self.eval_preds.append(to_video(frames).permute(0, 1, 3, 4, 2))
        gt = torch.cat([frames[:, : -self.hparams.num_val_frames], y], dim=1)
        self.eval_gts.append(to_video(gt).permute(0, 1, 3, 4, 2))
        return {"x": batch[0]}

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        # if prefix == "test":
        if self.trainer.is_global_zero:
            self.video_eval_step(batch, batch_idx, prefix)
        shapes = [b.shape[0] for b in batch]

        if (
            False
        ):  # not self.is_image and prefix == "val" and shapes.count(shapes[0]) == len(shapes):
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

    def validation_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "val")

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        # calculate FID between generated and real images from above
        wandb_logger = get_wandb_logger(self.loggers)
        if prefix == "test":
            os.makedirs("images", exist_ok=True)
            if len(os.listdir("images")) > 0:
                path = ""
                from pytorch_fid import fid_score

                fid = fid_score.calculate_fid_given_paths(["images", path], 256, "cuda", 2048, 0)
                self.log(f"{prefix}/fid", fid)

        # ts, x, x0, x_rest = self.preprocess_epoch_end(outputs, prefix)
        # trajs, full_trajs = self.forward_eval_integrate(ts, x0, x_rest, outputs, prefix)
        # plot samples like in yoda
        if prefix == "val":
            preds, gt = self.eval_preds, self.eval_gts
        else:
            preds, gt = self.test_videos, self.test_gts

        if self.hparams.plot:
            for pred, gt in zip(preds, gt):
                plot_samples(
                    pred,
                    gt,
                    title=f"{self.current_epoch}_samples",
                    path=self.hparams.path,
                    wandb_logger=wandb_logger,
                )

    def test_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "test")

    def on_validation_epoch_end(self):
        outputs = self.eval_preds
        self.eval_epoch_end(outputs, "val")

    def on_test_epoch_end(self, outputs: List[Any]):
        outputs = self.test_outputs
        self.eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """Pass model parameters to optimizer."""
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        scheduler.step(epoch=self.current_epoch)
