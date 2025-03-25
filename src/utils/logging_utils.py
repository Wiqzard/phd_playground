import os
from typing import Any, Dict, List, Optional
from pathlib import Path

import PIL
import torch
import numpy as np
import imageio
import wandb

from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
from torchvision.io import write_video
#from lightning_utilities.core.rank_zero import rank_zero_only
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger


log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
    return wandb_logger


def plot_samples(preds, gts, title="samples", path="./", wandb_logger=None):
    path = os.path.join(path, "samples")
    os.makedirs(path, exist_ok=True)

    for i in range(len(preds)):
        write_video(
            os.path.join(path, f"pred_{title}_{i}.mp4"), preds[i].squeeze(0), fps=7
        )
        write_video(os.path.join(path, f"gt_{title}_{i}.mp4"), gts[i].squeeze(0), fps=7)

    if wandb_logger:
        try:
            wandb_logger.log(key=f"preds_{title}", media=f"samples/preds_{title}.mp4")
            wandb_logger.log(key=f"gts_{title}", media=f"samples/gts_{title}.mp4")

        except PIL.UnidentifiedImageError:
            print(f"ERROR logging {title}")


#@rank_zero_only
def log_video(
    observation_hats: List[torch.Tensor] | torch.Tensor,
    observation_gt: Optional[torch.Tensor] = None,
    step=0,
    namespace="train",
    prefix="video",
    postfix=[],
    captions=[],
    indent=0,
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
    n_frames=None,
    raw_dir=None,
    fps=2
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_gt: ground-truth observation tensor of shape (batch, frame, channel, height, width)
    :param observation_hats: list of predicted observation tensor of shape (batch, frame, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param postfix: a list of strings specify postfixes for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """
    if not logger:
        logger = wandb
    if isinstance(observation_hats, torch.Tensor):
        observation_hats = [observation_hats]
    if observation_gt is None:
        observation_gt = torch.zeros_like(observation_hats[0])
    observation_gt = observation_gt.type_as(observation_hats[0])

    if isinstance(context_frames, int):
        context_frames = torch.arange(context_frames, device=observation_gt.device)
    for observation_hat in observation_hats:
        observation_hat[:, context_frames] = observation_gt[:, context_frames]
    
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        observation_gt_np, observation_hat_np = map(
            lambda x: (
                np.clip(x.detach().cpu().numpy(), a_min=0.0, a_max=1.0) * 255
            ).astype(np.uint8),
            (observation_gt, observation_hats[0]),
        )

        for i, (gt, hat) in enumerate(zip(observation_gt_np, observation_hat_np)):
            (raw_dir / f"{i + indent}").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw_dir / f"{i + indent}/data.npz", gt=gt, gen=hat)

            frames = [np.transpose(frame, (1, 2, 0)) for frame in hat]
            imageio.mimwrite(
                (raw_dir / f"{i + indent}") / "gen_preview.mp4",
                frames,
                fps=5,
                macro_block_size=None,
            )

    # Add red border of 1 pixel width to the context frames
    context_frames, indices = torch.meshgrid(
        context_frames,
        torch.tensor([0, -1], device=observation_gt.device, dtype=torch.long),
        indexing="ij",
    )
    for i, c in enumerate(color):
        c = c / 255.0
        for observation_hat in observation_hats:
            observation_hat[:, context_frames, i, indices, :] = c
            observation_hat[:, context_frames, i, :, indices] = c
        observation_gt[:, :, i, [0, -1], :] = c
        observation_gt[:, :, i, :, [0, -1]] = c
    video = torch.cat([*observation_hats, observation_gt], -1).detach().cpu().numpy()

    # reshape to original shape
    if n_frames is not None:
        video = rearrange(
            video, "(b n) t c h w -> b (n t) c h w", n=n_frames // video.shape[1]
        )

    video = (np.clip(video, a_min=0.0, a_max=1.0) * 255).astype(np.uint8)
    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)
    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    if isinstance(captions, str):
        captions = [captions] * n_samples
    for i in range(n_samples):
        name = f"{namespace}/{prefix}_{i + indent}" + (
            f"_{postfix[i]}" if i < len(postfix) else ""
        )
        caption = captions[i] if i < len(captions) else None
        logger.log(
            {
                name: wandb.Video(video[i], fps=fps, caption=caption),
                "trainer/global_step": step,
            },
            
        )
