import os
from typing import Any, Dict

import PIL
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torchvision.io import write_video

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
        write_video(os.path.join(path, f"pred_{title}_{i}.mp4"), preds[i].squeeze(0), fps=7)
        write_video(os.path.join(path, f"gt_{title}_{i}.mp4"), gts[i].squeeze(0), fps=7)

    if wandb_logger:
        try:
            wandb_logger.log(key=f"preds_{title}", media=f"samples/preds_{title}.mp4")
            wandb_logger.log(key=f"gts_{title}", media=f"samples/gts_{title}.mp4")

        except PIL.UnidentifiedImageError:
            print(f"ERROR logging {title}")
