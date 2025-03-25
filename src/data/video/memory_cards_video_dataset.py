from typing import Any, Dict, Optional, List 
import io
import tarfile
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from lightning import LightningDataModule

from torch.utils.data import DataLoader

from src.data.video.base_video import BaseAdvancedVideoDataset



class MemoryAdvancedVideoDataset(BaseAdvancedVideoDataset):
    """
    Minecraft dataset that loads sub-clips, handles frame skip, latents, etc. (advanced).
    """

    def __init__(
        self,
        # BaseVideoDataset required args
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: str = "training",

        # Additional advanced-video args
        current_epoch: Optional[int] = None,
        latent_enable: bool = False,
        latent_type: str = "pre_sample",
        external_cond_dim: int = 4,
        external_cond_stack: bool = True,
        max_frames: int = 50,
        n_frames: int = 17,
        frame_skip: int = 2,
        filter_min_len: Optional[int] = None,
        subdataset_size: Optional[int] = None,
        num_eval_videos: Optional[int] = None,
        **kwargs,
    ):
        # Force 'test' to be treated as 'validation'
        if split == "test":
            split = "validation"
        # 2) Call BaseAdvancedVideoDataset’s __init__
        BaseAdvancedVideoDataset.__init__(
            self,
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
            current_epoch=current_epoch,
            latent_enable=latent_enable,
            latent_type=latent_type,
            external_cond_dim=external_cond_dim,
            external_cond_stack=external_cond_stack,
            max_frames=max_frames,
            n_frames=n_frames,
            frame_skip=frame_skip,
            filter_min_len=filter_min_len,
            subdataset_size=subdataset_size,
            num_eval_videos=num_eval_videos,
            **kwargs,
        )

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        """
        In this dataset, we know each .mp4 is 250 frames.
        """
        return 250

    def build_transform(self):
        """
        Use nearest-neighbor upsampling to match the specified resolution.
        """
        return transforms.Resize(
            self.resolution,
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )
    
    def download_dataset(self):
        pass

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        """
        Example: load discrete actions from a .npz file, one-hot-encode them,
        and return shape [T, 4].
        """
        return None
        #path = video_metadata["video_paths"].with_suffix(".npz")
        #actions = np.load(path)["actions"][start_frame:end_frame]
        ## E.g. 4 possible actions => shape [T, 4]
        #return torch.from_numpy(np.eye(4)[actions]).float()


class MemoryDataModule(LightningDataModule):
    """
    LightningDataModule for the MemoryAdvancedVideoDataset.
    """

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: str = "training",  # typically "training", "validation", or "test"
        current_epoch: Optional[int] = None,
        latent_enable: bool = False,
        latent_type: str = "pre_sample",
        external_cond_dim: int = 4,
        external_cond_stack: bool = True,
        max_frames: int = 50,
        n_frames: int = 17,
        frame_skip: int = 2,
        filter_min_len: Optional[int] = None,
        subdataset_size: Optional[int] = None,
        num_eval_videos: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 16,
        pin_memory: bool = False,
        **kwargs,
        # You can add other parameters or hyperparameters if needed
    ):
        """
        Args:
            save_dir (str): Base directory where the dataset files reside.
            resolution (int): Desired resolution of the frames.
            latent_downsampling_factor (List[int]): Downsampling factor for latent representation.
            latent_suffix (str): Optional suffix for latent files.
            split (str): Which split to use (training, validation, or test). 
                         Note that inside MemoryAdvancedVideoDataset, 'test' is auto-converted to 'validation'.
            current_epoch (int, optional): Current training epoch; can be used by the dataset for adaptive logic.
            latent_enable (bool): Whether to enable latents or not.
            latent_type (str): Type of the latents (e.g. "pre_sample").
            external_cond_dim (int): Dimension of external conditioning (e.g. action embeddings).
            external_cond_stack (bool): Whether to stack external conditioning over time.
            max_frames (int): Maximum frames in a single video clip.
            n_frames (int): Number of frames to be sampled in a single sub-clip.
            frame_skip (int): Frame skip for sub-clip sampling.
            filter_min_len (int, optional): Filter out videos shorter than this.
            subdataset_size (int, optional): If set, limit the total size of the dataset (for debugging).
            num_eval_videos (int, optional): Number of videos to be used in eval split.
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of workers for dataloaders.
            pin_memory (bool): Whether to pin GPU memory for dataloaders.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Store hyperparameters for easy access
        self.save_dir = save_dir
        self.resolution = resolution
        self.latent_downsampling_factor = latent_downsampling_factor
        self.latent_suffix = latent_suffix
        self.split = split
        self.current_epoch = current_epoch
        self.latent_enable = latent_enable
        self.latent_type = latent_type
        self.external_cond_dim = external_cond_dim
        self.external_cond_stack = external_cond_stack
        self.max_frames = max_frames
        self.n_frames = n_frames
        self.frame_skip = frame_skip
        self.filter_min_len = filter_min_len
        self.subdataset_size = subdataset_size
        self.num_eval_videos = num_eval_videos

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Placeholders for datasets
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called by Lightning at the start of fit/test/predict.
        stage can be 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == "fit" or stage is None:
            # Training dataset
            self.dataset_train = MemoryAdvancedVideoDataset(
                save_dir=self.save_dir,
                resolution=self.resolution,
                latent_downsampling_factor=self.latent_downsampling_factor,
                latent_suffix=self.latent_suffix,
                split="training",  # Force the training split
                current_epoch=self.current_epoch,
                latent_enable=self.latent_enable,
                latent_type=self.latent_type,
                external_cond_dim=self.external_cond_dim,
                external_cond_stack=self.external_cond_stack,
                max_frames=self.max_frames,
                n_frames=self.n_frames,
                frame_skip=self.frame_skip,
                filter_min_len=self.filter_min_len,
                subdataset_size=self.subdataset_size,
                num_eval_videos=self.num_eval_videos,
            )

            # Validation dataset
            self.dataset_val = MemoryAdvancedVideoDataset(
                save_dir=self.save_dir,
                resolution=self.resolution,
                latent_downsampling_factor=self.latent_downsampling_factor,
                latent_suffix=self.latent_suffix,
                split="validation",  # Force the validation split
                current_epoch=self.current_epoch,
                latent_enable=self.latent_enable,
                latent_type=self.latent_type,
                external_cond_dim=self.external_cond_dim,
                external_cond_stack=self.external_cond_stack,
                max_frames=self.max_frames,
                n_frames=self.n_frames,
                frame_skip=self.frame_skip,
                filter_min_len=self.filter_min_len,
                subdataset_size=self.subdataset_size,
                num_eval_videos=self.num_eval_videos,
            )

        if stage == "test" or stage is None:
            # Test dataset
            self.dataset_test = MemoryAdvancedVideoDataset(
                save_dir=self.save_dir,
                resolution=self.resolution,
                latent_downsampling_factor=self.latent_downsampling_factor,
                latent_suffix=self.latent_suffix,
                split="test",  # Will be auto-converted to 'validation' inside the dataset
                current_epoch=self.current_epoch,
                latent_enable=self.latent_enable,
                latent_type=self.latent_type,
                external_cond_dim=self.external_cond_dim,
                external_cond_stack=self.external_cond_stack,
                max_frames=self.max_frames,
                n_frames=self.n_frames,
                frame_skip=self.frame_skip,
                filter_min_len=self.filter_min_len,
                subdataset_size=self.subdataset_size,
                num_eval_videos=self.num_eval_videos,
            )

    def train_dataloader(self) -> DataLoader:
        if self.dataset_train is None:
            raise ValueError("Training dataset is not initialized. Call `.setup('fit')` first.")
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.dataset_val is None:
            raise ValueError("Validation dataset is not initialized. Call `.setup('fit')` first.")
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.dataset_test is None:
            raise ValueError("Test dataset is not initialized. Call `.setup('test')` first.")
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        

                 

