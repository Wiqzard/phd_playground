import random
from typing import Any, Dict, Optional, Tuple

import albumentations as alb
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from .h5 import HDF5Dataset


class Aug(nn.Module):
    def __init__(self, b: float, c: float, s: float, h: float):
        super().__init__()

        self.b = b
        self.c = c
        self.s = s
        self.h = h

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        im = F.adjust_brightness(im, brightness_factor=1 + self.b)
        im = F.adjust_contrast(im, contrast_factor=1 + self.c)
        im = F.adjust_saturation(im, saturation_factor=1 + self.s)
        im = F.adjust_hue(im, hue_factor=self.h)

        return im


class RandomConsistentAugFactory(nn.Module):
    def __init__(self, aug: bool = True):
        super().__init__()

        self.aug = aug

    def forward(self):
        if self.aug:
            b = (torch.rand(1).item() - 0.5) / 5
            c = (torch.rand(1).item() - 0.5) / 5
            s = (torch.rand(1).item() - 0.5) / 5
            h = (torch.rand(1).item() - 0.5) / 2
            aug = Aug(b, c, s, h)

            return aug

        else:
            return T.Lambda(lambda x: x)


class VideoDataset(Dataset):
    def __init__(
        self,
        data_path,
        input_size: int,
        crop_size: int,
        frames_per_sample=5,
        skip_frames=0,
        random_time=True,
        random_horizontal_flip=True,
        random_time_reverse=False,
        aug=False,
        albumentations=False,
        with_flows=False,
        total_videos=-1,
        num_steps=1,
    ):
        self.data_path = data_path
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.skip_frames = skip_frames
        self.random_horizontal_flip = random_horizontal_flip
        self.random_time_reverse = random_time_reverse
        self.total_videos = total_videos
        self.with_flows = with_flows
        self.num_steps = num_steps

        if self.random_time_reverse:
            assert (
                not self.with_flows
            ), "Random time reversal only applicable without precalculated flows"

        self.albumentations = albumentations

        self.input_size = input_size
        self.crop_size = crop_size

        self.aug = RandomConsistentAugFactory(aug)

        if self.albumentations:
            self.tr = alb.Compose(
                [
                    alb.SmallestMaxSize(max_size=self.input_size),
                    alb.CenterCrop(height=self.crop_size, width=self.crop_size),
                    # albumentations.HorizontalFlip(p=flip_p),
                ]
            )
        else:
            self.tr = T.Compose(
                [
                    T.Resize(size=self.input_size, antialias=True),
                    T.CenterCrop(size=self.crop_size),
                    # T.RandomHorizontalFlip(p=flip_p),
                ]
            )
        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        # print(f"Dataset length: {self.total_videos}")
        # print(f"Number of steps: {self.num_steps}")

    def __len__(self):
        return self.num_steps

    @property
    def num_videos(self):
        return self.total_videos if self.total_videos > 0 else len(self.videos_ds)

    def max_index(self):
        return len(self.videos_ds)

    def __getitem__(self, index, time_idx=0):
        video_index = round(
            (index % self.num_videos) / (self.num_videos - 1) * (self.max_index() - 1)
        )
        # video_index = round(index / (self.num_videos - 1) * (self.max_index() - 1))
        # shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

        # Setup augmentations
        color_tr = self.aug()

        prefinals = []
        flows = []
        with h5py.File(self.videos_ds.shard_paths[shard_idx], "r") as f:
            video_len = f["len"][str(idx_in_shard)][()]
            num_frames = (self.skip_frames + 1) * (self.frames_per_sample - 1) + 1
            assert (
                video_len >= num_frames
            ), "The video is shorter than the desired sample size"
            if self.random_time:
                time_idx = np.random.choice(video_len - num_frames)
            assert time_idx < video_len, "Time index out of video boundary"

            for i in range(
                time_idx, min(time_idx + num_frames, video_len), self.skip_frames + 1
            ):
                if "videos" in f:
                    img = f["videos"][str(idx_in_shard)][str(i)][()]
                else:
                    img = f[str(idx_in_shard)][str(i)][()]
                if self.albumentations:
                    arr = self.tr(image=img)["image"]
                else:
                    arr = img
                prefinals.append(torch.Tensor(arr).to(torch.uint8))

                if self.with_flows:
                    flow = f["flows"][str(idx_in_shard)][str(i)][()]

                    flow = torch.Tensor(flow).to(torch.float32)

                    flows.append(flow)

        data = torch.stack(prefinals)
        if not self.albumentations:
            data = self.tr(data.permute(0, 3, 1, 2))
        else:
            data = data.permute(0, 3, 1, 2)
        data = color_tr(data).to(torch.float32) / 127.5 - 1.0

        if self.random_time_reverse and np.random.randint(2) == 0:
            data = torch.flip(data, dims=[0])

        if self.with_flows:
            flows = torch.stack(flows)
            return data, flows

        return data


class VideoDataModule(LightningDataModule):
    """LightningDataModule for a video dataset."""

    def __init__(
        self,
        data_path: str,
        input_size: int,
        crop_size: int,
        frames_per_sample: int = 5,
        num_steps: int = 10000,
        skip_frames: int = 0,
        random_time: bool = True,
        random_horizontal_flip: bool = True,
        random_time_reverse: bool = False,
        aug: bool = False,
        albumentations: bool = False,
        with_flows: bool = False,
        total_videos: int = -1,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a VideoDataModule.

        :param data_path: Path to the dataset.
        :param input_size: Input size for the video frames.
        :param crop_size: Crop size for the video frames.
        :param frames_per_sample: Number of frames per video sample.
        :param skip_frames: Number of frames to skip between selected frames.
        :param random_time: Whether to randomly select the start time for sampling.
        :param random_horizontal_flip: Whether to apply random horizontal flip.
        :param random_time_reverse: Whether to randomly reverse the order of frames.
        :param aug: Whether to apply augmentations.
        :param albumentations: Whether to use Albumentations library for augmentations.
        :param with_flows: Whether to include optical flow data.
        :param total_videos: Total number of videos to include (-1 for all).
        :param batch_size: The batch size. Defaults to 32.
        :param num_workers: Number of workers for data loading. Defaults to 4.
        :param pin_memory: Whether to pin memory in DataLoader. Defaults to False.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Initialize the dataset with the provided parameters

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Prepare data if needed.

        This method is called only on one process.
        """
        # You can implement downloading or processing logic here if needed.
        pass

    def create_video_dataset(self, path):
        return VideoDataset(
            data_path=path,
            input_size=self.hparams.input_size,
            crop_size=self.hparams.crop_size,
            frames_per_sample=self.hparams.frames_per_sample,
            skip_frames=self.hparams.skip_frames,
            random_time=self.hparams.random_time,
            random_horizontal_flip=self.hparams.random_horizontal_flip,
            random_time_reverse=self.hparams.random_time_reverse,
            aug=self.hparams.aug,
            albumentations=self.hparams.albumentations,
            with_flows=self.hparams.with_flows,
            total_videos=self.hparams.total_videos,
            num_steps=self.hparams.num_steps,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training, validation, and testing."""
        # Split the dataset into train, val, and test sets

        # if stage == "fit" or stage is None:
        path = self.hparams.data_path + "/train"
        self.data_train = self.create_video_dataset(path)
        # elif stage == "validate" and not self.data_val:
        #    raise ValueError("Validation dataset not available")
        path = self.hparams.data_path + "/val"
        self.data_val = self.create_video_dataset(path)
        # elif stage == "test" and not self.data_test:
        # path = self.hparams.data_path + "/test"
        # self.data_test = self.create_video_dataset(path)

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Save datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load datamodule state."""
        pass
