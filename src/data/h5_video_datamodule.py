import glob
import os
import random
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from lightning import LightningDataModule


class H5VideoDataset(Dataset):
    @staticmethod
    def _get_num_in_shard(shard_p):
        print(f"\rh5: Opening {shard_p}... ", end="")
        try:
            with h5py.File(shard_p, "r") as f:
                num_per_shard = len(f["len"].keys())
        except Exception as e:
            print(f"h5: Could not open {shard_p}! Exception: {e}")
            num_per_shard = -1
        return num_per_shard

    @staticmethod
    def check_shard_lengths(file_paths):
        """
        Filter away the last shard, which is assumed to be smaller. this double checks that all other shards have the
        same number of entries.
        :param file_paths: list of .hdf5 files
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        shard_lengths = []
        print("Checking shard_lengths in", file_paths)
        for i, p in enumerate(file_paths):
            shard_lengths.append(H5VideoDataset._get_num_in_shard(p))
        return shard_lengths

    def __init__(
        self,
        base_folder: str,
        width=1024,
        height=576,
        sample_frames=25,
        skip_frames=0,
        max_steps=None,
    ):
        self.base_folder = base_folder
        self.sample_frames = sample_frames
        self.channels = 3
        self.height = height
        self.width = width
        self.skip_frames = skip_frames
        self.max_steps = max_steps

        self.resize_transform = T.Resize((self.height, self.width))
        assert self.channels == 3, "h5: Only 3 channels supported!"

        assert os.path.isdir(self.base_folder)

        self.data_dir = self.base_folder
        self.shard_paths = sorted(
            glob.glob(os.path.join(self.data_dir, "*.hdf5"))
            + glob.glob(os.path.join(self.data_dir, "*.h5"))
        )

        assert len(self.shard_paths) > 0, (
            "h5: Directory does not have any .hdf5 files! Dir: " + self.data_dir
        )

        self.shard_lengths = H5VideoDataset.check_shard_lengths(self.shard_paths)
        self.num_per_shard = self.shard_lengths[0]
        self.total_num = sum(self.shard_lengths)

        assert len(self.shard_paths) > 0, (
            "h5: Could not find .hdf5 files! Dir: "
            + self.data_dir
            + " ; len(self.shard_paths) = "
            + str(len(self.shard_paths))
        )

        self.num_of_shards = len(self.shard_paths)

        print(
            "h5: paths",
            len(self.shard_paths),
            "; shard_lengths",
            self.shard_lengths,
            "; total",
            self.total_num,
        )

    def __len__(self):
        if self.max_steps is not None:
            return self.max_steps
        return self.total_num

    def get_indices(self, idx):
        shard_idx = np.digitize(idx, np.cumsum(self.shard_lengths))
        idx_in_shard = str(idx - sum(self.shard_lengths[:shard_idx]))
        return shard_idx, idx_in_shard

    def __getitem__(self, index):
        idx = index % self.total_num
        shard_idx, idx_in_shard = self.get_indices(idx)
        pixel_values = torch.empty(
            (self.sample_frames, self.channels, self.height, self.width)
        )

        # Read from shard
        with h5py.File(self.shard_paths[shard_idx], "r") as f:
            num_frames = len(f[idx_in_shard])
            start_idx = random.randint(
                0, num_frames - self.sample_frames * (self.skip_frames + 1)
            )

            # Adjusted loop to prevent index errors
            frame_indices = range(
                start_idx,
                start_idx + self.sample_frames * (self.skip_frames + 1),
                self.skip_frames + 1,
            )
            frame_indices = frame_indices[
                : self.sample_frames
            ]  # Ensure only self.sample_frames are used

            for j, i in enumerate(frame_indices):
                img_tensor = torch.from_numpy(f[idx_in_shard][str(i)][:]).float()
                img_tensor = img_tensor / 127.5 - 1
                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = self.resize_transform(img_tensor)
                pixel_values[j] = img_tensor

        return {"pixel_values": pixel_values}


class RandomImageDataset(Dataset):
    def __init__(self, base_val_path):
        folders = []
        for root, dirs, files in os.walk(base_val_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                folders.append(dir_path)

        self.all_val_paths = []
        for folder in folders:
            folder_path = os.path.join(base_val_path, folder)
            files = os.listdir(folder_path)
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    self.all_val_paths.append(os.path.join(folder_path, file))

    def __len__(self):
        return len(self.all_val_paths)

    def __getitem__(self, idx):
        # Randomly select an image path
        img_path = random.choice(self.all_val_paths)
        # Load the image
        return img_path

    def collate_fn(batch):
        # return list of strings/ paths
        return batch


class H5VideoDataModule(LightningDataModule):
    """LightningDataModule for a video dataset."""

    def __init__(
        self,
        data_path: str,
        width: int,
        height: int,
        skip_frames: int,
        frames_per_sample: int = 5,
        num_steps: int = 10000,
        aug: bool = False,
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
        :param aug: Whether to apply augmentations.
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
        return H5VideoDataset(
            base_folder=path,
            width=self.hparams.width,
            height=self.hparams.height,
            sample_frames=self.hparams.sample_frames,
            skip_frames=self.hparams.skip_frames,
            max_steps=self.hparams.num_steps,
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
