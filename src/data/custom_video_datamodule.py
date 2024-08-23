import os
import random
from glob import glob
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class VideoFramesDataset(Dataset):
    """Initialize a `VideoFramesDataset`.

    :param root_dir: Root directory containing video frames.
    :param transform: Transform to be applied on a sample. Defaults to `None`.
    :param n_frames: Number of frames to retrieve in one sample. Defaults to `5`.
    :param skip_k: Number of frames to skip. Defaults to `1`.
    :param ext: Extension of the frame files. Defaults to `"png"`.
    :param random_video: Flag to sample random videos instead of sequential frames. Defaults to `False`.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        n_frames: int = 5,
        skip_k: int = 1,
        ext: str = "png",
        random_video: bool = False,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.n_frames = n_frames
        self.skip_k = skip_k
        self.ext = ext
        self.random_video = random_video

        self.video_dict = {}
        self.samples = []

        for root, _, files in os.walk(root_dir):
            frames = sorted([os.path.join(root, f) for f in files if f.endswith(ext)])
            if len(frames) >= n_frames * skip_k:
                self.video_dict[root] = frames
                num_samples = len(frames) // (n_frames * skip_k)
                self.samples.extend([(root, i) for i in range(num_samples)])

        if not self.samples:
            raise ValueError(f"No frames with extension {ext} found in {root_dir}")

    def __len__(self):
        """Return the total number of samples.

        :returns: The total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieve a sample from the dataset.

        :param idx: Index of the sample to retrieve.
        :returns: A tensor of stacked frames transformed and converted to tensors.
        :raises IndexError: If the index is out of range.
        :raises ValueError: If not enough frames are available in the video.
        :raises RuntimeError: If there is an error opening an image file.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        video_path, sample_idx = self.samples[idx]
        frames = self.video_dict[video_path]

        start_frame = sample_idx * self.n_frames * self.skip_k
        selected_frames = frames[
            start_frame : start_frame + self.n_frames * self.skip_k : self.skip_k
        ]

        images = []
        for frame in selected_frames:
            try:
                image = Image.open(frame).convert("RGB")
            except OSError as e:
                raise RuntimeError(f"Error opening image {frame}: {e}")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        return torch.stack(images)


class VideoFramesDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (7, 2, 1),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        n_frames: int = 15,
        skip_k: int = 1,
        transform_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if transform_cfg:
            self.transforms = hydra.utils.instantiate(transform_cfg)
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        # No download required for custom dataset
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = VideoFramesDataset(
                self.hparams.data_dir,
                transform=self.transforms,
                n_frames=self.hparams.n_frames,
                skip_k=self.hparams.skip_k,
            )
            total_size = len(dataset)
            train_size = int(
                total_size
                * self.hparams.train_val_test_split[0]
                / sum(self.hparams.train_val_test_split)
            )
            val_size = int(
                total_size
                * self.hparams.train_val_test_split[1]
                / sum(self.hparams.train_val_test_split)
            )
            test_size = total_size - train_size - val_size

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    # _ = VideoFramesDataModule()

    d = VideoFramesDataModule(
        data_dir="/var/tmp/europe_videos_converted",
        skip_k=50,
        train_val_test_split=[8, 2, 0],
    )
    d.setup()
    dd = d.train_dataloader()
    from tqdm import tqdm

    for i, data in tqdm(enumerate(dd)):
        continue
    print(1)
