from typing import Sequence, List, Dict, Any, Optional
import tarfile
import io

import numpy as np
import torch
from torch.utils.data import DataLoader
from internetarchive import download

from src.data.video.base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
)


class MinecraftBaseVideoDataset(BaseVideoDataset):
    def download_dataset(self) -> Sequence[int]:
        part_suffixes = [
            "aa",
            "ab",
            "ac",
            "ad",
            "ae",
            "af",
            "ag",
            "ah",
            "ai",
            "aj",
            "ak",
        ]
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            download(identifier, file_name, destdir=self.save_dir, verbose=True)

        combined_bytes = io.BytesIO()
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_file = self.save_dir / identifier / file_name
            with open(part_file, "rb") as part:
                combined_bytes.write(part.read())
        combined_bytes.seek(0)

        with tarfile.open(fileobj=combined_bytes, mode="r") as combined_archive:
            combined_archive.extractall(self.save_dir)

        (self.save_dir / "minecraft/test").rename(self.save_dir / "validation")
        (self.save_dir / "minecraft/train").rename(self.save_dir / "training")
        (self.save_dir / "minecraft").rmdir()

        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_file = self.save_dir / identifier / file_name
            part_file.unlink()

    def get_data_paths(self, split):
        data_dir = self.save_dir / split
        return sorted(list(data_dir.glob("**/*.npz")), key=lambda x: x.name)

    def __getitem__(self, idx):
        idx = idx
        file_idx, frame_idx = self.split_idx(idx)
        video = self.load_video(self.data_paths[file_idx])
        video = video[frame_idx : frame_idx + self.n_frames]
        return torch.from_numpy(video / 255.0).float()


class MinecraftSimpleVideoDataset(MinecraftBaseVideoDataset, BaseSimpleVideoDataset):
    """
    Minecraft simple video dataset
    """

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str,
        split: str = "training",
    ):
        BaseSimpleVideoDataset.__init__(
            self, save_dir, resolution, latent_downsampling_factor, latent_suffix, split
        )


class MinecraftAdvancedVideoDataset(
    MinecraftBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    Minecraft advanced video dataset
    """

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str,
        split: str = "training",
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
    ):
        if split == "test":
            split = "validation"
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
        )

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"].with_suffix(".npz")
        actions = np.load(path)["actions"][start_frame:end_frame]
        return torch.from_numpy(np.eye(4)[actions]).float()


# Usage example
if __name__ == "__main__":
    dataset = MinecraftSimpleVideoDataset(
        save_dir="data",
        resolution=64,
        latent_downsampling_factor=[1, 1, 1, 1],
        latent_suffix="",
        split="training",
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader.train_dataloader():
        print(batch)
