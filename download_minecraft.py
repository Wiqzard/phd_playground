import torch
from typing import Sequence
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import tarfile
import io
from internetarchive import download

from src.data.video.base_video import BaseVideoDataset, BaseSimpleVideoDataset
#from base_video_dataset import BaseVideoDataset


class MinecraftVideoDataset(BaseVideoDataset):
    def download_dataset(self) -> Sequence[int]:
        part_suffixes = [
            "aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak"
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

#class MinecraftSimpleVideoDataset(MinecraftVideoDataset, BaseSimpleVideoDataset):
#    """
#    Minecraft simple video dataset
#    """
#
#    def __init__(self, 
#
#        if split == "test":
#            split = "validation"
#        BaseSimpleVideoDataset.__init__(self, cfg, split)


class MinecraftDataModule(LightningDataModule):
    def __init__(self, save_dir, resolution, external_cond_dim, n_frames, frame_skip, validation_multiplier, batch_size):
        super().__init__()
        self.save_dir = save_dir
        self.resolution = resolution
        self.external_cond_dim = external_cond_dim
        self.n_frames = n_frames
        self.frame_skip = frame_skip
        self.validation_multiplier = validation_multiplier
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = MinecraftVideoDataset(
                self.save_dir, self.resolution, latent_downsampling_factor=[1,1], latent_suffix="jksd2ea",split="training"
            )
            self.val_dataset = MinecraftVideoDataset(
                self.save_dir, self.resolution, latent_downsampling_factor=[1,1], latent_suffix="jksd2ea",split="validation"
            )
        if stage == "test":
            self.test_dataset = MinecraftVideoDataset(
                self.save_dir, self.resolution, self.external_cond_dim, self.n_frames, self.frame_skip, self.validation_multiplier, "validation"
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)


# Usage example
if __name__ == "__main__":
    data_module = MinecraftDataModule(
        save_dir="/data/cvg/sebastian/minecraft_marsh",
        resolution=64,
        external_cond_dim=0,
        n_frames=64,
        frame_skip=2,
        validation_multiplier=1,
        batch_size=4
    )
    data_module.setup()


    for batch in data_module.train_dataloader():
        print(batch)
