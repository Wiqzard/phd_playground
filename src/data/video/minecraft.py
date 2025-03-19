from typing import Any, Dict, Optional, List
import io
import tarfile
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
)

class MinecraftBaseVideoDataset(BaseVideoDataset):
    """
    Base class for Minecraft video datasets.
    Handles dataset download + specialized transforms.
    """

    _ALL_SPLITS = ["training", "validation"]  # we rename test -> validation

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: str = "training",
    ):
        """
        Args:
            save_dir: Path to the Minecraft dataset directory (or where to download).
            resolution: Spatial resolution (e.g. 256).
            latent_downsampling_factor: [temporal_down, spatial_down].
            latent_suffix: Extra string for the latent directory name.
            split: "training", "validation" (or "test" which we’ll map to "validation").
        """
        super().__init__(
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
        )

    def download_dataset(self):
        """
        Example logic that downloads multi-part archives from archive.org,
        concatenates them, and extracts them into save_dir.
        Then it renames "test" -> "validation" for consistency.
        """
        from internetarchive import download

        part_suffixes = ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak"]

        # 1) Download each .part file
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            download(identifier, file_name, destdir=self.save_dir, verbose=True)

        # 2) Concatenate all .part files
        combined_bytes = io.BytesIO()
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_file = Path(self.save_dir) / identifier / file_name
            with open(part_file, "rb") as part:
                combined_bytes.write(part.read())
        combined_bytes.seek(0)

        # 3) Extract final tar
        with tarfile.open(fileobj=combined_bytes, mode="r") as combined_archive:
            combined_archive.extractall(self.save_dir)

        # 4) Move /minecraft/test -> /minecraft/validation, etc.
        (Path(self.save_dir) / "minecraft" / "test").rename(
            Path(self.save_dir) / "validation"
        )
        (Path(self.save_dir) / "minecraft" / "train").rename(
            Path(self.save_dir) / "training"
        )
        (Path(self.save_dir) / "minecraft").rmdir()

        # 5) Clean up parts
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_dir = Path(self.save_dir) / identifier
            part_file = part_dir / file_name
            if part_file.exists():
                part_file.unlink()
            part_dir.rmdir()

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        """
        In this dataset, we know each .mp4 is 300 frames.
        """
        return 300

    def build_transform(self):
        """
        Use nearest-neighbor upsampling to match the specified resolution.
        """
        return transforms.Resize(
            (self.resolution, self.resolution),
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )


class MinecraftSimpleVideoDataset(MinecraftBaseVideoDataset, BaseSimpleVideoDataset):
    """
    Minecraft dataset that loads entire videos (simple version).
    """

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: str = "training",
    ):
        # Force 'test' to be treated as 'validation'
        if split == "test":
            split = "validation"

        # We directly call each parent’s __init__
        # so that the shared BaseVideoDataset logic is invoked once in each parent.
        MinecraftBaseVideoDataset.__init__(
            self,
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
        )
        BaseSimpleVideoDataset.__init__(
            self,
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
        )


class MinecraftAdvancedVideoDataset(MinecraftBaseVideoDataset, BaseAdvancedVideoDataset):
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
    ):
        # Force 'test' to be treated as 'validation'
        if split == "test":
            split = "validation"

        # 1) Call MinecraftBaseVideoDataset’s __init__
        MinecraftBaseVideoDataset.__init__(
            self,
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
        )

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
        )

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        """
        Example: load discrete actions from a .npz file, one-hot-encode them,
        and return shape [T, 4].
        """
        path = video_metadata["video_paths"].with_suffix(".npz")
        actions = np.load(path)["actions"][start_frame:end_frame]
        # E.g. 4 possible actions => shape [T, 4]
        return torch.from_numpy(np.eye(4)[actions]).float()

if __name__ == "__main__":
    import hydra
    path = "/home/ss24m050/Documents/phd_playground/configs/data/minecraft.yaml"
    dataset = hydra.utils.instantiate(path)
    print(0)


