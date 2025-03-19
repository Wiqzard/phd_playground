from typing import Optional
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

class MemoryMazeDataset(Dataset):
    def __init__(self, base_folder, num_frames, transform=None):
        """
        Args:
            base_folder (str): Path to the folder containing NPZ trajectory files.
            num_frames (int): Number of contiguous frames to sample from each trajectory.
            transform (callable, optional): Optional transform to apply on a sample.
        """
        self.base_folder = base_folder
        self.num_frames = num_frames
        self.transform = transform
        self.num_actions = 6 # 

        # Find all NPZ files in the base folder.
        self.files = sorted(glob.glob(os.path.join(base_folder, "*.npz")))
        if len(self.files) == 0:
            raise ValueError(f"No NPZ files found in {base_folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the trajectory (episode)
        file_path = self.files[idx]
        # np.load returns a NpzFile; we convert it to a dict.
        data = np.load(file_path)
        episode = {key: data[key] for key in data.keys()}
        data.close()

        # Assume that the 'image' key is dynamic and determines the trajectory length.
        total_length = episode["image"].shape[0]
        if total_length < self.num_frames:
            raise ValueError(
                f"num_frames ({self.num_frames}) is greater than trajectory length ({total_length}) in {file_path}"
            )
        # Randomly choose a starting index such that the segment is fully contained.
        start = np.random.randint(0, total_length - self.num_frames + 1)
        end = start + self.num_frames

        # For each key, if the first dimension matches the episode length,
        # assume it is a time series and sample the segment.
        sampled_episode = {}
        for key, value in episode.items():
            if value.ndim > 0 and value.shape[0] == total_length:
                sampled_episode[key] = value[start:end]
            else:
                # Static keys (e.g. maze_layout may be stored only once) are kept as-is.
                sampled_episode[key] = value

        # Optionally apply a transform.
        #if self.transform:

        # Convert numpy arrays to torch tensors.
        for key, value in sampled_episode.items():
            if isinstance(value, np.ndarray):
                sampled_episode[key] = torch.from_numpy(value)


        sampled_episode["video"]  = ((sampled_episode["image"]) / 255 * 2 - 1 ).permute(3, 0, 1, 2).to(torch.float32)
        sampled_episode["first_frame"] = sampled_episode["video"][:, 0, ...]
        del sampled_episode["image"]

        # do not take the last action
        sampled_episode["action"] = sampled_episode["action"][:-1,:].to(torch.long)
        return sampled_episode

class MemoryMazeDataModule(LightningDataModule):
    def __init__(self, base_folder, num_frames, transform=None, batch_size=32, num_workers=16):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=("transform",))

        self.transform = transform

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_folder = os.path.join(self.hparams.base_folder, "train")
            self.train_dataset = MemoryMazeDataset(train_folder , self.hparams.num_frames, self.transform)
            val_folder = os.path.join(self.hparams.base_folder, "eval")
            self.val_dataset = MemoryMazeDataset(val_folder, self.hparams.num_frames, self.transform)
        if stage == "test":
            test_folder = os.path.join(self.hparams.base_folder, "test")
            self.test_dataset = MemoryMazeDataset(test_folder, self.num_frames, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)


# Usage example:
if __name__ == "__main__":
    from torchvision import transforms
    # divide by 255, normalize, resize
    transform = None

    dataset = MemoryMazeDataset(base_folder="/data/cvg/sebastian/memory_maze/memory-maze-9x9/eval", num_frames=100,
                                transform=transform)
    sample = dataset[0]
    print(sample.keys())
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in loader:
        print(batch["video"].shape)
        break
    