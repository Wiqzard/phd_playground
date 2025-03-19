import json
import os
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from tqdm import tqdm


class BasaltFrameDataset(Dataset):
    """
    Loads the preprocessed frames from each video folder and corresponding actions
    (from a .json file), letting you sample clips or single frames without re-decoding the .mp4.
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 8,
        overlap: int = 0,
        image_ext: str = ".jpg",
        transform=None,
    ):
        """
        :param root_dir: Directory produced by preprocess.py, e.g. "processed/".
        :param seq_len: Number of consecutive frames in each sample.
        :param overlap: Overlap between consecutive clips.
        :param image_ext: Extension used for stored frames (".jpg", ".png", etc.).
        :param transform: Transforms to apply to each loaded frame (e.g. ToTensor, Resize, etc.).
        """
        super().__init__()
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.image_ext = image_ext
        self.transform = transform if transform else transforms.ToTensor()
        self.stride = max(1, seq_len - overlap)

        # We'll store a list of clips (video_folder, start_frame_idx)
        self.clips: List[Tuple[str, int, int]] = []  # (video_folder, start_frame, total_frames)
        # Also store a parallel list of actions_file for each video_folder
        self.video_actions_map: Dict[str, str] = {}  # "video_folder" -> "actions_path.json"

        self._enumerate_clips()

    def _enumerate_clips(self):
        """Scan root_dir for subfolders (videos), find number of frames, create clips."""
        # e.g. root_dir/ENV_NAME/VIDEO_NAME/...
        for env_name in os.listdir(self.root_dir):
            env_path = os.path.join(self.root_dir, env_name)
            if not os.path.isdir(env_path):
                continue

            # each subfolder in env_path is a video name
            videos = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
            for vid_name in tqdm(videos):
                vid_folder = os.path.join(env_path, vid_name)
                frames_list = sorted(
                    [f for f in os.listdir(vid_folder) if f.endswith(self.image_ext)]
                )
                # e.g. ["frame_000000.jpg", "frame_000001.jpg", ...]

                total_frames = len(frames_list)
                if total_frames == 0:
                    continue

                # Also find the actions.json for this video
                # we expect it at env_path/vid_name_actions.json or similar
                actions_path = os.path.join(env_path, vid_name + "_actions.json")
                if not os.path.exists(actions_path):
                    # maybe fallback or skip
                    continue
                self.video_actions_map[vid_folder] = actions_path

                # Now create all possible clips for this video
                start = 0
                while start + self.seq_len <= total_frames:
                    self.clips.append((vid_folder, start, total_frames))
                    start += self.stride

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a dictionary with:
            'video'   -> [seq_len, C, H, W] frames
            'actions' -> (parsed from JSON) relevant to these frames
        """
        vid_folder, start, total_frames = self.clips[idx]
        actions_path = self.video_actions_map[vid_folder]

        # 1) Load the action array (cached? or loaded once if large?)
        with open(actions_path, "r") as f:
            all_actions = json.load(f)

        # 2) Collect frames from start -> start+seq_len
        frames = []
        for i in range(self.seq_len):
            frame_idx = start + i
            frame_path = os.path.join(vid_folder, f"frame_{frame_idx:06d}{self.image_ext}")
            # load image
            img = Image.open(frame_path)
            img = img.convert("RGB")  # ensure 3 channels
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        video_tensor = torch.stack(frames, dim=0)  # shape [seq_len, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # shape [C, seq_len, H, W]
        # plt.imsave(f"frame_{frame_idx:06d}_first.jpg", video_tensor[0].permute(1, 2, 0).numpy())
        # safe self.clips as pickle

        # 3) Extract the corresponding actions
        #clip_actions = all_actions[start : start + self.seq_len]
        #clip_actions = [
        #    {"x": action["mouse"]["x"],
        #     "y": action["mouse"]["y"]} for action in clip_actions]

        #assert len(clip_actions) == self.seq_len, f"Expected {self.seq_len} actions, got {len(clip_actions)}"

        return {
            "video": video_tensor,
        #    "actions": clip_actions,
            "video_folder": vid_folder,
            "start_frame": start,
        }


class BasaltDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 8,
        overlap: int = 2,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_ext: str = ".jpg",
        height: int = 128,
        width: int = 128,
        transform: Optional[Any] = None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_ext = image_ext
        self.height = height
        self.width = width

        self.transform = transform or transforms.Compose(
            [
                transforms.CenterCrop(360),
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

            ]
        )

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def prepare_data(self) -> None:
        """No downloading required, but add any preprocessing or data checks here if needed."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Split the dataset into train, validation, and test sets."""

        #self.full_dataset = BasaltFrameDataset(
        #        root_dir=self.data_dir,
        #        seq_len=self.seq_len,
        #        overlap=self.overlap,
        #        image_ext=self.image_ext,
        #        transform=self.transform,
        #    )

        if stage == "fit" or stage is None:
            full_dataset = BasaltFrameDataset(
                root_dir=self.data_dir,
                seq_len=self.seq_len,
                overlap=self.overlap,
                image_ext=self.image_ext,
                transform=self.transform,
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.dataset_train, self.dataset_val = random_split(
                full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test" or stage is None:
            self.dataset_test = BasaltFrameDataset(
                root_dir=self.data_dir,
                seq_len=self.seq_len,
                overlap=self.overlap,
                image_ext=self.image_ext,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            #self.full_dataset,
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            #self.full_dataset,
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        ) 

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit, validation, or test."""
        pass

    def state_dict(self) -> dict:
        """Save the state of the data module."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the data module."""
        pass


if __name__ == "__main__":

    data_module = BasaltDataModule(
        data_dir="/data/cvg/sebastian/minecraft_processed",
        seq_len=8,
        overlap=2,
        batch_size=9,
        num_workers=4,
        pin_memory=True,
        height=128,
        width=128,
        transform=None,
    )

    data_module.prepare_data()
    data_module.setup("fit")

    for batch in data_module.train_dataloader():
        print(batch["video"].shape, len(batch["actions"]))
        break

    # Example usage

    # Example with transforms
    # resize and crop to square, then normalize to [-1, 1], totensor

    transform = transforms.Compose(
        [
            transforms.CenterCrop(360),
            transforms.Resize(128),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


    dataset = BasaltFrameDataset(
        "/data/cvg/sebastian/minecraft_processed", seq_len=8, overlap=2, transform=transform
    )
    sample = dataset[0]
    print(sample["video"].shape, len(sample["actions"]))
