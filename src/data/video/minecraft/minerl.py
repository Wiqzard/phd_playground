import os
import subprocess
import multiprocessing
import json
import cv2
import torch
import random
from typing import Any, Dict, List, Tuple, Optional
from torch.utils.data import Dataset


def get_frame_count_ffprobe(video_path: str) -> int:
    """
    Run ffprobe to get the total frame count of a video file.
    Returns 0 on error.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-print_format",
        "csv",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        # The output is typically: "stream,nb_read_frames,<number_of_frames>"
        # e.g., "stream,nb_read_frames,1234"
        output = result.stdout.strip()
        frame_count_str = output.split(",")[-1]
        return int(frame_count_str)
    except Exception as e:
        print(f"ffprobe error on {video_path}: {e}")
        return 0


class BasaltClipsDataset(Dataset):
    """
    A dataset where each index corresponds to a specific clip of length `seq_len`
    from a given video. Clips can overlap based on `stride`.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 8,
        overlap: int = 0,
        video_extensions: Tuple[str] = (".mp4",),
        transform=None,
        use_ffprobe: bool = True,
    ):
        """
        :param data_dir: Root directory with subfolders (MineRLBasaltBuildVillageHouse-v0, etc.)
        :param seq_len: Number of consecutive frames per clip.
        :param overlap: Number of overlapping frames between consecutive clips.
            - If overlap=0, then clips are consecutive with no overlap.
            - If overlap>0, there is a partial overlap.
        :param video_extensions: Valid extensions to consider as videos.
        :param transform: Optional transform to apply to each frame (e.g. ToTensor, Resize, etc.).
        :param use_ffprobe: Whether to use ffprobe to get frame counts (faster in parallel),
            otherwise uses OpenCV to get frame counts.
        """
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.overlap = overlap
        self.transform = transform
        self.video_extensions = video_extensions
        self.use_ffprobe = use_ffprobe

        # The "stride" = how many frames we move forward to get the next clip start
        # So stride = seq_len - overlap
        self.stride = seq_len - overlap if (seq_len - overlap) > 0 else 1

        # We'll store clip metadata here
        # self.clips is a list of tuples:
        # (video_path, json_path, start_frame)
        self.clips: List[Tuple[str, str, int]] = []

        # 1) Gather all videos + JSON files
        self.video_json_pairs: List[Tuple[str, str]] = []
        self._find_video_json_pairs()

        # 2) Get frame counts (in parallel if using ffprobe)
        self.video_frame_counts: List[int] = self._get_all_frame_counts()

        # 3) Build the list of all possible clips from each video
        self._enumerate_all_clips()

    def _find_video_json_pairs(self):
        """Collect all (video_path, json_path) pairs in self.video_json_pairs."""
        for env_dir in os.listdir(self.data_dir):
            env_path = os.path.join(self.data_dir, env_dir)
            if not os.path.isdir(env_path):
                continue

            files_in_env = os.listdir(env_path)
            video_files = [f for f in files_in_env if f.endswith(self.video_extensions)]
            for vf in video_files:
                video_path = os.path.join(env_path, vf)
                json_path = video_path.replace(".mp4", ".jsonl")  # or .mp4 => .json
                if os.path.exists(json_path):
                    self.video_json_pairs.append((video_path, json_path))

    def _get_all_frame_counts(self) -> List[int]:
        """
        Return a list of frame counts corresponding to self.video_json_pairs.
        For large datasets, we parallelize ffprobe calls; otherwise, we can use cv2.
        """
        if not self.video_json_pairs:
            return []

        if self.use_ffprobe:
            # Use ffprobe in parallel
            with multiprocessing.Pool() as pool:
                frame_counts = pool.map(
                    get_frame_count_ffprobe, [v[0] for v in self.video_json_pairs]
                )
        else:
            # Use OpenCV (slower, single-threaded)
            frame_counts = []
            for video_path, _ in self.video_json_pairs:
                cap = cv2.VideoCapture(video_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                frame_counts.append(n_frames)

        return frame_counts

    def _enumerate_all_clips(self):
        """
        For each video, create all possible clips of length seq_len (with self.stride).
        Store them in self.clips as (video_path, json_path, start_frame).
        """
        for i, (video_path, json_path) in enumerate(self.video_json_pairs):
            n_frames = self.video_frame_counts[i]
            if n_frames < self.seq_len:
                # Not enough frames to make even one clip
                continue

            # Slide over the video with "stride", enumerating start indices
            # up to (n_frames - seq_len).
            start = 0
            while start + self.seq_len <= n_frames:
                self.clips.append((video_path, json_path, start))
                start += self.stride

    def __len__(self) -> int:
        """Number of total clips across all videos."""
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load the clip (seq_len frames) starting at `start_frame` from the video at `video_path`.
        Also load the aligned JSON actions from the same frame range.
        Return them in a dict, e.g.:
          { 'video': Tensor of shape [seq_len, C, H, W],
            'actions': <whatever structure you want> }
        """
        video_path, json_path, start_frame = self.clips[idx]

        # 1) Load actions from JSON
        with open(json_path, "r") as f:
            json_lines = [json.loads(line) for line in f]
        # (You might want to handle mismatch if len(json_lines) != n_frames.)

        # We want frames [start_frame, start_frame + seq_len).
        actions = json_lines[start_frame : start_frame + self.seq_len]

        # 2) Load frames from video
        frames = []
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                # If the read fails unexpectedly, break early or handle it
                break
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        # 3) Stack frames into a single tensor: [seq_len, C, H, W]
        video_tensor = torch.stack(frames, dim=0)  # shape: (seq_len, C, H, W)

        # 4) Construct your chosen action representation
        # As an example, let's store the raw dictionary:
        # (In practice, parse mouse dx/dy, keys, etc.)
        out_dict = {
            "video": video_tensor,
            "actions": actions,  # or parse them into numeric tensors
        }
        return out_dict


if __name__ == "__main__":
    dataset = BasaltClipsDataset(
        data_dir="/data/cvg/sebastian/minecraft",
        seq_len=8,
        overlap=4,  # e.g. half overlap
        transform=None,  # or transforms.ToTensor() etc.
        use_ffprobe=True,  # requires ffprobe installed
    )

    print("Number of clips:", len(dataset))
    sample = dataset[0]
    print("Video clip shape:", sample["video"].shape)  # (seq_len, C, H, W)
    print("Number of actions in clip:", len(sample["actions"]))
