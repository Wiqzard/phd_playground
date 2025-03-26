import os
import glob
import json
import random

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

# -------------
# Original imports and constants
# -------------

# from openai_vpt.agent import ACTION_TRANSFORMER_KWARGS, resize_image, AGENT_RESOLUTION
# from openai_vpt.lib.actions import ActionTransformer

# Original constants
MINEREC_ORIGINAL_HEIGHT_PX = 720
CAMERA_SCALER = 360.0 / 2400.0
QUEUE_TIMEOUT = 10

# Hard-code path or place "cursors/mouse_cursor_white_16x16.png" somewhere
CURSOR_FILE = os.path.join(
    os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png"
)

# For adjusting camera scaling if GUI is open; simplified (using default “1.0” if not present)
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}


def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.
    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y : y + ch, x : x + cw, :] = (
        image1[y : y + ch, x : x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha
    ).astype(np.uint8)


def json_action_to_env_action(json_action):
    """
    Convert a single JSON action line from .jsonl to a MineRL action dict.
    Returns (env_action, is_null_action)
    """
    env_action = NOOP_ACTION.copy()
    env_action["camera"] = np.array([0, 0])  # reset camera each time

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER
    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        # If extremely large camera movement, ignore (sometimes weird from raw data)
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


# -------------
# MineRLVideoDataset
# -------------
class MineRLVideoDataset(Dataset):
    """
    A "standard" PyTorch Dataset that:
      1) Finds all .mp4 and corresponding .jsonl in dataset_dir.
      2) Parses each .jsonl to build a list of "non-null" actions.
      3) For each non-null action, store its (video_path, frame_idx, action_dict, isGuiOpen, mouse_x, mouse_y).
      4) __getitem__ loads exactly that frame from the video, composites the cursor if needed, transforms to agent res.
      5) Returns (transformed_frame, agent_action) or (transformed_frame, env_action) depending on your preference.

    NOTE: This approach re-opens the video in __getitem__ each time, which is simpler code
          but can be slow for large scale. Consider more efficient caching or pre-computed .npz for real usage.
    """

    def __init__(self, dataset_dir, transform_actions_to_agent=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform_actions_to_agent = transform_actions_to_agent

        # We'll need the cursor if we want to overlay it
        self.cursor_image_bgr = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
        if self.cursor_image_bgr is None:
            raise FileNotFoundError(
                f"Cursor image not found at {CURSOR_FILE}. "
                "Make sure it exists or update path."
            )
        self.cursor_alpha = self.cursor_image_bgr[:, :, 3:] / 255.0
        self.cursor_image_bgr = self.cursor_image_bgr[:, :, :3]

        # Prepare an ActionTransformer if we want to get the "agent" actions
        self.action_transformer = None
        if transform_actions_to_agent:
            self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        # 1) Gather all demonstration file pairs: (video_path, json_path)
        #    Each "unique_id" is the file basename.
        mp4_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))
        unique_ids = list(set(os.path.basename(x).split(".")[0] for x in mp4_files))
        # Build pairs
        demonstration_tuples = []
        for uid in unique_ids:
            video_path = os.path.join(dataset_dir, uid + ".mp4")
            json_path = os.path.join(dataset_dir, uid + ".jsonl")
            if os.path.exists(video_path) and os.path.exists(json_path):
                demonstration_tuples.append((video_path, json_path))

        # 2) Parse the .jsonl and build up a list of valid steps
        self.samples = []
        for video_path, json_path in demonstration_tuples:
            with open(json_path, "r") as f:
                json_lines = f.readlines()
            # Convert lines to JSON
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)

            # Keep track of hotbar for each step (scroll is not recorded)
            last_hotbar = 0
            # Attack stuck detection
            attack_is_stuck = False

            frame_counter = 0
            for i, step_data in enumerate(json_data):
                # Attack stuck logic
                if i == 0:
                    if step_data["mouse"]["newButtons"] == [0]:
                        attack_is_stuck = True
                elif attack_is_stuck:
                    if 0 in step_data["mouse"]["newButtons"]:
                        attack_is_stuck = False

                if attack_is_stuck:
                    # Remove mouse button 0
                    step_data["mouse"]["buttons"] = [
                        b for b in step_data["mouse"]["buttons"] if b != 0
                    ]

                # Convert JSON action -> env action
                env_action, is_null_action = json_action_to_env_action(step_data)

                # Update hotbar if changed
                current_hotbar = step_data["hotbar"]
                if current_hotbar != last_hotbar:
                    env_action[f"hotbar.{current_hotbar + 1}"] = 1
                last_hotbar = current_hotbar

                # Skip if "null action"
                if is_null_action:
                    # We still read the frame to keep "frame_counter" in sync
                    frame_counter += 1
                    continue

                # We have a valid sample => store
                self.samples.append(
                    {
                        "video_path": video_path,
                        "frame_idx": frame_counter,
                        "env_action": env_action,
                        "isGuiOpen": step_data["isGuiOpen"],
                        "mouse_x": step_data["mouse"]["x"],
                        "mouse_y": step_data["mouse"]["y"],
                    }
                )
                frame_counter += 1

        print(
            f"Found total of {len(self.samples)} valid (non-null) frames across {len(demonstration_tuples)} demos."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        video_path = sample_meta["video_path"]
        frame_idx = sample_meta["frame_idx"]
        env_action = sample_meta["env_action"]
        isGuiOpen = sample_meta["isGuiOpen"]
        mouse_x = sample_meta["mouse_x"]
        mouse_y = sample_meta["mouse_y"]

        # Optionally convert env_action -> agent_action
        if self.action_transformer is not None:
            # The ActionTransformer expects env_action as a dict of numpy arrays or ints
            # That matches what we have. This returns a 1D or 2D array representing the action in agent space.
            agent_action = self.action_transformer(env_action)
            # We'll output that as e.g. a torch.tensor
        else:
            agent_action = env_action  # remain as env-action

        # 3) Load the specific frame from the .mp4
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret or frame_bgr is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}.")

        # 4) If isGuiOpen, composite the mouse cursor
        if isGuiOpen:
            # The frame might be 720p, but sometimes the version is different.
            # We'll ignore version-specific scalers for brevity and just do a ratio:
            camera_scaling_factor = frame_bgr.shape[0] / float(
                MINEREC_ORIGINAL_HEIGHT_PX
            )
            cursor_x = int(mouse_x * camera_scaling_factor)
            cursor_y = int(mouse_y * camera_scaling_factor)
            composite_images_with_alpha(
                frame_bgr, self.cursor_image_bgr, self.cursor_alpha, cursor_x, cursor_y
            )

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Resize to agent input resolution
        frame_rgb = resize_image(frame_rgb, AGENT_RESOLUTION)  # default 128x128

        # 5) Convert to torch (C,H,W) if desired
        #    Note: VPT code uses (H, W, C); here we transform to typical PyTorch (C, H, W).
        frame_rgb = np.transpose(frame_rgb, (2, 0, 1))  # (C,H,W)
        frame_tensor = torch.from_numpy(
            frame_rgb
        ).float()  # [0..255] => up to you to normalize

        # Convert the action to a torch tensor as well, for your pipeline
        if isinstance(agent_action, dict):
            # For demonstration, let's just store it as a dictionary of torch.Tensors
            # You might want to flatten or keep it as-is
            final_action = {}
            for k, v in agent_action.items():
                if isinstance(v, np.ndarray):
                    final_action[k] = torch.from_numpy(v).float()
                else:
                    final_action[k] = torch.tensor(v, dtype=torch.float32)
        else:
            # If the ActionTransformer gave a single array, just convert
            final_action = torch.tensor(agent_action, dtype=torch.float32)

        return frame_tensor, final_action


# -------------
# LightningDataModule
# -------------
class MineRLDataModule(LightningDataModule):
    """
    Simple LightningDataModule wrapping the above dataset.
    Assumes you want only a single dataset for demonstration.
    You could add train/val/test splits as needed.
    """

    def __init__(
        self,
        dataset_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        transform_actions_to_agent: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_actions_to_agent = transform_actions_to_agent

    def setup(self, stage=None):
        # For simplicity, assume the entire dataset is "train"
        self.dataset = MineRLVideoDataset(
            dataset_dir=self.dataset_dir,
            transform_actions_to_agent=self.transform_actions_to_agent,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    # If you want val/test splits, define them similarly
    # def val_dataloader(self):
    #     ...
    # def test_dataloader(self):
    #     ...


# -------------
# Usage Example
# -------------
if __name__ == "__main__":
    # Suppose you have the data in "path/to/my_dataset"
    data_module = MineRLDataModule(
        dataset_dir="path/to/my_dataset",
        batch_size=4,
        num_workers=2,
        transform_actions_to_agent=True,
    )
    data_module.setup()

    # Example: retrieve one batch from the train_dataloader
    loader = data_module.train_dataloader()
    for frames, actions in loader:
        print("frames.shape:", frames.shape)  # (B, C, H, W)
        print("actions:", actions)  # dict of Tensors or single Tensor
        break

    # Then plug data_module into your Lightning Trainer:
    # trainer = pl.Trainer(gpus=1, max_epochs=5)
    # trainer.fit(your_model, datamodule=data_module)
