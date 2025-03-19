import os
import json
import cv2
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

def process_single_video(
    video_info: dict,
    image_ext: str,
    overwrite: bool,
):
    """
    Process a single video: 
      - read frames from .mp4
      - skip frames to adjust FPS
      - resize if desired
      - save frames to disk
      - save aligned actions to JSON

    :param video_info: dictionary containing:
        {
          "env_dir": str,
          "env_path": str,
          "video_file": str,
          "out_env_path": str,
          "video_ext": str,
          "desired_fps": float or None,
          "desired_size": (width, height) or None,
        }
    :param image_ext: output extension for frames (e.g. ".jpg")
    :param overwrite: whether to overwrite existing folders
    """
    env_dir = video_info["env_dir"]
    env_path = video_info["env_path"]
    video_file = video_info["video_file"]
    out_env_path = video_info["out_env_path"]
    video_ext = video_info["video_ext"]
    desired_fps = video_info["desired_fps"]
    desired_size = video_info["desired_size"]

    base_name = video_file.replace(video_ext, "")  
    video_path = os.path.join(env_path, video_file)
    jsonl_path = os.path.join(env_path, base_name + ".jsonl")

    out_video_dir = os.path.join(out_env_path, base_name)
    if os.path.exists(out_video_dir) and not overwrite:
        # Skip if already processed
        print(f"[{env_dir}] Skipping existing: {out_video_dir}")
        return

    # Make sure output folder exists
    os.makedirs(out_video_dir, exist_ok=True)

    # Read actions from JSON lines
    if not os.path.exists(jsonl_path):
        print(f"[{env_dir}] No corresponding .jsonl for {video_file}, skipping.")
        return
    with open(jsonl_path, "r") as f:
        action_lines = [json.loads(line) for line in f]

    # Open the video
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"[{env_dir}] {base_name}: original_fps={original_fps:.2f}, "
        f"total_frames={total_frames}, extracting..."
    )

    # Decide on frame skipping factor
    # e.g., if original_fps=30 & desired_fps=10 => skip=3 => keep every 3rd frame
    skip = 1
    if desired_fps is not None and desired_fps > 0 and original_fps > 0:
        ratio = original_fps / desired_fps
        # If ratio=3 => skip=3 => keep every 3rd frame
        skip = max(int(round(ratio)), 1)

    saved_frame_idx = 0
    read_frame_idx = 0
    extracted_actions = []

    # We won't use tqdm here in parallel for each frame because parallel printing 
    # can get messy. Instead, you could do a single "progress bar" at the master level.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_frame_idx % skip == 0:
            # Resize if desired
            if desired_size is not None:
                frame = cv2.resize(frame, desired_size)

            # Save frame
            out_frame_path = os.path.join(
                out_video_dir, f"frame_{saved_frame_idx:06d}{image_ext}"
            )
            cv2.imwrite(out_frame_path, frame)

            # Align action
            if read_frame_idx < len(action_lines):
                extracted_actions.append(action_lines[read_frame_idx])
            else:
                extracted_actions.append({})

            saved_frame_idx += 1

        read_frame_idx += 1

    cap.release()

    # Save the extracted actions
    out_actions_path = os.path.join(out_env_path, base_name + "_actions.json")
    with open(out_actions_path, "w") as f:
        json.dump(extracted_actions, f)

    print(f"[{env_dir}] Done: {video_file} -> {saved_frame_idx} frames saved.")


def preprocess_videos_parallel(
    data_dir: str,
    out_dir: str,
    video_ext: str = ".mp4",
    image_ext: str = ".jpg",
    overwrite: bool = False,
    desired_fps: float = None,
    desired_size: tuple = None,
    num_workers: int = 4,
):
    """
    Parallel preprocessing:
      - collects all videos from data_dir
      - spawns a multiprocessing Pool
      - each process calls process_single_video on one video
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect all videos
    video_info_list = []
    envs = [
        d for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    for env_dir in envs:
        env_path = os.path.join(data_dir, env_dir)
        out_env_path = os.path.join(out_dir, env_dir)
        os.makedirs(out_env_path, exist_ok=True)

        files_in_env = os.listdir(env_path)
        video_files = [f for f in files_in_env if f.endswith(video_ext)]

        for vf in video_files:
            video_info_list.append({
                "env_dir": env_dir,
                "env_path": env_path,
                "video_file": vf,
                "out_env_path": out_env_path,
                "video_ext": video_ext,
                "desired_fps": desired_fps,
                "desired_size": desired_size,
            })

    print(f"Found {len(video_info_list)} videos total.")
    if not video_info_list:
        return

    # Partial function so we only pass (video_info) to pool.map
    worker_func = partial(
        process_single_video,
        image_ext=image_ext,
        overwrite=overwrite
    )

    # Create a pool of workers, map over video_info_list in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        # optional: tqdm for the pool
        list(tqdm(
            pool.imap_unordered(worker_func, video_info_list),
            total=len(video_info_list),
            desc="Processing videos",
        ))

    print("All videos processed in parallel!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/", help="Input data directory")
    parser.add_argument("--out_dir", type=str, default="processed/", help="Output directory for frames/actions")
    parser.add_argument("--video_ext", type=str, default=".mp4", help="Extension of the input video files")
    parser.add_argument("--image_ext", type=str, default=".jpg", help="Extension of the output frame files")
    parser.add_argument("--desired_fps", type=float, default=0.0, help="Desired FPS (0 to disable downsampling)")
    parser.add_argument("--desired_width", type=int, default=0, help="Desired image width (0 to keep original)")
    parser.add_argument("--desired_height", type=int, default=0, help="Desired image height (0 to keep original)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes")

    args = parser.parse_args()

    # If either dimension is 0, we won't resize
    if args.desired_width > 0 and args.desired_height > 0:
        desired_size = (args.desired_width, args.desired_height)
    else:
        desired_size = None
    
    # If desired_fps <= 0, treat it as None => no downsampling
    if args.desired_fps <= 0:
        desired_fps = None
    else:
        desired_fps = args.desired_fps

    preprocess_videos_parallel(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        video_ext=args.video_ext,
        image_ext=args.image_ext,
        overwrite=args.overwrite,
        desired_fps=desired_fps,
        desired_size=desired_size,
        num_workers=args.num_workers,
    )

# (mem_wm) ss24m050@vnode17:~/Documents/phd_playground$ python src/data/minecraft/process.py --data_dir /data/cvg/sebastian/minecraft --out_dir /data/cvg/sebastian/minecraft_processed --desired_height 128 --desired_fps 5