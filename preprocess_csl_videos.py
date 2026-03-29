import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def normalize_path(path_value):
    return os.path.normpath(path_value)


def collect_video_paths_from_splits(split_files):
    video_paths = []
    seen = set()
    for split_file in split_files:
        with open(split_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = line.split()
                if len(parts) < 2:
                    continue
                video_path = normalize_path(parts[0])
                if video_path not in seen:
                    seen.add(video_path)
                    video_paths.append(video_path)
    return video_paths


def collect_video_paths_from_label_dirs(color_root, label_start, label_end):
    video_paths = []
    color_root_path = Path(color_root)
    if not color_root_path.exists():
        return video_paths

    for label_idx in range(label_start, label_end + 1):
        label_dir = color_root_path / f"{label_idx:06d}"
        if not label_dir.exists() or not label_dir.is_dir():
            continue
        for video_path in sorted(label_dir.glob("*.avi")):
            video_paths.append(normalize_path(str(video_path)))
    return video_paths


def read_and_sample_video(video_path, frame_sample_stride, resize_hw):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    last_kept_frame = None
    width, height = resize_hw

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_sample_stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frames.append(frame)
            last_kept_frame = frame

        frame_idx += 1

    cap.release()

    if frame_idx > 0 and (frame_idx - 1) % frame_sample_stride != 0 and last_kept_frame is not None:
        frames.append(last_kept_frame.copy())

    if not frames:
        return np.zeros((1, height, width, 3), dtype=np.uint8)

    return np.asarray(frames, dtype=np.uint8)


def build_output_path(video_path, output_root):
    normalized_video_path = normalize_path(video_path)
    drive, tail = os.path.splitdrive(normalized_video_path)
    safe_tail = tail.lstrip("\\/")
    return os.path.join(output_root, safe_tail) + ".npy"


def main():
    parser = argparse.ArgumentParser(description="Preprocess CSL videos into sampled and resized .npy frame arrays.")
    parser.add_argument("--train-split", default="CSL/splits/train_split.txt")
    parser.add_argument("--valid-split", default="CSL/splits/valid_split.txt")
    parser.add_argument("--test-split", default="CSL/splits/test_split.txt")
    parser.add_argument("--output-root", default="CSL/preprocessed")
    parser.add_argument("--frame-sample-stride", type=int, default=4)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--color-root", default="CSL/color")
    parser.add_argument("--label-start", type=int, default=0)
    parser.add_argument("--label-end", type=int, default=99)
    parser.add_argument("--source-mode", choices=["split", "labels"], default="labels")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_files = [args.train_split, args.valid_split, args.test_split]
    if args.source_mode == "split":
        video_paths = collect_video_paths_from_splits(split_files)
    else:
        video_paths = collect_video_paths_from_label_dirs(
            color_root=args.color_root,
            label_start=args.label_start,
            label_end=args.label_end,
        )

    os.makedirs(args.output_root, exist_ok=True)

    print(f"Found {len(video_paths)} videos to preprocess")
    print(f"Output root: {args.output_root}")
    print(f"Frame sample stride: {args.frame_sample_stride}")
    print(f"Resize: {args.resize}x{args.resize}")
    print(f"Source mode: {args.source_mode}")
    if args.source_mode == "labels":
        print(f"Label range: {args.label_start:06d}-{args.label_end:06d}")

    for video_path in tqdm(video_paths):
        output_path = build_output_path(video_path, args.output_root)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        if (not args.overwrite) and os.path.exists(output_path):
            continue

        frames = read_and_sample_video(
            video_path=video_path,
            frame_sample_stride=max(1, args.frame_sample_stride),
            resize_hw=(args.resize, args.resize),
        )
        np.save(output_path, frames)

    print("Preprocessing completed")


if __name__ == "__main__":
    main()
