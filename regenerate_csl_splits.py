from pathlib import Path

BASE_DIR = Path("CSL/color")
SPLIT_DIR = Path("CSL/splits")
TARGET_LABELS = {"000000", "000001", "000002", "000003", "000004", "000005"}
SKIP_VIDEOS = {"P39_s1_00_2_color.avi"}
SPLITS = {
    "train": range(1, 43),
    "valid": range(43, 47),
    "test": range(47, 51),
}


def build_split_lines(label_dir: Path, persons):
    lines = []
    for person in persons:
        prefix = f"P{person:02d}_"
        files = sorted(label_dir.glob(f"{prefix}*.avi"))
        for video_path in files:
            if video_path.name in SKIP_VIDEOS:
                continue
            lines.append(f"{video_path.as_posix()}\t{label_dir.name}")
    return lines


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    label_dirs = sorted(
        [p for p in BASE_DIR.iterdir() if p.is_dir() and p.name in TARGET_LABELS],
        key=lambda p: p.name,
    )

    print("labels:", [p.name for p in label_dirs])
    for split_name, persons in SPLITS.items():
        all_lines = []
        for label_dir in label_dirs:
            all_lines.extend(build_split_lines(label_dir, persons))

        out_path = SPLIT_DIR / f"{split_name}_split.txt"
        out_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
        print(split_name, len(all_lines), out_path.as_posix())


if __name__ == "__main__":
    main()
