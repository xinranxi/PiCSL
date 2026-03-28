import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "colab_bundle"


INCLUDE_FILES = [
    ".gitignore",
    "BiLSTM.py",
    "DataProcessMoudle.py",
    "Module.py",
    "Net.py",
    "ReadConfig.py",
    "SLR.py",
    "Train.py",
    "WER.py",
    "decode.py",
    "export_test_predictions.py",
    "regenerate_csl_splits.py",
    "requirements.txt",
    "videoAugmentation.py",
    "COLAB_UPLOAD_GUIDE.md",
    "README.md",
    "preprocess_csl_videos.py",
]

INCLUDE_DIRS = [
    "evaluation",
    "evaluationT",
    "params",
]

SKIP_SUFFIXES = {".pth", ".zip", ".pyc", ".log", ".tmp"}
SKIP_NAMES = {
    "__pycache__",
    ".DS_Store",
    "tmp.ctm",
    "tmp.stm",
    "tmp2.ctm",
    "wer.txt",
    "wer（复件）.txt",
    "wer（另一个复件）.txt",
}


def reset_bundle_dir():
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)


def should_skip(path: Path) -> bool:
    if path.name in SKIP_NAMES:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def copy_file(rel_path: str):
    src = ROOT / rel_path
    if not src.exists() or should_skip(src):
        return
    dst = BUNDLE_DIR / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(rel_path: str):
    src_dir = ROOT / rel_path
    if not src_dir.exists():
        return
    for src in src_dir.rglob("*"):
        if src.is_dir():
            continue
        rel_file = src.relative_to(ROOT)
        if should_skip(src):
            continue
        dst = BUNDLE_DIR / rel_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_manifest():
    manifest = BUNDLE_DIR / "BUNDLE_CONTENTS.txt"
    rows = []
    for path in sorted(BUNDLE_DIR.rglob("*")):
        if path.is_file():
            rows.append(path.relative_to(BUNDLE_DIR).as_posix())
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_zip() -> Path:
    archive_base = ROOT / "colab_bundle"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=BUNDLE_DIR)
    return Path(archive_path)


def main():
    reset_bundle_dir()

    for rel_path in INCLUDE_FILES:
        copy_file(rel_path)

    for rel_path in INCLUDE_DIRS:
        copy_dir(rel_path)

    write_manifest()
    zip_path = make_zip()

    print(f"Colab bundle folder created: {BUNDLE_DIR}")
    print(f"Colab bundle zip created: {zip_path}")


if __name__ == "__main__":
    main()
