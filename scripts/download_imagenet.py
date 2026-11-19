"""Acquire and verify ILSVRC2012 archives.

Credentials are read from the environment only and are never printed or stored.
Downloads default to val + devkit; use --include-train for the 138 GB archive.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from unlearning.data.verify import verify_archives

SOURCES = ("local", "gdrive", "kaggle", "hf", "tfds-manual")
KAGGLE_COMPETITION = "imagenet-object-localization-challenge"
HF_REPO = "ILSVRC/imagenet-1k"


def default_root() -> Path:
    return Path(os.environ.get("IMAGENET_ROOT", Path.home() / "datasets/imagenet"))


def require_env(*names: str) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing environment variables: {', '.join(missing)}")


def download_gdrive(raw_dir: Path, remote: str) -> None:
    if shutil.which("rclone") is None:
        raise SystemExit("rclone not found; install it with: brew install rclone")
    source = f"{remote.rstrip('/')}/raw"
    subprocess.run(
        ["rclone", "copy", source, str(raw_dir), "--progress"], check=True
    )


def download_kaggle(raw_dir: Path) -> None:
    require_env("KAGGLE_USERNAME", "KAGGLE_KEY")
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI not found; install it with: pip install kaggle")
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            KAGGLE_COMPETITION,
            "-p",
            str(raw_dir),
        ],
        check=True,
    )


def download_hf(raw_dir: Path) -> None:
    require_env("HF_TOKEN")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(raw_dir),
        token=os.environ["HF_TOKEN"],
        allow_patterns=["*.tar", "*.tar.gz"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default="local")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--remote",
        default=os.environ.get(
            "IMAGENET_GDRIVE_REMOTE", "ann-drive:ANN-Group-Project/imagenet"
        ),
    )
    args = parser.parse_args()

    root = args.root or default_root()
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        if args.source == "gdrive":
            download_gdrive(raw_dir, args.remote)
        elif args.source == "kaggle":
            download_kaggle(raw_dir)
        elif args.source == "hf":
            download_hf(raw_dir)
        elif args.source == "tfds-manual":
            print(f"Place the official ILSVRC2012 archives in {raw_dir}, then rerun.")
        elif args.source == "local":
            print(f"Verifying existing archives in {raw_dir}.")
        else:
            raise SystemExit(f"unsupported source: {args.source}")

    report = verify_archives(raw_dir, include_train=args.include_train)
    for message in report.errors:
        print(f"ERROR: {message}")
    print(f"verified files: {report.checked}")
    if not report.ok:
        raise SystemExit(1)
    print("Archive verification passed.")


if __name__ == "__main__":
    main()
