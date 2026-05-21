"""Download the Kaggle casting dataset using credentials from .env."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .utils import load_kaggle_env


def download(dataset: str = config.KAGGLE_DATASET, dest: Path = config.RAW_DIR) -> Path:
    config.ensure_dirs()
    load_kaggle_env()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"[download] dataset = {dataset}")
    print(f"[download] dest    = {dest}")
    api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
    print("[download] done")
    return dest


def summarize(root: Path) -> None:
    from .utils import find_data_root

    data_root = find_data_root(root)
    print(f"[summary] data_root = {data_root}")
    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"  {split}: MISSING")
            continue
        for cls in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            n = sum(1 for _ in cls.glob("*"))
            print(f"  {split}/{cls.name}: {n} files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=config.KAGGLE_DATASET)
    parser.add_argument("--dest", default=str(config.RAW_DIR))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    if not args.skip_download:
        download(args.dataset, dest)
    summarize(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
