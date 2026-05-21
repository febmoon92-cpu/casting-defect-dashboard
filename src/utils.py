"""Generic helpers: seeding, env loading, device pick, image utils."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from . import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_kaggle_env() -> tuple[str, str]:
    """Load Kaggle credentials from .env and expose them as KAGGLE_USERNAME/KAGGLE_KEY.

    `python-dotenv` 는 로컬 개발 전용 의존성(requirements-dev.txt) 이므로
    Streamlit Cloud 배포 환경에는 설치되지 않는다. 따라서 import 를 함수
    내부로 지연시켜, Kaggle 다운로드 기능을 호출하지 않는 한 ImportError 가
    발생하지 않도록 한다.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError as exc:  # pragma: no cover - prod path
        raise RuntimeError(
            "python-dotenv 가 설치돼 있지 않습니다. 로컬에서 Kaggle 다운로드를 "
            "실행하려면 `pip install -r requirements-dev.txt` 로 의존성을 설치하세요."
        ) from exc

    load_dotenv(config.ROOT_DIR / ".env")
    username = os.environ.get("KAGGLE_USERNAME")
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if not username or not token:
        raise RuntimeError(
            "KAGGLE_USERNAME and KAGGLE_API_TOKEN must be set in .env"
        )
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = token
    return username, token


def find_data_root(raw_dir: Path = config.RAW_DIR) -> Path:
    """Locate the directory that directly contains 'train/' and 'test/'.

    Different Kaggle archives nest the data at different depths; walk the tree.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    candidates = []
    for path in raw_dir.rglob("train"):
        if path.is_dir() and (path.parent / "test").is_dir():
            candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a folder containing both 'train' and 'test' under {raw_dir}"
        )
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]
