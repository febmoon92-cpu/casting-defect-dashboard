"""Project-wide configuration: paths, hyperparameters, constants."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MISCLASSIFIED_DIR = REPORTS_DIR / "misclassified"

KAGGLE_DATASET = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"

CLASS_NAMES = ["def_front", "ok_front"]
DEFECT_CLASS = "def_front"
OK_CLASS = "ok_front"

IMG_SIZE = 224
SEED = 42

BASELINE_BATCH = 32
RESNET_BATCH = 32
NUM_WORKERS = 2

BASELINE_EPOCHS = 15
RESNET_EPOCHS = 15

BASELINE_LR = 1e-3
RESNET_LR = 1e-4

VAL_SPLIT = 0.15

EARLY_STOP_PATIENCE = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dirs() -> None:
    for d in [
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        MISCLASSIFIED_DIR / "FN",
        MISCLASSIFIED_DIR / "FP",
    ]:
        d.mkdir(parents=True, exist_ok=True)
