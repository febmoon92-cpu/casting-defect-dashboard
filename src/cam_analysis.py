"""Aggregate Grad-CAM analysis across the entire test set.

Outputs (saved to reports/cam_analysis/):
  mean_cam_TP.npy, mean_cam_TN.npy, mean_cam_FN.npy   (HxW float32 arrays)
  per_image_stats.csv  - one row per test image with attention summary stats
  region_grid.npy      - 7x7 average attention by image region per group
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from . import config
from .gradcam import ClassifierOutputTarget, GradCAM, load_model_for_cam
from .models import get_target_layer_for
from .utils import get_device


def _eval_transform(img_size: int = config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def _classify_group(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 1 and pred == 0:
        return "FN"
    return "FP"


def _entropy_of(cam: np.ndarray) -> float:
    """Shannon entropy of normalized CAM as a 'spread' indicator (high = diffuse)."""
    flat = cam.flatten().astype(np.float64)
    if flat.sum() <= 1e-9:
        return 0.0
    p = flat / flat.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _region_grid_sum(cam: np.ndarray, grid: int = 7) -> np.ndarray:
    """Downsample CAM into a grid x grid grid of mean attention values."""
    h, w = cam.shape
    rh = h // grid
    rw = w // grid
    out = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            block = cam[i * rh : (i + 1) * rh, j * rw : (j + 1) * rw]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return out


def run_cam_analysis(
    model_tag: str,
    model_name: str,
    ckpt_path: Path,
    pred_csv: Path,
    out_dir: Path,
    img_size: int = config.IMG_SIZE,
    grid: int = 7,
    target_class: int = 1,
) -> Path:
    device = get_device()
    model = load_model_for_cam(model_name, ckpt_path, device)
    target_layer = get_target_layer_for(model)
    tf = _eval_transform(img_size)

    preds = pd.read_csv(pred_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_cam = {g: np.zeros((img_size, img_size), dtype=np.float64) for g in ("TP", "TN", "FN", "FP")}
    region = {g: np.zeros((grid, grid), dtype=np.float64) for g in mean_cam}
    counts = {g: 0 for g in mean_cam}
    rows: list[dict] = []

    with GradCAM(model=model, target_layers=[target_layer]) as cam_ctx:
        for _, r in tqdm(preds.iterrows(), total=len(preds), desc="cam"):
            path = r["path"]
            label = int(r["label"])
            pred = int(r["pred"])
            prob = float(r["prob_defect"])
            group = _classify_group(label, pred)
            try:
                with Image.open(path) as im:
                    im_rgb = im.convert("RGB").resize((img_size, img_size))
                x = tf(im_rgb).unsqueeze(0).to(device)
                cam = cam_ctx(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]
            except Exception as e:  # pragma: no cover - robustness
                print(f"  skipped {path}: {e}")
                continue
            cam = np.asarray(cam, dtype=np.float32)
            cam = np.clip(cam, 0.0, 1.0)
            mean_cam[group] += cam
            region[group] += _region_grid_sum(cam, grid=grid)
            counts[group] += 1

            top10 = float(np.quantile(cam, 0.9))
            rows.append({
                "path": path,
                "label": label,
                "pred": pred,
                "prob_defect": prob,
                "group": group,
                "cam_mean": float(cam.mean()),
                "cam_max": float(cam.max()),
                "cam_p90": top10,
                "cam_entropy": _entropy_of(cam),
                "cam_centroid_y": float((np.arange(img_size)[:, None] * cam).sum() / max(cam.sum(), 1e-9)),
                "cam_centroid_x": float((np.arange(img_size)[None, :] * cam).sum() / max(cam.sum(), 1e-9)),
            })

    for g in mean_cam:
        if counts[g] > 0:
            mean_cam[g] /= counts[g]
            region[g] /= counts[g]
        np.save(out_dir / f"mean_cam_{g}.npy", mean_cam[g].astype(np.float32))
        np.save(out_dir / f"region_grid_{g}.npy", region[g].astype(np.float32))

    stats_df = pd.DataFrame(rows)
    stats_path = out_dir / "per_image_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    counts_path = out_dir / "group_counts.json"
    pd.Series(counts).to_json(counts_path)
    print(f"[cam_analysis] counts={counts}")
    print(f"[cam_analysis] saved to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="resnet18")
    parser.add_argument("--model", default="resnet18", choices=["baseline", "resnet18", "efficientnet_b0"])
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        defaults = {
            "baseline": "baseline_cnn.pt",
            "resnet18": "resnet18_best.pt",
            "efficientnet_b0": "efficientnet_b0.pt",
        }
        ckpt = config.MODELS_DIR / defaults[args.model]
    pred_csv = config.REPORTS_DIR / f"{args.tag}_predictions.csv"
    out_dir = config.REPORTS_DIR / "cam_analysis" / args.tag
    run_cam_analysis(args.tag, args.model, ckpt, pred_csv, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
