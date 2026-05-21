"""Evaluate a trained model on the test set and dump artifacts to reports/.

Outputs:
  reports/figures/{model}_confusion_matrix.png
  reports/figures/{model}_roc.png
  reports/figures/{model}_pr.png
  reports/figures/{model}_threshold_sweep.png
  reports/{model}_test_metrics.json
  reports/{model}_predictions.csv
  reports/misclassified/{FN,FP}/...
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc as sk_auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from . import config
from .dataset import IDX_TO_CLASS, build_loaders
from .models import build_model
from .utils import get_device, set_seed


def _infer(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_paths: list[str] = []
    with torch.no_grad():
        for xb, yb, paths in tqdm(loader, desc="predict", leave=False):
            xb = xb.to(device, non_blocking=True)
            if tta:
                logits = model(xb) + model(torch.flip(xb, dims=[3]))
                logits = logits / 2.0
            else:
                logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())
            all_paths.extend(paths)
    return np.concatenate(all_probs), np.concatenate(all_labels), all_paths


def _plot_confusion(cm: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["ok_front", "def_front"])
    ax.set_yticklabels(["ok_front", "def_front"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_roc(labels, probs, out: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    a = sk_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC={a:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_pr(labels, probs, out: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision, label=f"AP={ap:.4f}")
    ax.set_xlabel("Recall (defect)")
    ax.set_ylabel("Precision (defect)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _threshold_sweep(labels: np.ndarray, probs: np.ndarray) -> list[dict]:
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (probs >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(labels, pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(labels, pred, pos_label=1, zero_division=0)),
            "accuracy": float(accuracy_score(labels, pred)),
        })
    return rows


def _plot_sweep(rows: list[dict], out: Path, title: str) -> None:
    ts = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ts, [r["precision"] for r in rows], marker="o", label="precision")
    ax.plot(ts, [r["recall"] for r in rows], marker="o", label="recall")
    ax.plot(ts, [r["f1"] for r in rows], marker="o", label="f1")
    ax.set_xlabel("Threshold for defect probability")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _dump_misclassified(paths, labels, preds, model_tag: str) -> None:
    base = config.MISCLASSIFIED_DIR
    for sub in ("FN", "FP"):
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
    fn_count = fp_count = 0
    for path, y, p in zip(paths, labels, preds):
        src = Path(path)
        if y == 1 and p == 0:
            dst = base / "FN" / f"{model_tag}__{src.parent.name}__{src.name}"
            shutil.copy2(src, dst)
            fn_count += 1
        elif y == 0 and p == 1:
            dst = base / "FP" / f"{model_tag}__{src.parent.name}__{src.name}"
            shutil.copy2(src, dst)
            fp_count += 1
    print(f"[misclassified] copied FN={fn_count}, FP={fp_count} -> {base}")


def evaluate(model_name: str, ckpt: Path, tag: str | None = None, tta: bool = False) -> dict:
    set_seed()
    config.ensure_dirs()
    device = get_device()
    run_tag = tag or model_name
    print(f"[evaluate] model={model_name} tag={run_tag} ckpt={ckpt} tta={tta} device={device}")

    model = build_model(model_name)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    model.to(device)

    loaders = build_loaders(batch_size=32)
    probs, labels, paths = _infer(model, loaders["test"], device, tta=tta)
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_defect": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_defect": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_defect": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold_sweep": _threshold_sweep(labels, probs),
        "n_test": int(len(labels)),
    }

    title = f"{run_tag} | Test"
    figs = config.FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)
    _plot_confusion(cm, figs / f"{run_tag}_confusion_matrix.png", title)
    _plot_roc(labels, probs, figs / f"{run_tag}_roc.png", title)
    _plot_pr(labels, probs, figs / f"{run_tag}_pr.png", title)
    _plot_sweep(metrics["threshold_sweep"], figs / f"{run_tag}_threshold_sweep.png", title)

    out_metrics = config.REPORTS_DIR / f"{run_tag}_test_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[evaluate] saved {out_metrics}")

    pred_csv = config.REPORTS_DIR / f"{run_tag}_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "label_name", "pred", "pred_name", "prob_defect"])
        for path, y, p, prob in zip(paths, labels, preds, probs):
            w.writerow([path, int(y), IDX_TO_CLASS[int(y)], int(p), IDX_TO_CLASS[int(p)], float(prob)])
    print(f"[evaluate] saved {pred_csv}")

    _dump_misclassified(paths, labels, preds, run_tag)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "resnet18", "efficientnet_b0"], required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--tta", action="store_true", help="enable test-time augmentation (horizontal flip)")
    args = parser.parse_args()

    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        default_ckpts = {
            "baseline": "baseline_cnn.pt",
            "resnet18": "resnet18_best.pt",
            "efficientnet_b0": "efficientnet_b0_best.pt",
        }
        ckpt = config.MODELS_DIR / default_ckpts[args.model]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    metrics = evaluate(args.model, ckpt, tag=args.tag, tta=args.tta)
    print(json.dumps({k: v for k, v in metrics.items() if k != "threshold_sweep"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
