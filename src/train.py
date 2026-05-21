"""Training loop for both baseline CNN and ResNet18.

Usage:
    python -m src.train --model baseline   --epochs 15
    python -m src.train --model resnet18   --epochs 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from . import config
from .dataset import build_loaders
from .losses import build_loss
from .models import build_model
from .utils import get_device, set_seed


def _epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb, _ in tqdm(loader, leave=False, desc="train" if train else "val"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            all_logits.append(logits.detach().float().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds = logits.argmax(axis=1)
    probs_def = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()

    acc = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0))
    try:
        auc = float(roc_auc_score(labels, probs_def))
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "f1": f1,
        "auc": auc,
    }


def train_model(
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    out_path: Path,
    metrics_path: Path,
    loss_name: str = "ce",
    tag: str | None = None,
):
    set_seed()
    device = get_device()
    run_tag = tag or model_name
    print(f"[train] model={model_name} loss={loss_name} tag={run_tag} device={device} epochs={epochs} batch={batch_size}")

    loaders = build_loaders(batch_size=batch_size)
    train_loader, val_loader = loaders["train"], loaders["val"]

    train_labels = np.array([s.label for s in loaders["train_samples"]])
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = (counts.sum() / (2.0 * np.clip(counts, 1, None)))
    print(f"[train] class counts={counts.tolist()} weights={class_weights.tolist()}")

    model = build_model(model_name).to(device)
    criterion = build_loss(loss_name, class_weights=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    history: list[dict] = []
    best_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = _epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val = _epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)
        scheduler.step()
        dt = time.time() - t0

        entry = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": tr,
            "val": val,
            "seconds": dt,
        }
        history.append(entry)
        print(
            f"  epoch {epoch:02d}/{epochs} ({dt:5.1f}s) "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} f1={tr['f1']:.4f} | "
            f"val loss={val['loss']:.4f} acc={val['acc']:.4f} f1={val['f1']:.4f} auc={val['auc']:.4f}"
        )

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            bad_epochs = 0
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "val_f1": best_f1,
                    "epoch": epoch,
                    "loss": loss_name,
                    "tag": run_tag,
                },
                out_path,
            )
            print(f"    -> saved best (val f1={best_f1:.4f}) to {out_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= config.EARLY_STOP_PATIENCE:
                print(f"  early stop at epoch {epoch} (no val F1 improvement for {bad_epochs} epochs)")
                break

    metrics_payload = {
        "model": model_name,
        "loss": loss_name,
        "tag": run_tag,
        "best_val_f1": best_f1,
        "history": history,
    }
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    else:
        existing = {}
    existing[run_tag] = metrics_payload
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[train] metrics appended to {metrics_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "resnet18", "efficientnet_b0"], required=True)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help="run name used for checkpoint/metrics keys")
    parser.add_argument("--out", type=str, default=None, help="explicit checkpoint path")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.model == "baseline":
        epochs = args.epochs or config.BASELINE_EPOCHS
        lr = args.lr or config.BASELINE_LR
        batch = args.batch or config.BASELINE_BATCH
        default_out = config.MODELS_DIR / "baseline_cnn.pt"
        default_tag = "baseline"
    elif args.model == "resnet18":
        epochs = args.epochs or config.RESNET_EPOCHS
        lr = args.lr or config.RESNET_LR
        batch = args.batch or config.RESNET_BATCH
        default_out = config.MODELS_DIR / "resnet18_best.pt"
        default_tag = "resnet18"
    else:  # efficientnet_b0
        epochs = args.epochs or config.RESNET_EPOCHS
        lr = args.lr or config.RESNET_LR
        batch = args.batch or config.RESNET_BATCH
        default_out = config.MODELS_DIR / "efficientnet_b0_best.pt"
        default_tag = "efficientnet_b0"

    tag = args.tag or (f"{default_tag}_{args.loss}" if args.loss != "ce" else default_tag)
    out = Path(args.out) if args.out else (config.MODELS_DIR / f"{tag}.pt" if args.tag else default_out)

    metrics_path = config.MODELS_DIR / "metrics.json"
    train_model(args.model, epochs, lr, batch, out, metrics_path, loss_name=args.loss, tag=tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
