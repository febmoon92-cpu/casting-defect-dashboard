"""Experiment registry + statistical validation utilities.

- bootstrap_metric: percentile bootstrap CI for any scalar metric defined over
  (labels, probs).
- mcnemar_test: paired comparison of two models on the same test set.
- collect_runs: scan reports/*_predictions.csv and build a summary table with CIs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


METRIC_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
    "accuracy": lambda y, p, t: accuracy_score(y, (p >= t).astype(int)),
    "precision_defect": lambda y, p, t: precision_score(y, (p >= t).astype(int), pos_label=1, zero_division=0),
    "recall_defect": lambda y, p, t: recall_score(y, (p >= t).astype(int), pos_label=1, zero_division=0),
    "f1_defect": lambda y, p, t: f1_score(y, (p >= t).astype(int), pos_label=1, zero_division=0),
    "roc_auc": lambda y, p, t: roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
    "pr_auc": lambda y, p, t: average_precision_score(y, p) if len(set(y)) > 1 else float("nan"),
}


@dataclass
class RunSummary:
    tag: str
    metric: dict[str, float]
    ci: dict[str, tuple[float, float]]
    n: int


def bootstrap_metric(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str,
    threshold: float = 0.5,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point, lower, upper) for the metric over n_iter bootstrap resamples."""
    fn = METRIC_FUNCS[metric]
    rng = np.random.default_rng(seed)
    n = len(labels)
    base = float(fn(labels, probs, threshold))
    values = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        p = probs[idx]
        try:
            values[i] = fn(y, p, threshold)
        except Exception:
            values[i] = np.nan
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return base, float("nan"), float("nan")
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return base, lo, hi


def mcnemar_test(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict[str, float]:
    """McNemar test: compare two classifiers on the same test set.

    Returns dict with discordant counts (b, c) and continuity-corrected chi^2 p-value.
    b = A correct, B wrong; c = A wrong, B correct.
    """
    a_correct = preds_a == labels
    b_correct = preds_b == labels
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, df=1))
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": p}


def collect_runs(reports_dir: Path = config.REPORTS_DIR) -> dict[str, pd.DataFrame]:
    """Load every *_predictions.csv in reports/ and return prob frames keyed by tag."""
    frames: dict[str, pd.DataFrame] = {}
    for p in sorted(reports_dir.glob("*_predictions.csv")):
        tag = p.stem.replace("_predictions", "")
        frames[tag] = pd.read_csv(p)
    return frames


def summarize_runs(
    frames: dict[str, pd.DataFrame],
    threshold: float = 0.5,
    n_iter: int = 1000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for tag, df in frames.items():
        y = df["label"].values
        p = df["prob_defect"].values
        row: dict[str, object] = {"tag": tag, "n": len(df)}
        for name in METRIC_FUNCS:
            base, lo, hi = bootstrap_metric(y, p, name, threshold=threshold, n_iter=n_iter, alpha=alpha)
            row[name] = base
            row[f"{name}_lo"] = lo
            row[f"{name}_hi"] = hi
        pred = (p >= threshold).astype(int)
        row["TP"] = int(((y == 1) & (pred == 1)).sum())
        row["TN"] = int(((y == 0) & (pred == 0)).sum())
        row["FP"] = int(((y == 0) & (pred == 1)).sum())
        row["FN"] = int(((y == 1) & (pred == 0)).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("tag").sort_values("f1_defect", ascending=False)


def pairwise_mcnemar(
    frames: dict[str, pd.DataFrame],
    threshold: float = 0.5,
) -> pd.DataFrame:
    tags = list(frames.keys())
    out = []
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            df_a = frames[a]
            df_b = frames[b]
            df = df_a.merge(df_b, on="path", suffixes=("_a", "_b"))
            y = df["label_a"].values
            pa = (df["prob_defect_a"].values >= threshold).astype(int)
            pb = (df["prob_defect_b"].values >= threshold).astype(int)
            res = mcnemar_test(y, pa, pb)
            res.update({"model_A": a, "model_B": b})
            out.append(res)
    cols = ["model_A", "model_B", "b", "c", "chi2", "p_value"]
    return pd.DataFrame(out)[cols] if out else pd.DataFrame(columns=cols)


def save_experiments_json(
    frames: dict[str, pd.DataFrame],
    out_path: Path = None,
    threshold: float = 0.5,
    n_iter: int = 1000,
) -> Path:
    out_path = out_path or (config.REPORTS_DIR / "experiments.json")
    summary = summarize_runs(frames, threshold=threshold, n_iter=n_iter)
    pairwise = pairwise_mcnemar(frames, threshold=threshold)
    payload = {
        "threshold": threshold,
        "n_iter": n_iter,
        "summary": json.loads(summary.reset_index().to_json(orient="records")),
        "mcnemar": json.loads(pairwise.to_json(orient="records")) if not pairwise.empty else [],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    frames = collect_runs()
    if not frames:
        print("No prediction files found in reports/. Run src.evaluate first.")
        raise SystemExit(1)
    print(f"Loaded {len(frames)} runs: {list(frames.keys())}")
    out = save_experiments_json(frames, n_iter=1000)
    print(f"Saved {out}")
    print(summarize_runs(frames).round(4).to_string())
