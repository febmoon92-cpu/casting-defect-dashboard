"""Misclassification deep-dive utilities.

Functions:
  categorize_errors(preds_df, threshold)        -> dict of close/large margin FN/FP
  consensus_misclassified(frames, threshold)    -> rows misclassified by >=k models
  build_inspection_table(preds_df)              -> sorted view with helpful columns
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


def categorize_errors(
    preds: pd.DataFrame,
    threshold: float = 0.5,
    close_margin: float = 0.20,
) -> dict[str, pd.DataFrame]:
    """Split FN/FP into close-margin (almost-right) and large-margin (very wrong)."""
    y = preds["label"].values
    p = preds["prob_defect"].values
    pred = (p >= threshold).astype(int)
    is_fn = (y == 1) & (pred == 0)
    is_fp = (y == 0) & (pred == 1)
    margin = p - threshold

    fn = preds[is_fn].copy()
    fn["margin"] = margin[is_fn]
    fn_close = fn[fn["prob_defect"] >= max(0.0, threshold - close_margin)].sort_values("prob_defect", ascending=False)
    fn_large = fn[fn["prob_defect"] < max(0.0, threshold - close_margin)].sort_values("prob_defect", ascending=True)

    fp = preds[is_fp].copy()
    fp["margin"] = margin[is_fp]
    fp_close = fp[fp["prob_defect"] <= threshold + close_margin].sort_values("prob_defect", ascending=True)
    fp_large = fp[fp["prob_defect"] > threshold + close_margin].sort_values("prob_defect", ascending=False)

    return {
        "FN_close": fn_close,
        "FN_large": fn_large,
        "FP_close": fp_close,
        "FP_large": fp_large,
    }


def consensus_misclassified(
    frames: dict[str, pd.DataFrame],
    threshold: float = 0.5,
    min_models: int = 2,
) -> pd.DataFrame:
    """Rows misclassified by at least `min_models` distinct runs.

    Useful for flagging probable label issues or genuinely hard samples.
    """
    if not frames:
        return pd.DataFrame()
    base = next(iter(frames.values()))[["path", "label"]].copy()
    base = base.set_index("path")
    base["miss_count"] = 0
    base["mean_prob_defect"] = 0.0
    base["models_missed"] = ""
    for tag, df in frames.items():
        df = df.set_index("path")
        pred = (df["prob_defect"].values >= threshold).astype(int)
        miss = pred != df["label"].values
        base.loc[df.index, "mean_prob_defect"] = base.loc[df.index, "mean_prob_defect"] + df["prob_defect"]
        for path in df.index[miss]:
            base.at[path, "miss_count"] = int(base.at[path, "miss_count"]) + 1
            current = base.at[path, "models_missed"]
            base.at[path, "models_missed"] = f"{current},{tag}" if current else tag
    base["mean_prob_defect"] = base["mean_prob_defect"] / max(len(frames), 1)
    out = base[base["miss_count"] >= min_models].reset_index()
    return out.sort_values(["miss_count", "label"], ascending=[False, False])


def build_inspection_table(preds: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    df = preds.copy()
    df["pred_at_thr"] = (df["prob_defect"] >= threshold).astype(int)
    df["error"] = df["label"] != df["pred_at_thr"]
    df["margin_from_thr"] = (df["prob_defect"] - threshold).round(4)
    df["confidence"] = (df["prob_defect"].where(df["pred_at_thr"] == 1, 1 - df["prob_defect"])).round(4)
    return df


def save_summary(frames: dict[str, pd.DataFrame], out_path: Path | None = None, threshold: float = 0.5) -> Path:
    out_path = out_path or (config.REPORTS_DIR / "fn_analysis.json")
    payload = {
        "threshold": float(threshold),
        "per_model": {},
        "consensus_min2": [],
    }
    for tag, df in frames.items():
        cats = categorize_errors(df, threshold=threshold)
        payload["per_model"][tag] = {
            "FN_close": int(len(cats["FN_close"])),
            "FN_large": int(len(cats["FN_large"])),
            "FP_close": int(len(cats["FP_close"])),
            "FP_large": int(len(cats["FP_large"])),
        }
    consensus = consensus_misclassified(frames, threshold=threshold, min_models=2)
    payload["consensus_min2"] = consensus.to_dict(orient="records")
    out_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    from .experiments import collect_runs

    frames = collect_runs()
    if not frames:
        print("No prediction files found.")
        raise SystemExit(1)
    out = save_summary(frames, threshold=0.5)
    print(f"Saved {out}")
    for tag, df in frames.items():
        cats = categorize_errors(df)
        print(f"{tag}: FN_close={len(cats['FN_close'])} FN_large={len(cats['FN_large'])} "
              f"FP_close={len(cats['FP_close'])} FP_large={len(cats['FP_large'])}")
