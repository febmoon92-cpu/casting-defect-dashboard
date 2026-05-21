"""Copy a small set of demo images into ``reports/sample_images/`` so the
Streamlit Cloud build (which does not have the raw Kaggle dataset on disk)
can still render the misclassification deep-dive views.

Selection rule
--------------
For every ``reports/<tag>_predictions.csv`` we take **all** rows where the
model's prediction (at threshold 0.5) disagrees with the label - i.e. every
FN and FP across every model. The union across models is small (<150 files,
<2 MB at 300x300 grayscale jpeg), but it covers every case the dashboard's
Tab 3 (오분류 심화 분석) might want to visualize, including the consensus
mislabels.

Usage
-----
Run once from the project root, then ``git add reports/sample_images && commit``::

    python scripts/export_sample_images.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
SAMPLE_DIR = REPORTS / "sample_images"
THRESHOLD = 0.5


def is_error_row(row: pd.Series) -> bool:
    pred = 1 if float(row["prob_defect"]) >= THRESHOLD else 0
    return pred != int(row["label"])


def main() -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    csvs = sorted(REPORTS.glob("*_predictions.csv"))
    if not csvs:
        print(f"No predictions CSVs found in {REPORTS}")
        return 1

    copied: set[str] = set()
    missing: set[str] = set()
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        errors = df[df.apply(is_error_row, axis=1)]
        print(f"[{csv_path.name}] errors={len(errors)} / total={len(df)}")
        for raw_path in errors["path"]:
            src = Path(raw_path)
            if not src.exists():
                missing.add(str(src))
                continue
            dst = SAMPLE_DIR / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            copied.add(src.name)

    total_files = sum(1 for _ in SAMPLE_DIR.glob("*"))
    total_size = sum(p.stat().st_size for p in SAMPLE_DIR.glob("*"))
    print()
    print(f"copied {len(copied)} new files this run")
    print(f"sample_images/ now contains {total_files} files ({total_size/1024:.1f} KB total)")
    if missing:
        print(f"WARNING: {len(missing)} source files were missing on disk; first 5:")
        for m in list(missing)[:5]:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
