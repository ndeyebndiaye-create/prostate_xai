"""
Baseline comparison script — reads results/baselines.csv and prints a
formatted table comparing all modality configurations.

Usage:
    python scripts/compare_baselines.py

The script is intentionally simple: read CSV, group by (experiment, modality),
display mean AUROC ± std, highlight the best configuration. No external
plotting dependencies (matplotlib is not required for a thesis workflow that
needs quick textual summaries before the final figure generation step).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Resolve results/ relative to this script's location (scripts/ is a sibling of results/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CSV_PATH = _PROJECT_ROOT / "results" / "baselines.csv"


def load_results() -> pd.DataFrame:
    if not _CSV_PATH.exists():
        print(f"[ERROR] Results file not found: {_CSV_PATH}", file=sys.stderr)
        print("Run train_unimodal_baseline.py and/or train_fusion_baseline.py first.")
        sys.exit(1)
    df = pd.read_csv(_CSV_PATH)
    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print a formatted comparison table.

    Filters to summary rows only (fold == "summary") to avoid double-counting,
    then groups by (experiment, modality) to produce one row per run.
    """
    # Summary rows have fold == "summary" (stored as string).
    summary = df[df["fold"].astype(str) == "summary"].copy()

    if summary.empty:
        print("[WARNING] No summary rows found in CSV. The CSV may be incomplete.")
        return

    # Build a clean display table.
    display_cols = ["experiment", "modality", "mean_auroc", "std_auroc", "n_folds", "timestamp"]
    available_cols = [c for c in display_cols if c in summary.columns]
    table = summary[available_cols].sort_values("mean_auroc", ascending=False).reset_index(drop=True)

    # Format AUROC values.
    table["mean_auroc"] = table["mean_auroc"].map(lambda x: f"{x:.4f}")
    table["std_auroc"] = table["std_auroc"].map(lambda x: f"{x:.4f}")
    table["auroc_str"] = table.apply(
        lambda r: f"{r['mean_auroc']} ± {r['std_auroc']}", axis=1
    )

    # Print header.
    sep = "=" * 80
    print(f"\n{sep}")
    print("  BASELINE COMPARISON — mean AUROC ± std (5-fold stratified CV, ISUP≥3 label)")
    print(sep)
    print(f"  {'Experiment':<35} {'Modality':<20} {'AUROC (mean ± std)':<22} {'Folds'}")
    print(f"  {'-'*35} {'-'*20} {'-'*22} {'-'*5}")

    for _, row in table.iterrows():
        n_folds = int(row["n_folds"]) if "n_folds" in row else "?"
        print(f"  {row['experiment']:<35} {row['modality']:<20} {row['auroc_str']:<22} {n_folds}")

    print(sep)

    # Verdict — best configuration by mean AUROC.
    best_row = table.iloc[0]
    print(f"\n  ► Best configuration: [{best_row['experiment']}]  modality={best_row['modality']}")
    print(f"    AUROC = {best_row['auroc_str']}")

    # Compare against Bertelli et al. 2022 benchmark if T2W-only result is present.
    t2w_rows = table[table["modality"].str.contains("t2w", case=False, na=False)]
    if not t2w_rows.empty:
        t2w_auroc = float(t2w_rows.iloc[0]["mean_auroc"])
        print(f"\n  ► T2W-only AUROC = {t2w_auroc:.4f}")
        print(
            "    Reference: Bertelli et al. (2022) reports T2W dominating in unimodal ablation."
        )
        if t2w_auroc > 0.55:
            print("    ✓ T2W result is above chance — baseline is informative.")
        else:
            print(
                "    ✗ T2W AUROC is near chance (≤0.55). Investigate labels, preprocessing, "
                "or class balance before advancing to Phase 2."
            )

    print(f"\n  Full results CSV: {_CSV_PATH}\n")


def main() -> None:
    df = load_results()
    print_comparison_table(df)


if __name__ == "__main__":
    main()
