"""
Step 6 — Evaluate Clinical Diagnostic Metrics & Probabilistic Calibration (Section 3.8):
Reads saved fold checkpoints (`best.pt`) containing `best_probs` and `best_labels`
for any trained deep learning experiment (`C1`–`C7`, `phase4`, ablations) and computes:
- AUROC
- Locked clinical decision thresholds (90% sensitivity target & Youden's J)
- Sensitivity, Specificity, PPV (Precision), NPV, F1-score
- Expected Calibration Error (ECE) & Brier Score

Saves summary table to `results/clinical_and_calibration_metrics.csv`.

Usage:
    python3 -m scripts.evaluate_calibration_and_metrics
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from src.utils.calibration_metrics import compute_clinical_and_calibration_metrics
from src.utils.config_loader import get_git_hash


def evaluate_saved_checkpoints(
    checkpoints_root: Path | None = None,
    output_csv: Path | None = None,
) -> list[dict]:
    out_root = Path(os.environ.get("RUN_OUTPUT_ROOT", "/workspace"))
    if checkpoints_root is None:
        checkpoints_root = out_root / "checkpoints"
    if output_csv is None:
        output_csv = out_root / "results" / "clinical_and_calibration_metrics.csv"

    if not checkpoints_root.exists():
        print(f"[evaluate_calibration] Aucun dossier checkpoints trouvé sous {checkpoints_root}")
        return []

    git_hash = get_git_hash()
    all_rows: list[dict] = []

    for exp_dir in sorted(checkpoints_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        fold_dirs = sorted(exp_dir.glob("fold_*"))
        if not fold_dirs:
            continue

        for strat in ("sens90", "youden"):
            fold_metrics: list[dict[str, float]] = []
            for fold_dir in fold_dirs:
                best_pt = fold_dir / "best.pt"
                if not best_pt.exists():
                    continue
                fold_num = fold_dir.name.replace("fold_", "")
                state = torch.load(best_pt, map_location="cpu", weights_only=False)
                probs = state.get("best_probs", [])
                labels = state.get("best_labels", [])
                if not probs or not labels:
                    continue
                m = compute_clinical_and_calibration_metrics(
                    labels, probs, threshold_strategy=strat
                )
                fold_metrics.append(m)
                all_rows.append(
                    {
                        "experiment": exp_name,
                        "modality": "from_checkpoint",
                        "threshold_strategy": strat,
                        "fold": fold_num,
                        **{k: round(v, 6) for k, v in m.items()},
                        "seed": 42,
                        "git_hash": git_hash,
                    }
                )

            if fold_metrics:
                mean_m = {
                    k: round(float(np.mean([fm[k] for fm in fold_metrics])), 6)
                    for k in fold_metrics[0]
                }
                all_rows.append(
                    {
                        "experiment": exp_name,
                        "modality": "from_checkpoint",
                        "threshold_strategy": strat,
                        "fold": "summary_mean",
                        **mean_m,
                        "seed": 42,
                        "git_hash": git_hash,
                    }
                )
                if strat == "sens90":
                    print(
                        f"  [{exp_name} | {len(fold_metrics)} folds @ sens90] "
                        f"AUROC={mean_m['auroc']:.4f} | Sens={mean_m['sensitivity']:.3f} | "
                        f"Spec={mean_m['specificity']:.3f} | F1={mean_m['f1']:.3f} | "
                        f"ECE={mean_m['ece']:.4f} | Brier={mean_m['brier']:.4f}"
                    )

    if all_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "experiment",
            "modality",
            "threshold_strategy",
            "fold",
            "auroc",
            "threshold",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
            "brier",
            "ece",
            "seed",
            "git_hash",
        ]
        write_header = not output_csv.exists()
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(all_rows)
        print(f"[evaluate_calibration] {len(all_rows)} lignes sauvegardées -> {output_csv}")

    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 6: Evaluate clinical & calibration metrics on saved fold checkpoints."
    )
    parser.add_argument("--checkpoints_root", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    evaluate_saved_checkpoints(
        checkpoints_root=args.checkpoints_root,
        output_csv=args.output_csv,
    )
