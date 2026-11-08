"""
Step 5.2 & Step 6 — Clinical-Only Baselines & Calibration Evaluation (Section 3.6.1 & 3.8):
1. Logistic Regression (Guerbet-inspired clinical baseline, Section 2.2.1 & 3.6.1)
   on [patient_age, log(PSA), PSAD, prostate_volume]
2. Gradient Boosting (shallow tree ensemble on the 4 clinical features)
3. Descriptive PI-RADS clinical reference (evaluated on the subset of cases with valid
   PI-RADS >= 1, strictly as an external clinical reference — never used as model input)

All models use the exact same 5-fold patient-level stratified split (`seed=42`) as C1-C7,
log AUROC to `results/baselines.csv`, and write full clinical & calibration metrics
(AUROC, Sensitivity, Specificity, PPV, NPV, F1, ECE, Brier @ 90% sensitivity & Youden)
to `results/clinical_and_calibration_metrics.csv`.

Usage:
    python3 -m scripts.train_clinical_baselines
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import DATA_ROOT
from src.data.multimodal_dataset import (
    CLINICAL_FEATURE_NAMES,
    fit_transform_clinical_features,
)
from src.data.picai_paths import find_complete_patients, load_marksheet
from src.utils.calibration_metrics import compute_clinical_and_calibration_metrics
from src.utils.config_loader import get_git_hash
from src.utils.fold_guard import assert_no_patient_overlap
from src.utils.results_logger import save_results
from src.utils.seed import set_seed


def _save_detailed_metrics(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
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
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_clinical_baselines(
    data_root: Path = DATA_ROOT,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, list[float]]:
    set_seed(seed)
    git_hash = get_git_hash()

    csv_path = Path(data_root) / "picai" / "picai_labels" / "clinical_information" / "marksheet.csv"
    if not csv_path.exists():
        import urllib.request

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/DIAGNijmegen/picai_labels/main/clinical_information/marksheet.csv"
        print(f"[train_clinical_baselines] Téléchargement de marksheet.csv depuis {url} -> {csv_path}")
        urllib.request.urlretrieve(url, csv_path)

    complete_ids, incomplete = find_complete_patients(data_root)
    marksheet = load_marksheet(data_root, complete_ids=complete_ids if complete_ids else None)
    patient_ids = marksheet["patient_id"].tolist()
    labels = marksheet["label"].tolist()

    print(
        f"=== Baselines Cliniques Seules (Section 3.6.1) — {len(patient_ids)} patients | "
        f"features={CLINICAL_FEATURE_NAMES} | seed={seed} | git={git_hash} ==="
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    models_spec = {
        "clinical_logreg": (
            "clin_4vars",
            lambda: LogisticRegression(
                class_weight="balanced", C=1.0, max_iter=1000, random_state=seed
            ),
        ),
        "clinical_gboost": (
            "clin_4vars",
            lambda: GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=seed,
            ),
        ),
    }

    out_root = Path(os.environ.get("RUN_OUTPUT_ROOT", "/workspace"))
    detailed_csv = out_root / "results" / "clinical_and_calibration_metrics.csv"
    results_summary: dict[str, list[float]] = {}

    for exp_name, (mod_str, model_fn) in models_spec.items():
        fold_aurocs: list[float] = []
        detailed_rows: list[dict] = []
        strat_buckets: dict[str, list[dict[str, float]]] = {
            "sens90": [],
            "youden": [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
            fold_num = fold_idx + 1
            train_ids = [patient_ids[i] for i in train_idx]
            val_ids = [patient_ids[i] for i in val_idx]
            assert_no_patient_overlap(train_ids, val_ids)

            train_clin, val_clin, _ = fit_transform_clinical_features(
                marksheet, train_ids, val_ids
            )
            X_train = np.stack([train_clin[pid] for pid in train_ids], axis=0)
            y_train = np.array([labels[i] for i in train_idx], dtype=np.int64)
            X_val = np.stack([val_clin[pid] for pid in val_ids], axis=0)
            y_val = np.array([labels[i] for i in val_idx], dtype=np.int64)

            clf = model_fn()
            clf.fit(X_train, y_train)

            train_probs = clf.predict_proba(X_train)[:, 1]
            val_probs = clf.predict_proba(X_val)[:, 1]

            for strat in ("sens90", "youden"):
                # Lock threshold strictly on the training fold, then apply to validation fold
                train_m = compute_clinical_and_calibration_metrics(
                    y_train, train_probs, threshold_strategy=strat
                )
                locked_thresh = train_m["threshold"]
                val_m = compute_clinical_and_calibration_metrics(
                    y_val, val_probs, threshold=locked_thresh, threshold_strategy=strat
                )
                strat_buckets[strat].append(val_m)
                detailed_rows.append(
                    {
                        "experiment": exp_name,
                        "modality": mod_str,
                        "threshold_strategy": strat,
                        "fold": fold_num,
                        **{k: round(v, 6) for k, v in val_m.items()},
                        "seed": seed,
                        "git_hash": git_hash,
                    }
                )

            auroc_fold = strat_buckets["sens90"][-1]["auroc"]
            fold_aurocs.append(auroc_fold)

        for strat, metrics_list in strat_buckets.items():
            mean_row = {
                k: round(float(np.mean([m[k] for m in metrics_list])), 6)
                for k in metrics_list[0]
            }
            detailed_rows.append(
                {
                    "experiment": exp_name,
                    "modality": mod_str,
                    "threshold_strategy": strat,
                    "fold": "summary_mean",
                    **mean_row,
                    "seed": seed,
                    "git_hash": git_hash,
                }
            )

        _save_detailed_metrics(detailed_rows, detailed_csv)
        save_results(
            exp_name,
            mod_str,
            fold_aurocs,
            {
                "seed": seed,
                "git_hash": git_hash,
                "config_file": "scripts/train_clinical_baselines.py",
            },
        )
        results_summary[exp_name] = fold_aurocs

        s90 = strat_buckets["sens90"]
        print(
            f"  -> {exp_name}: AUROC={np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f} | "
            f"Sens@90={np.mean([m['sensitivity'] for m in s90]):.3f} | "
            f"Spec@90={np.mean([m['specificity'] for m in s90]):.3f} | "
            f"ECE={np.mean([m['ece'] for m in s90]):.4f} | "
            f"Brier={np.mean([m['brier'] for m in s90]):.4f}"
        )

    # Descriptive PI-RADS reference (Section 3.2.2 & 3.6.1)
    pirads_col = "lesion_PIRADS" if "lesion_PIRADS" in marksheet.columns else (
        "pirads_max" if "pirads_max" in marksheet.columns else None
    )
    if pirads_col is not None:
        valid_pirads = marksheet.copy()
        # Handle string/multi-lesion formats in lesion_PIRADS (e.g., "4", "4,3", "NO") by taking max integer score 1..5
        def _parse_max_pirads(val: object) -> float:
            if pd.isna(val):
                return np.nan
            s = str(val).strip()
            nums = []
            for tok in s.replace(";", ",").replace("/", ",").split(","):
                try:
                    v = float(tok.strip())
                    if 1.0 <= v <= 5.0:
                        nums.append(v)
                except ValueError:
                    continue
            return max(nums) if nums else np.nan

        valid_pirads["_pirads_num"] = valid_pirads[pirads_col].apply(_parse_max_pirads)
        valid_pirads = valid_pirads.dropna(subset=["_pirads_num"]).copy()
        if len(valid_pirads) > 10 and valid_pirads["label"].nunique() >= 2:
            p_scores = (valid_pirads["_pirads_num"].to_numpy(dtype=np.float64) - 1.0) / 4.0
            p_labels = valid_pirads["label"].to_numpy(dtype=np.int64)
            m_pirads = compute_clinical_and_calibration_metrics(
                p_labels, p_scores, threshold=0.5  # corresponds to PI-RADS >= 3
            )
            print(
                f"\n[Référentiel Clinique Externe — {pirads_col} (n={len(valid_pirads)} cas annotés)] "
                f"AUROC={m_pirads['auroc']:.4f} | "
                f"Sens(PI-RADS>=3)={m_pirads['sensitivity']:.3f} | "
                f"Spec(PI-RADS>=3)={m_pirads['specificity']:.3f}"
            )

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 5.2 & 6: Clinical Baselines (LogReg, GBoost) + Calibration Metrics."
    )
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_clinical_baselines(
        data_root=args.data_root,
        n_folds=args.n_folds,
        seed=args.seed,
    )
