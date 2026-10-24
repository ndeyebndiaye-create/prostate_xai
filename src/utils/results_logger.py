"""
Experiment result logger — appends one row per fold to results/baselines.csv.

Design rules:
- Append-only: never overwrites existing rows, so historical runs are preserved.
- Creates results/ directory automatically if absent.
- One row per fold + one summary row per experiment call (mean/std).
- All AUROC values are rounded to 6 decimal places to keep the CSV human-readable
  while retaining enough precision for downstream analysis.
- Colonnes de reproductibilité (git_hash, seed, config_file) ajoutées en fin
  de ligne pour permettre de retrouver exactement quel code et quelle config
  a produit chaque résultat.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path

# Chemin résolu depuis l'emplacement de ce fichier (indépendant du CWD).
_RESULTS_DIR = Path(os.environ.get("RUN_OUTPUT_ROOT", "/workspace")) / "results"
_CSV_PATH = _RESULTS_DIR / "baselines.csv"

_COLUMNS = [
    "timestamp",
    "experiment",
    "modality",
    "fold",         # entier 1-based pour les lignes fold, "summary" pour le résumé
    "auroc",
    "mean_auroc",
    "std_auroc",
    "n_folds",
    # ── colonnes de reproductibilité ──────────────────────────────────────────
    "git_hash",     # hash court du commit Git (ex. "bf67ce3")
    "seed",         # seed aléatoire utilisée
    "config_file",  # fichier YAML utilisé, ou "argparse" si aucun
]


def _ensure_csv_exists() -> None:
    """Crée le dossier results/ et écrit le header du CSV s'ils n'existent pas."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not _CSV_PATH.exists():
        with open(_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
            writer.writeheader()


def save_results(
    experiment: str,
    modality: str,
    fold_aurocs: list[float],
    config: dict,
) -> None:
    """
    Append une ligne par fold + une ligne résumé dans results/baselines.csv.

    Parameters
    ----------
    experiment:
        Identifiant court de l'expérience, ex. "phase1_t2w_only" ou
        "fusion_simple". Clé primaire pour filtrer dans le CSV.
    modality:
        Chaîne de modalités, ex. "t2w", "adc", "hbv", ou "t2w+adc+hbv".
    fold_aurocs:
        Liste des AUROC par fold (une valeur par fold).
    config:
        Dict d'hyperparamètres. Les clés suivantes sont lues et écrites
        dans le CSV si présentes :
          - "seed"        (int) : graine aléatoire
          - "config_file" (str) : chemin YAML ou "argparse"
          - "git_hash"    (str) : hash du commit
        Les autres clés sont ignorées (non écrites dans le CSV pour rester
        compact, mais disponibles dans la console via les prints du script).
    """
    import numpy as np  # import local : ce module doit rester léger à l'import

    _ensure_csv_exists()

    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs))
    n_folds = len(fold_aurocs)

    # Extraire les métadonnées de reproductibilité depuis config.
    git_hash = config.get("git_hash", "unknown")
    seed = config.get("seed", "unknown")
    config_file = config.get("config_file", "argparse")

    rows = []
    for fold_idx, auroc in enumerate(fold_aurocs, start=1):
        rows.append({
            "timestamp": ts,
            "experiment": experiment,
            "modality": modality,
            "fold": fold_idx,
            "auroc": round(auroc, 6),
            "mean_auroc": round(mean_auroc, 6),
            "std_auroc": round(std_auroc, 6),
            "n_folds": n_folds,
            "git_hash": git_hash,
            "seed": seed,
            "config_file": config_file,
        })

    # Ligne résumé — fold="summary" pour filtrage pandas facile.
    rows.append({
        "timestamp": ts,
        "experiment": experiment,
        "modality": modality,
        "fold": "summary",
        "auroc": round(mean_auroc, 6),
        "mean_auroc": round(mean_auroc, 6),
        "std_auroc": round(std_auroc, 6),
        "n_folds": n_folds,
        "git_hash": git_hash,
        "seed": seed,
        "config_file": config_file,
    })

    with open(_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writerows(rows)

    print(f"[results_logger] Saved {n_folds} fold rows + 1 summary → {_CSV_PATH}")
    print(
        f"[results_logger] {experiment} | {modality} | "
        f"AUROC {mean_auroc:.3f} ± {std_auroc:.3f} | "
        f"seed={seed} | git={git_hash}"
    )
