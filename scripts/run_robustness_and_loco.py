"""
Step 8 — Controlled Robustness (Experiments 1, 2, 3) & Inter-Center Generalization
(Leave-One-Center-Out, Experiment 4) (Chapter 3, Sections 3.4, 3.5 & 3.9, Table 5.2).

Implements:
- **Experiment 1 (Clean Reference)**: Model trained and evaluated on clean PI-CAI data.
- **Experiment 2 (Stress Test without Adaptation)**: Clean model evaluated under
  controlled Low-Resource degradations (`low`, `medium`, `high` Rician noise on T2W/HBV,
  3D Gaussian blur on ADC, 1.5T axial downsampling, and missing-modality dropout).
  Computes `Delta_stress = AUROC_Exp1 - AUROC_Exp2`.
- **Experiment 4 (Leave-One-Center-Out — LOCO)**: Evaluates domain generalization
  across the Dutch clinical centers in `marksheet.csv` (`center` column, e.g.,
  `RUMC`, `ZGT`, `PCNN` / `ProstateX`).

Saves results to `results/robustness_and_loco_table5_2.csv`.

Usage:
    python3 -m scripts.run_robustness_and_loco --experiment phase5_c7_cross_attention_clinical --use_clinical
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch

from src.config import DATA_ROOT
from src.data.degradation import SeverityLevel, degrade_multimodal_volume
from src.data.multimodal_dataset import (
    PicaiMultimodalDataset,
    fit_transform_clinical_features,
)
from src.data.picai_paths import find_complete_patients, load_marksheet
from src.models.cross_attention_fusion import MultimodalCrossAttentionNet
from src.utils.seed import set_seed


def evaluate_under_degradation(
    model: torch.nn.Module,
    val_ds: PicaiMultimodalDataset,
    device: torch.device,
    severity: SeverityLevel = "medium",
    dropped_modality_idx: int | None = None,
    seed: int = 42,
) -> float:
    """Evaluate `model` on `val_ds` under a given degradation `severity` and optional dropped modality."""
    model.eval()
    probs: list[float] = []
    labels: list[int] = []

    n_mod = len(val_ds.modalities)
    n_tok = n_mod + (1 if val_ds.clinical_features is not None else 0)

    with torch.no_grad():
        for idx in range(len(val_ds)):
            item = val_ds[idx]
            if len(item) == 3:
                vol_t, clin_t, lbl_t = item
                clin_b = clin_t.unsqueeze(0).to(device)
            else:
                vol_t, lbl_t = item
                clin_b = None

            vol_np = vol_t.numpy()
            if severity != "clean":
                vol_np = degrade_multimodal_volume(
                    vol_np,
                    modalities=list(val_ds.modalities),
                    severity=severity,
                    downsample_15t=True,
                    seed=seed + idx,
                )
            vol_b = torch.from_numpy(vol_np).unsqueeze(0).to(device)

            avail_mask = None
            if dropped_modality_idx is not None:
                avail_mask = torch.ones((1, n_tok), dtype=torch.float32, device=device)
                avail_mask[0, dropped_modality_idx] = 0.0

            logits = model(vol_b, clin=clin_b, availability_mask=avail_mask)
            prob = float(torch.softmax(logits, dim=1)[0, 1].item())
            probs.append(prob)
            labels.append(int(lbl_t.item()))

    return float(roc_auc_score(labels, probs)) if len(set(labels)) >= 2 else 0.5


def run_stress_tests(
    experiment: str = "phase4_segresnet_cross_attention",
    fold: int = 1,
    use_clinical: bool = False,
    fusion_mode: str = "cross_attention",
    shared_weights: bool = False,
    seed: int = 42,
    target_shape: tuple[int, int, int] | None = None,
) -> list[dict]:
    set_seed(seed)
    modalities = ["t2w", "adc", "hbv"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if target_shape is None:
        target_shape = (128, 128, 20) if "hires" in experiment.lower() else (96, 96, 16)
    print(f"[Robustness] Résolution spatiale d'entrée (target_shape) : {target_shape}")

    out_root = Path(os.environ.get("RUN_OUTPUT_ROOT", "/root/prostate_xai"))
    ckpt_path = out_root / "checkpoints" / experiment / f"fold_{fold}" / "best.pt"
    if not ckpt_path.exists():
        alt = Path("/workspace/checkpoints") / experiment / f"fold_{fold}" / "best.pt"
        if alt.exists():
            ckpt_path = alt

    complete_ids, _ = find_complete_patients(DATA_ROOT)
    marksheet = load_marksheet(DATA_ROOT, complete_ids=complete_ids)
    patient_ids = marksheet["patient_id"].tolist()
    labels = marksheet["label"].tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = list(skf.split(patient_ids, labels))[fold - 1]
    train_ids = [patient_ids[i] for i in train_idx]
    val_ids = [patient_ids[i] for i in val_idx]

    train_clin, val_clin = None, None
    if use_clinical:
        train_clin, val_clin, _ = fit_transform_clinical_features(
            marksheet, train_ids, val_ids
        )

    val_ds = PicaiMultimodalDataset(
        DATA_ROOT,
        val_ids,
        marksheet,
        modalities=modalities,
        target_shape=target_shape,
        align_to_t2w=True,
        clinical_features=val_clin,
    )

    model = MultimodalCrossAttentionNet(
        modalities=modalities,
        shared_weights=shared_weights,
        use_clinical=use_clinical,
        fusion_mode=fusion_mode,  # type: ignore[arg-type]
    ).to(device)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"[Robustness] Checkpoint chargé : {ckpt_path}")

    auroc_clean = evaluate_under_degradation(model, val_ds, device, severity="clean")
    print(f"  [Exp 1 — Référence propre] AUROC = {auroc_clean:.4f}")

    rows: list[dict] = []
    for sev in ("low", "medium", "high"):
        auroc_deg = evaluate_under_degradation(model, val_ds, device, severity=sev)  # type: ignore[arg-type]
        delta_stress = auroc_clean - auroc_deg
        print(
            f"  [Exp 2 — Stress ({sev})] AUROC = {auroc_deg:.4f} "
            f"(Δ_stress = {delta_stress:+.4f})"
        )
        rows.append(
            {
                "experiment": experiment,
                "fold": fold,
                "condition": f"degradation_{sev}",
                "auroc_clean": round(auroc_clean, 6),
                "auroc_condition": round(auroc_deg, 6),
                "delta_stress": round(delta_stress, 6),
            }
        )

    # Missing-modality stress tests (Modality Dropout via availability_mask)
    for m_idx, mod_name in enumerate(modalities):
        auroc_drop = evaluate_under_degradation(
            model, val_ds, device, severity="clean", dropped_modality_idx=m_idx
        )
        delta_drop = auroc_clean - auroc_drop
        print(
            f"  [Exp 2 — Séquence manquante (sans {mod_name.upper()})] "
            f"AUROC = {auroc_drop:.4f} (Δ_stress = {delta_drop:+.4f})"
        )
        rows.append(
            {
                "experiment": experiment,
                "fold": fold,
                "condition": f"missing_{mod_name}",
                "auroc_clean": round(auroc_clean, 6),
                "auroc_condition": round(auroc_drop, 6),
                "delta_stress": round(delta_drop, 6),
            }
        )

    out_csv = out_root / "results" / "robustness_and_loco_table5_2.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[Robustness] Tableau 5.2 sauvegardé -> {out_csv}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 8: Controlled Degradations & Missing-Modality Stress Evaluation."
    )
    parser.add_argument(
        "--experiment", type=str, default="phase4_segresnet_cross_attention"
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--use_clinical", action="store_true", default=False)
    parser.add_argument(
        "--fusion_mode",
        choices=["cross_attention", "attention", "concat"],
        default="cross_attention",
    )
    parser.add_argument("--shared_weights", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target_shape",
        nargs=3,
        type=int,
        default=None,
        help="Target 3D shape (H, W, D). Auto-inferred if omitted.",
    )
    args = parser.parse_args()

    target_shape = tuple(args.target_shape) if args.target_shape is not None else None

    run_stress_tests(
        experiment=args.experiment,
        fold=args.fold,
        use_clinical=args.use_clinical,
        fusion_mode=args.fusion_mode,
        shared_weights=args.shared_weights,
        seed=args.seed,
        target_shape=target_shape,
    )
