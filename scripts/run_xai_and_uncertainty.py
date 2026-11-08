"""
Step 7 — Full Explainability (XAI) & Uncertainty Evaluation Pipeline (Chapter 4, Sections 4.6 & 4.7):
1. Per-modality 3D Grad-CAM++ (`Heatmap_T2W`, `Heatmap_ADC`, `Heatmap_HBV`)
   + Occlusion Fidelity Drop (`Delta_prob`) + Pointing Accuracy (if lesion masks exist)
2. Exact Shapley Values (`SHAP`) on the positive-class logit for the 4 clinical features
   `[patient_age, log_psa, psad, prostate_volume]` (when `--use_clinical` is active)
3. Monte Carlo Dropout Uncertainty (`T in {10, 25, 50}`) + Selective Abstention Curve
4. Publication-grade multi-panel figures saved to `docs/figures/`:
   - `xai_gradcam3d_per_modality.png`
   - `xai_clinical_shap_summary.png`
   - `xai_mc_dropout_abstention.png`
   and quantitative tables saved to `results/xai_and_uncertainty_summary.csv`.

Usage:
    # Evaluate on Fold 1 of Phase 4 (Image-only) or Config C7 (Image + Clinical):
    python3 -m scripts.run_xai_and_uncertainty --experiment phase5_c7_cross_attention_clinical --use_clinical
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from src.config import DATA_ROOT
from src.data.multimodal_dataset import (
    CLINICAL_FEATURE_NAMES,
    PicaiMultimodalDataset,
    fit_transform_clinical_features,
)
from src.data.picai_paths import find_complete_patients, load_marksheet
from src.models.cross_attention_fusion import MultimodalCrossAttentionNet
from src.utils.seed import set_seed
from src.xai.gradcam3d import (
    compute_occlusion_fidelity_drop,
    compute_per_modality_gradcam_pp_3d,
)
from src.xai.mc_dropout import evaluate_selective_abstention, mc_dropout_predict
from src.xai.shap_clinical import compute_exact_clinical_shap_on_logit


def _plot_gradcam_per_modality(
    patient_id: str,
    volume_np: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    modalities: list[str],
    out_path: Path,
) -> None:
    """Save a 2x3 panel showing raw MRI slice + Grad-CAM++ overlay per modality."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pick axial slice with highest total activation across modalities
    combined = sum(heatmaps[m] for m in modalities)
    z_peak = int(np.argmax(combined.max(axis=(1, 2))))

    fig, axes = plt.subplots(2, len(modalities), figsize=(4.5 * len(modalities), 8))
    for idx, mod in enumerate(modalities):
        img_slice = volume_np[idx, z_peak]
        cam_slice = heatmaps[mod][z_peak]

        ax_top = axes[0, idx]
        ax_top.imshow(img_slice, cmap="gray")
        ax_top.set_title(f"{mod.upper()} — Coupe axiale z={z_peak}", fontweight="bold")
        ax_top.axis("off")

        ax_bot = axes[1, idx]
        ax_bot.imshow(img_slice, cmap="gray")
        im = ax_bot.imshow(cam_slice, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        py, px = np.unravel_index(int(np.argmax(cam_slice)), cam_slice.shape)
        ax_bot.scatter([px], [py], c="white", edgecolors="black", s=70, marker="*", label="Pic d'attention")
        ax_bot.set_title(f"Heatmap_{mod.upper()} (Grad-CAM++ 3D)", fontweight="bold")
        ax_bot.axis("off")
        plt.colorbar(im, ax=ax_bot, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Explicabilité 3D Grad-CAM++ Séparée par Modalité — Patient {patient_id} (Section 4.6.1)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_shap_summary(shap_rows: list[dict[str, float]], out_path: Path) -> None:
    if not shap_rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    means = [
        float(np.mean([abs(r[feat]) for r in shap_rows]))
        for feat in CLINICAL_FEATURE_NAMES
    ]
    labels_pretty = ["Âge patient", "log(PSA)", "PSAD", "Volume prostatique"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = np.argsort(means)
    ax.barh(
        [labels_pretty[i] for i in order],
        [means[i] for i in order],
        color="#2b5c8f",
        edgecolor="black",
    )
    ax.set_xlabel("Importance moyenne |φ_j| sur le logit csPCa (SHAP exact)")
    ax.set_title("Attribution Clinique SHAP sur le Logit (Section 4.6.2)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_abstention_curve(
    curves_by_T: dict[int, list[dict[str, float]]], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {10: "#66c2a5", 25: "#2b5c8f", 50: "#d95f02"}
    for T, curve in curves_by_T.items():
        rates = [c["abstention_rate"] * 100.0 for c in curve]
        aurocs = [c["auroc_retained"] for c in curve]
        ax.plot(
            rates,
            aurocs,
            marker="o",
            linewidth=2.0,
            color=colors.get(T, None),
            label=f"MC Dropout (T = {T} passes)",
        )
    ax.set_xlabel("Taux d'abstention sélective (% cas incertains orientés au radiologue)")
    ax.set_ylabel("AUROC sur les cas retenus")
    ax.set_title(
        "Incertitude MC Dropout & Règle d'Abstention Sélective (Section 4.7)",
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_xai_pipeline(
    experiment: str = "phase4_segresnet_cross_attention",
    fold: int = 1,
    modalities: list[str] | None = None,
    use_clinical: bool = False,
    fusion_mode: str = "cross_attention",
    shared_weights: bool = False,
    max_xai_samples: int = 20,
    seed: int = 42,
) -> None:
    set_seed(seed)
    if modalities is None:
        modalities = ["t2w", "adc", "hbv"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(os.environ.get("RUN_OUTPUT_ROOT", "/root/prostate_xai"))
    ckpt_path = out_root / "checkpoints" / experiment / f"fold_{fold}" / "best.pt"
    if not ckpt_path.exists():
        alt_ckpt = Path("/workspace/checkpoints") / experiment / f"fold_{fold}" / "best.pt"
        if alt_ckpt.exists():
            ckpt_path = alt_ckpt

    complete_ids, _ = find_complete_patients(DATA_ROOT)
    marksheet = load_marksheet(DATA_ROOT, complete_ids=complete_ids)
    patient_ids = marksheet["patient_id"].tolist()
    labels = marksheet["label"].tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(patient_ids, labels))
    train_idx, val_idx = splits[fold - 1]
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
        align_to_t2w=True,
        clinical_features=val_clin,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    model = MultimodalCrossAttentionNet(
        modalities=modalities,
        shared_weights=shared_weights,
        use_clinical=use_clinical,
        fusion_mode=fusion_mode,  # type: ignore[arg-type]
    ).to(device)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Support both checkpoint key formats
        model_key = "model_state_dict" if "model_state_dict" in state else "model"
        model.load_state_dict(state[model_key])
        print(f"[XAI] Checkpoint chargé : {ckpt_path}")
    else:
        print(f"[XAI] Attention : {ckpt_path} non trouvé, exécution avec les poids courants.")

    # 1. Grad-CAM++ 3D & Occlusion Fidelity + SHAP on top positive cases
    occlusion_rows: list[dict[str, float]] = []
    shap_rows: list[dict[str, float]] = []
    n_plotted = 0

    n_done = 0
    for idx in range(len(val_ds)):
        if n_done >= max_xai_samples:
            break
        item = val_ds[idx]
        if use_clinical:
            vol_t, clin_t, lbl_t = item
            clin_b = clin_t.unsqueeze(0).to(device)
        else:
            vol_t, lbl_t = item
            clin_b = None

        if int(lbl_t.item()) != 1:
            continue

        vol_b = vol_t.unsqueeze(0).to(device)
        pid = val_ids[idx]

        heatmaps = compute_per_modality_gradcam_pp_3d(model, vol_b, clin=clin_b)
        drops = compute_occlusion_fidelity_drop(model, vol_b, heatmaps, clin=clin_b)
        occlusion_rows.append(drops)

        if n_plotted < 3:
            fig_path = Path("docs/xai_and_robustness_visuals") / f"xai_gradcam3d_{experiment}_patient{pid}.png"
            _plot_gradcam_per_modality(pid, vol_t.numpy(), heatmaps, modalities, fig_path)
            # Also save main figure in docs/figures/
            if n_plotted == 0:
                main_fig_path = Path("docs/figures/xai_gradcam3d_per_modality.png")
                _plot_gradcam_per_modality(pid, vol_t.numpy(), heatmaps, modalities, main_fig_path)
            print(f"[XAI] Figure Grad-CAM++ 3D sauvegardée -> {fig_path}")
            n_plotted += 1

        if use_clinical and clin_b is not None:
            shap_vals = compute_exact_clinical_shap_on_logit(model, vol_b, clin_b)
            shap_rows.append(shap_vals)

        n_done += 1

    if shap_rows:
        shap_fig = Path("docs/xai_and_robustness_visuals") / f"xai_clinical_shap_{experiment}.png"
        _plot_shap_summary(shap_rows, shap_fig)
        print(f"[XAI] Figure SHAP clinique sauvegardée -> {shap_fig}")

    # 2. MC Dropout (T = 10, 25, 50) & Selective Abstention across validation fold
    curves_by_T: dict[int, list[dict[str, float]]] = {}
    for T in (10, 25, 50):
        all_mean_p: list[float] = []
        all_var_p: list[float] = []
        all_lbls: list[int] = []
        for batch in val_loader:
            if len(batch) == 3:
                imgs, clins, lbls = batch
            else:
                imgs, lbls = batch
                clins = None
            mc_out = mc_dropout_predict(model, imgs, clin=clins, n_passes=T)
            all_mean_p.extend(mc_out["mean_prob"].tolist())
            all_var_p.extend(mc_out["var_prob"].tolist())
            all_lbls.extend(lbls.tolist())

        curve = evaluate_selective_abstention(all_lbls, all_mean_p, all_var_p)
        curves_by_T[T] = curve
        print(
            f"[MC Dropout T={T}] AUROC(0% abst.)={curve[0]['auroc_retained']:.4f} -> "
            f"AUROC(20% abst.)={curve[4]['auroc_retained']:.4f}"
        )

    abst_fig = Path("docs/xai_and_robustness_visuals") / f"xai_mc_dropout_abstention_{experiment}.png"
    _plot_abstention_curve(curves_by_T, abst_fig)
    print(f"[XAI] Courbe d'abstention MC Dropout sauvegardée -> {abst_fig}")

    # Save quantitative summary CSV (Table 5.3)
    csv_out = out_root / "results" / "xai_and_uncertainty_summary.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    mean_drop_joint = (
        float(np.mean([r["drop_joint"] for r in occlusion_rows]))
        if occlusion_rows
        else 0.0
    )
    row = {
        "experiment": experiment,
        "fold": fold,
        "n_xai_positive_cases": n_done,
        "mean_occlusion_drop_joint": round(mean_drop_joint, 6),
        "mc_auroc_T25_abst0": round(curves_by_T[25][0]["auroc_retained"], 6),
        "mc_auroc_T25_abst10": round(curves_by_T[25][2]["auroc_retained"], 6),
        "mc_auroc_T25_abst20": round(curves_by_T[25][4]["auroc_retained"], 6),
    }
    write_header = not csv_out.exists()
    with open(csv_out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[XAI] Résumé quantitatif sauvegardé -> {csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: XAI & Uncertainty Pipeline.")
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
    parser.add_argument("--max_xai_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_xai_pipeline(
        experiment=args.experiment,
        fold=args.fold,
        use_clinical=args.use_clinical,
        fusion_mode=args.fusion_mode,
        shared_weights=args.shared_weights,
        max_xai_samples=args.max_xai_samples,
        seed=args.seed,
    )
