"""
Visualisation complète de tous les résultats d'entraînement, des métriques cliniques
(Sensibilité@90%, Youden, ECE, Brier) et des courbes d'apprentissage par epoch.

Lit automatiquement :
- `results/baselines.csv` (C1, C2, C3, C4, clinical_logreg, clinical_gboost, phase4, C5, C6, C7)
- `results/clinical_and_calibration_metrics.csv` (Sensibilité, Spécificité, F1, ECE, Brier)
- `logs/<experiment>_fold<1..5>.csv` (Courbes de loss d'entraînement et d'AUROC de validation sur 30 epochs)

Génère et sauvegarde dans `docs/figures/` :
1. `docs/figures/training_all_models_comparison.png` : Comparaison globale des AUROC (Moyenne ± Écart-type + points des 5 folds) de tous les modèles entraînés.
2. `docs/figures/training_clinical_and_calibration_tradeoff.png` : Compromis Sensibilité vs Spécificité (@90% vs Youden) et Calibration (ECE & Brier).
3. `docs/figures/training_epoch_learning_curves.png` : Courbes d'apprentissage (Train Loss & Val AUROC sur les 30 epochs, moyenne ± écart-type sur les 5 folds).

Usage:
    python3 -m scripts.visualize_training_results
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = REPO_ROOT / "docs" / "xai_and_robustness_visuals"
VIS_DIR.mkdir(parents=True, exist_ok=True)

BASELINES_CSV = REPO_ROOT / "results" / "baselines.csv"
CLINICAL_CSV = REPO_ROOT / "results" / "clinical_and_calibration_metrics.csv"
LOGS_DIR = REPO_ROOT / "logs"

PRETTY_NAMES = {
    "phase1_hbv_only": "C3 — HBV seul (SimpleCNN3D)",
    "phase1_adc_only": "C2 — ADC seul (SimpleCNN3D)",
    "phase1_t2w_only": "C1 — T2W seul (SimpleCNN3D)",
    "fusion_simple": "C4 — Fusion précoce (T2W+ADC+HBV)",
    "phase4_segresnet_cross_attention": "Phase 4 — SegResNet 3D + Cross-Attn (Image seule)",
    "clinical_gboost": "Clinique seule — Gradient Boosting",
    "clinical_logreg": "Clinique seule — Régression Logistique (Guerbet)",
    "phase5_c5_concat_clinical": "C5 — SegResNet 3D + Clinique (Concaténation)",
    "phase5_c6_attention_clinical": "C6 — SegResNet 3D + Clinique (Self-Attention)",
    "phase5_c7_cross_attention_clinical": "C7 Baseline (Crop 96x96x16)",
    "phase5_c7_cross_attention_clinical_hires": "C7 Hi-Res (128x128x20, FOV étendu)",
}


def plot_all_models_comparison() -> None:
    if not BASELINES_CSV.exists():
        print(f"[visualize] Fichier introuvable : {BASELINES_CSV}")
        return

    df = pd.read_csv(BASELINES_CSV)
    # Keep latest run per (experiment, fold)
    df = df.drop_duplicates(subset=["experiment", "fold"], keep="last").copy()

    summary_df = df[df["fold"] == "summary"].copy()
    fold_df = df[df["fold"] != "summary"].copy()
    fold_df["auroc"] = fold_df["auroc"].astype(float)
    summary_df["mean_auroc"] = summary_df["mean_auroc"].astype(float)
    summary_df["std_auroc"] = summary_df["std_auroc"].astype(float)
    summary_df = summary_df.sort_values("mean_auroc", ascending=True)

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ── Panel A : Horizontal Bar Chart of Mean AUROC ± SD + Fold Dots ────────
    exps = summary_df["experiment"].tolist()
    y_labels = [PRETTY_NAMES.get(e, e) for e in exps]
    means = summary_df["mean_auroc"].to_numpy()
    stds = summary_df["std_auroc"].to_numpy()

    palette = []
    for e in exps:
        if "hires" in e:
            palette.append("#008080")  # Dark Teal for C7 Hi-Res
        elif "clinical" in e and "phase5" not in e:
            palette.append("#d95f02")  # Orange for clinical-only
        elif "phase5" in e or "phase4" in e:
            palette.append("#1b9e77")  # Green/Teal for SegResNet / Cross-Attention
        else:
            palette.append("#2b5c8f")  # Blue for C1-C4 image baselines

    y_pos = np.arange(len(exps))
    bars = axes[0].barh(
        y_pos,
        means,
        xerr=stds,
        capsize=5,
        color=palette,
        edgecolor="black",
        alpha=0.88,
        height=0.6,
    )

    # Overlay individual fold points
    for i, e in enumerate(exps):
        f_vals = fold_df[fold_df["experiment"] == e]["auroc"].to_numpy()
        jitter = np.linspace(-0.14, 0.14, num=len(f_vals)) if len(f_vals) > 1 else [0.0]
        axes[0].scatter(
            f_vals,
            np.full_like(f_vals, y_pos[i]) + jitter,
            color="white",
            edgecolors="black",
            s=42,
            zorder=4,
            label="Plis individuels (Folds 1–5)" if i == 0 else None,
        )

    for bar, m, s in zip(bars, means, stds):
        axes[0].text(
            min(0.96, m + s + 0.012),
            bar.get_y() + bar.get_height() / 2.0,
            f"{m:.4f} ± {s:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    axes[0].axvline(0.50, color="gray", linestyle=":", linewidth=1.3, label="Hasard (0.50)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(y_labels, fontsize=10.5, fontweight="bold")
    axes[0].set_xlim(0.48, 0.96)
    axes[0].set_xlabel("Validation AUROC (5-Fold CV, N = 1 180 patients)", fontsize=11, fontweight="bold")
    axes[0].set_title(
        "A. Comparaison Globale des Modèles (5-Fold CV Stratifiée)",
        fontsize=12.5,
        fontweight="bold",
    )
    axes[0].legend(loc="lower right", frameon=True)

    # ── Panel B : Fold-by-Fold Trajectory Across Key Models ──────────────────
    for idx, e in enumerate(exps):
        sub = fold_df[fold_df["experiment"] == e].copy()
        sub["fold_num"] = sub["fold"].astype(int)
        sub = sub.sort_values("fold_num")
        axes[1].plot(
            sub["fold_num"],
            sub["auroc"],
            marker="o",
            linewidth=2.0,
            label=PRETTY_NAMES.get(e, e),
        )

    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].set_xticklabels([f"Fold {k}" for k in range(1, 6)], fontweight="bold")
    axes[1].set_ylim(0.52, 0.88)
    axes[1].set_ylabel("Validation AUROC par pli", fontsize=11, fontweight="bold")
    axes[1].set_title(
        "B. Stabilité Inter-Plis (Folds 1 à 5)",
        fontsize=12.5,
        fontweight="bold",
    )
    axes[1].legend(fontsize=8.5, loc="lower left", frameon=True)

    plt.suptitle(
        "Synthèse Comparative des Entraînements sur la Cohorte PI-CAI (1 180 Patients Complets)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out1 = OUTPUT_DIR / "training_all_models_comparison.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.savefig(VIS_DIR / "training_all_models_comparison.png", dpi=300, bbox_inches="tight")
    # Also update legacy filename for backward compatibility
    plt.savefig(OUTPUT_DIR / "training_baseline_results.png", dpi=300, bbox_inches="tight")
    plt.savefig(VIS_DIR / "training_baseline_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Sauvegardé -> {out1}")


def plot_clinical_and_calibration_tradeoff() -> None:
    if not CLINICAL_CSV.exists():
        return

    df = pd.read_csv(CLINICAL_CSV)
    sum_df = df[df["fold"] == "summary_mean"].drop_duplicates(
        subset=["experiment", "threshold_strategy"], keep="last"
    ).copy()
    if sum_df.empty:
        return

    for col in ("auroc", "sensitivity", "specificity", "f1", "brier", "ece"):
        sum_df[col] = sum_df[col].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    # ── Panel A : Sensitivity vs Specificity (@90% vs Youden) ────────────────
    labels_bar = []
    sens_vals = []
    spec_vals = []
    f1_vals = []

    for _, row in sum_df.iterrows():
        exp_short = row["experiment"].replace("clinical_", "Clin-")
        strat_lbl = "Sens90%" if row["threshold_strategy"] == "sens90" else "Youden"
        labels_bar.append(f"{exp_short}\n({strat_lbl})")
        sens_vals.append(row["sensitivity"])
        spec_vals.append(row["specificity"])
        f1_vals.append(row["f1"])

    x = np.arange(len(labels_bar))
    width = 0.26
    axes[0].bar(x - width, sens_vals, width, label="Sensibilité (Rappel)", color="#2b5c8f", edgecolor="black")
    axes[0].bar(x, spec_vals, width, label="Spécificité", color="#66c2a5", edgecolor="black")
    axes[0].bar(x + width, f1_vals, width, label="F1-Score", color="#fdae61", edgecolor="black")
    axes[0].axhline(0.90, color="#d73027", linestyle="--", linewidth=1.4, label="Cible clinique Sensibilité = 90%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_bar, fontsize=9.5, fontweight="bold")
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("Score sur Validation (Moyenne 5-Fold)", fontweight="bold")
    axes[0].set_title("A. Compromis Sensibilité / Spécificité selon le Seuil Verrouillé", fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9, frameon=True)

    # ── Panel B : Calibration Metrics (ECE & Brier Score) ────────────────────
    uniq_exps = sum_df.drop_duplicates(subset=["experiment"], keep="last")
    x_cal = np.arange(len(uniq_exps))
    w_cal = 0.32
    b1 = axes[1].bar(
        x_cal - w_cal / 2,
        uniq_exps["ece"],
        w_cal,
        label="ECE (Expected Calibration Error ↓)",
        color="#d95f02",
        edgecolor="black",
    )
    b2 = axes[1].bar(
        x_cal + w_cal / 2,
        uniq_exps["brier"],
        w_cal,
        label="Score de Brier (↓)",
        color="#7570b3",
        edgecolor="black",
    )
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.005,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[1].set_xticks(x_cal)
    axes[1].set_xticklabels(
        [PRETTY_NAMES.get(e, e) for e in uniq_exps["experiment"]],
        fontsize=9.5,
        fontweight="bold",
    )
    axes[1].set_ylim(0.0, 0.28)
    axes[1].set_ylabel("Erreur de Calibration (plus bas = meilleur)", fontweight="bold")
    axes[1].set_title("B. Qualité de Calibration Probabiliste (ECE & Brier)", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out2 = OUTPUT_DIR / "training_clinical_and_calibration_tradeoff.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.savefig(VIS_DIR / "training_clinical_and_calibration_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Sauvegardé -> {out2}")


def plot_epoch_learning_curves() -> None:
    if not LOGS_DIR.exists():
        return

    csv_files = sorted(LOGS_DIR.glob("*_fold*.csv"))
    if not csv_files:
        return

    # Group by experiment prefix
    exp_groups: dict[str, list[pd.DataFrame]] = {}
    for f in csv_files:
        stem = f.stem  # e.g. phase4_segresnet_cross_attention_fold1
        exp_name = stem.rsplit("_fold", 1)[0]
        df_log = pd.read_csv(f)
        if not df_log.empty:
            exp_groups.setdefault(exp_name, []).append(df_log)

    if not exp_groups:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#2b5c8f"]

    for idx, (exp_name, dfs) in enumerate(exp_groups.items()):
        concat_df = pd.concat(dfs, ignore_index=True)
        concat_df["epoch"] = concat_df["epoch"].astype(int)
        concat_df["train_loss"] = concat_df["train_loss"].astype(float)
        concat_df["val_auroc"] = concat_df["val_auroc"].astype(float)

        grouped = concat_df.groupby("epoch")
        epochs = grouped["train_loss"].mean().index.to_numpy()
        loss_mean = grouped["train_loss"].mean().to_numpy()
        loss_std = grouped["train_loss"].std().fillna(0.0).to_numpy()
        auc_mean = grouped["val_auroc"].mean().to_numpy()
        auc_std = grouped["val_auroc"].std().fillna(0.0).to_numpy()

        c = colors[idx % len(colors)]
        lbl = f"{PRETTY_NAMES.get(exp_name, exp_name)} ({len(dfs)} folds)"

        axes[0].plot(epochs, loss_mean, color=c, linewidth=2.2, label=lbl)
        axes[0].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color=c, alpha=0.18)

        axes[1].plot(epochs, auc_mean, color=c, linewidth=2.2, label=lbl)
        axes[1].fill_between(epochs, auc_mean - auc_std, auc_mean + auc_std, color=c, alpha=0.18)

    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Perte d'Entraînement (Weighted Cross-Entropy)", fontweight="bold")
    axes[0].set_title("A. Dynamique de Convergence (Train Loss ± SD)", fontweight="bold")
    axes[0].legend(fontsize=9, frameon=True)

    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("AUROC de Validation", fontweight="bold")
    axes[1].set_title("B. Progression de l'AUROC de Validation par Epoch (± SD)", fontweight="bold")
    axes[1].legend(fontsize=9, frameon=True)

    plt.tight_layout()
    out3 = OUTPUT_DIR / "training_epoch_learning_curves.png"
    plt.savefig(out3, dpi=300, bbox_inches="tight")
    plt.savefig(VIS_DIR / "training_epoch_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Sauvegardé -> {out3}")


if __name__ == "__main__":
    plot_all_models_comparison()
    plot_clinical_and_calibration_tradeoff()
    plot_epoch_learning_curves()
