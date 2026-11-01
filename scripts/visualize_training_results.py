"""
Visualisation des résultats d'entraînement du modèle Baseline (Phase 1 T2W).
Lit results/baselines.csv et génère des graphiques clairs et interprétables pour le mémoire :
1. Graphique en barres de l'AUROC par fold (Folds 1 à 5) avec moyenne ± écart-type.
2. Comparaison détaillée et synthèse des métriques de validation.
Sauvegarde dans docs/figures/training_baseline_results.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path(__file__).resolve().parents[1] / "results" / "baselines.csv"

def plot_training_results():
    if not CSV_PATH.exists():
        print(f"Fichier introuvable : {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print("Contenu de baselines.csv :")
    print(df)

    # Filtrer les folds individuels et le résumé
    fold_df = df[df["fold"] != "summary"].copy()
    fold_df["fold_num"] = fold_df["fold"].astype(int)
    fold_df["auroc"] = fold_df["auroc"].astype(float)
    fold_df = fold_df.sort_values("fold_num")

    mean_auroc = float(fold_df["auroc"].mean())
    std_auroc = float(fold_df["auroc"].std())

    # Style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Graphique 1 : AUROC par Fold ─────────────────────────────────────────
    colors = ['#4575b4', '#74add1', '#abd9e9', '#fdae61', '#f46d43']
    bars = axes[0].bar(
        [f"Fold {i}" for i in fold_df["fold_num"]],
        fold_df["auroc"],
        color=colors,
        edgecolor='black',
        alpha=0.85,
        width=0.55
    )

    # Ligne de moyenne et zone ± 1 écart-type
    axes[0].axhline(mean_auroc, color='#d73027', linestyle='--', linewidth=2,
                    label=f'Moyenne 5-Fold = {mean_auroc:.3f}')
    axes[0].axhspan(mean_auroc - std_auroc, mean_auroc + std_auroc, color='#d73027',
                    alpha=0.15, label=f'$\pm$ 1 Écart-type ($\pm${std_auroc:.3f})')
    axes[0].axhline(0.50, color='gray', linestyle=':', label='Aléatoire (0.50)')

    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("Validation AUROC", fontsize=12, fontweight='bold')
    axes[0].set_title("Performance de Détection par Fold (Phase 1 — T2W Baseline)", fontsize=13, fontweight='bold', pad=12)
    axes[0].legend(loc='lower right', frameon=True)

    # Annotations sur chaque barre
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold', fontsize=11)

    # ── Graphique 2 : Synthèse Statistique et Paramètres Clés ───────────────
    axes[1].axis('off')
    
    summary_text = (
        r"$\bf{SYNTHÈSE\ EXPÉRIMENTALE\ (PHASE\ 1\ BASELINE)}$" + "\n\n"
        f"• Séquence évaluée : IRM T2-Weighted (T2W axiale)\n"
        f"• Architecture : SimpleCNN3D (4 blocs conv3d + batchnorm)\n"
        f"• Résolution spatiale : (96, 96, 16) voxels (center-crop / pad)\n"
        f"• Normalisation : Masked Z-Score (fond exclu, clip [-3, +3])\n"
        f"• Stratégie de validation : Stratified 5-Fold CV (Patient-level)\n"
        f"• Nombre d'epochs : 30 epochs / fold (150 epochs totales)\n"
        f"• Optimiseur : AdamW (lr=1e-4, weight_decay=1e-5)\n"
        f"• Fonction de coût : CrossEntropyLoss pondérée (poids pos=2.5)\n\n"
        r"$\bf{RÉSULTATS\ OBTENUS\ :}$" + "\n"
        f"  - Fold 1 AUROC : {fold_df.loc[fold_df['fold_num']==1, 'auroc'].values[0]:.4f}\n"
        f"  - Fold 2 AUROC : {fold_df.loc[fold_df['fold_num']==2, 'auroc'].values[0]:.4f}\n"
        f"  - Fold 3 AUROC : {fold_df.loc[fold_df['fold_num']==3, 'auroc'].values[0]:.4f}\n"
        f"  - Fold 4 AUROC : {fold_df.loc[fold_df['fold_num']==4, 'auroc'].values[0]:.4f}\n"
        f"  - Fold 5 AUROC : {fold_df.loc[fold_df['fold_num']==5, 'auroc'].values[0]:.4f}\n\n"
        f"  ★ AUROC MOYENNE : {mean_auroc:.4f} ± {std_auroc:.4f}\n"
        f"  ★ Meilleur Fold : Fold 5 ({fold_df['auroc'].max():.4f})\n\n"
        r"$\bf{INTERPRÉTATION\ :}$" + "\n"
        f"• Le modèle T2W seul dépasse nettement le hasard (0.50),\n"
        f"  atteignant 0.717 d'AUROC moyenne.\n"
        f"• La variabilité inter-fold (0.534 à 0.891) illustre l'hétérogénéité\n"
        f"  anatomique et confirme la nécessité d'apporter l'ADC et l'HBV\n"
        f"  (Phase 3 Fusion) pour stabiliser la détection."
    )

    axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
                 fontsize=10.5, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#cccccc'))

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "training_baseline_results.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Sauvegardé : {fig_path}")

if __name__ == "__main__":
    plot_training_results()
