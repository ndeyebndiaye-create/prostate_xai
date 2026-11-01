"""
Génération des figures "Avant / Après" et "Étape par Étape" du prétraitement pour présentation (Slides / Soutenance / Mémoire).

Produit 3 figures de qualité publication :
1. presentation_preprocessing_step_by_step.png : Décomposition pédagogique des 4 étapes mathématiques.
2. presentation_before_after_direct.png : Comparaison directe AVANT vs APRÈS grand format avec annotations.
3. presentation_multimodal_harmonization.png : Harmonisation des 3 modalités hétérogènes (T2W, ADC, HBV) pour la fusion.
"""

import sys
from pathlib import Path

# Assure l'accès au module src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import SimpleITK as sitk

from src.data.preprocessing import center_crop_or_pad, load_and_normalize_volume

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_patient_10008"

def make_figures():
    print("Chargement des images brutes pour les figures de présentation...")
    t2w_path = SAMPLE_DIR / "10008_1000008_t2w.mha"
    adc_path = SAMPLE_DIR / "10008_1000008_adc.mha"
    hbv_path = SAMPLE_DIR / "10008_1000008_hbv.mha"

    t2w_sitk = sitk.ReadImage(str(t2w_path))
    adc_sitk = sitk.ReadImage(str(adc_path))
    hbv_sitk = sitk.ReadImage(str(hbv_path))

    t2w_raw = sitk.GetArrayFromImage(t2w_sitk).astype(np.float32)  # (21, 384, 384)
    adc_raw = sitk.GetArrayFromImage(adc_sitk).astype(np.float32)  # (21, 128, 120)
    hbv_raw = sitk.GetArrayFromImage(hbv_sitk).astype(np.float32)  # (21, 128, 120)

    # Coupe centrale d'intérêt (prostate & lésion)
    slice_z = 14
    raw_slice = t2w_raw[slice_z]

    # Style global pour diapositives
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11

    # =========================================================================
    # FIGURE 1 : Pipeline étape par étape (Step-by-Step)
    # =========================================================================
    print("Génération de Figure 1 : Pipeline étape par étape...")
    
    # Étape 1 : Masked Z-Score
    mask = t2w_raw != 0
    mean_val, std_val = t2w_raw[mask].mean(), t2w_raw[mask].std()
    t2w_zscore = np.zeros_like(t2w_raw)
    t2w_zscore[mask] = (t2w_raw[mask] - mean_val) / (std_val + 1e-8)
    slice_zscore = t2w_zscore[slice_z]

    # Étape 2 : Écrêtage Robuste [-3, +3]
    t2w_clipped = np.clip(t2w_zscore, -3.0, 3.0)
    slice_clipped = t2w_clipped[slice_z]

    # Étape 3 : Center Crop (96x96x16)
    t2w_preprocessed = center_crop_or_pad(t2w_clipped, (96, 96, 16))
    slice_preprocessed = t2w_preprocessed[10] # Coupe centrale du volume recadré

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.2, 1.2], hspace=0.35, wspace=0.25)

    # -- Ligne 1 : Images axiales à chaque étape --
    # 1. Brut
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(raw_slice, cmap='gray')
    ax1.set_title("Étape 0 : Acquisition Brute\n(Raw T2W)", fontsize=13, fontweight='bold', color='#1a365d')
    ax1.set_xlabel(f"Dimensions : 384 x 384\nIntensités : [{raw_slice.min():.0f}, {raw_slice.max():.0f}] a.u.", fontsize=10.5)
    
    # Dessiner la boîte de recadrage futur en pointillés rouges
    crop_size = 96
    start_y = (384 - crop_size) // 2
    start_x = (384 - crop_size) // 2
    rect = patches.Rectangle((start_x, start_y), crop_size, crop_size,
                             linewidth=2.2, edgecolor='#e53e3e', facecolor='none', linestyle='--',
                             label='Zone du Center-Crop')
    ax1.add_patch(rect)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Masked Z-Score
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(slice_zscore, cmap='gray')
    ax2.set_title("Étape 1 : Normalisation Masked Z-score\n$\\hat{I} = (I - \\mu) / \\sigma$", fontsize=13, fontweight='bold', color='#1a365d')
    ax2.set_xlabel(f"Fond exclu (I > 0)\n$\\mu={mean_val:.1f}, \\sigma={std_val:.1f}$", fontsize=10.5)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Intensity Clipping
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(slice_clipped, cmap='gray', vmin=-3.0, vmax=3.0)
    ax3.set_title("Étape 2 : Écrêtage Robuste\n$[-3.0\\sigma, +3.0\\sigma]$", fontsize=13, fontweight='bold', color='#1a365d')
    ax3.set_xlabel("Suppression des artefacts\net valeurs aberrantes", fontsize=10.5)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Center Crop
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(slice_preprocessed, cmap='gray', vmin=-3.0, vmax=3.0)
    ax4.set_title("Étape 3 : Harmonisation Spatiale\nCenter-Crop (96 x 96 x 16)", fontsize=13, fontweight='bold', color='#1a365d')
    ax4.set_xlabel("Focus exclusif sur la prostate\nTenseur final pour SimpleCNN3D", fontsize=10.5)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # -- Ligne 2 : Profils et Histogrammes d'Intensités --
    # Histogramme Brut
    ax_h1 = fig.add_subplot(gs[1, 0:2])
    raw_vals = raw_slice[raw_slice > 0].ravel()
    ax_h1.hist(raw_vals, bins=60, color='#4a5568', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax_h1.set_title("Distribution des Intensités Brutes (Non-Standardisée)", fontsize=12, fontweight='bold')
    ax_h1.set_xlabel("Intensité de signal brute (Unités arbitraires scanner)")
    ax_h1.set_ylabel("Nombre de voxels")
    ax_h1.grid(True, linestyle='--', alpha=0.6)

    # Histogramme Après Normalisation & Clipping
    ax_h2 = fig.add_subplot(gs[1, 2:4])
    norm_vals = slice_preprocessed[slice_preprocessed != 0].ravel()
    ax_h2.hist(norm_vals, bins=40, range=(-3, 3), color='#3182ce', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax_h2.axvline(0, color='#e53e3e', linestyle='--', linewidth=1.5, label='Moyenne (0.0)')
    ax_h2.axvline(-3, color='#dd6b20', linestyle=':', linewidth=1.5, label='Bornes [-3, +3]')
    ax_h2.axvline(3, color='#dd6b20', linestyle=':', linewidth=1.5)
    ax_h2.set_title("Distribution Après Masked Z-Score & Écrêtage (Standardisée)", fontsize=12, fontweight='bold')
    ax_h2.set_xlabel("Intensité standardisée (Nombre d'écarts-types $\\sigma$)")
    ax_h2.set_ylabel("Nombre de voxels")
    ax_h2.legend(loc='upper right', frameon=True)
    ax_h2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("PIPELINE DE PRÉTRAITEMENT VOLUMIQUE DES IRM PROSTATIQUES (PI-CAI)", fontsize=16, fontweight='bold', y=0.98)
    fig1_path = OUTPUT_DIR / "presentation_preprocessing_step_by_step.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Sauvegardé : {fig1_path}")


    # =========================================================================
    # FIGURE 2 : Diapositive "Avant / Après" Directe et Percutante
    # =========================================================================
    print("Génération de Figure 2 : Comparatif direct Avant / Après...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    # Côté GAUCHE : AVANT
    p1, p99 = np.percentile(raw_slice, (1, 99))
    im_before = axes[0].imshow(raw_slice, cmap='gray', vmin=p1, vmax=p99)
    # Rectangle de délimitation
    rect2 = patches.Rectangle((start_x, start_y), crop_size, crop_size,
                              linewidth=3.0, edgecolor='#f56565', facecolor='none', linestyle='--',
                              label='Boîte englobante prostate (Crop 96x96)')
    axes[0].add_patch(rect2)
    axes[0].legend(loc='lower right', frameon=True, fontsize=11, facecolor='#2d3748', labelcolor='white')
    axes[0].set_title("AVANT PRÉTRAITEMENT\n(Acquisition Clinique Brute)", fontsize=15, fontweight='bold', color='#c53030', pad=12)
    
    # Encadré caractéristiques
    info_before = (
        "CARACTÉRISTIQUES BRUTES :\n"
        "• Dimensions spatiales : 384 x 384 x 21 voxels\n"
        "• Échelle d'intensité : Hétérogène [0, 1850] a.u.\n"
        "• Volume total : ~3 100 000 voxels\n"
        "• Bruits : Air ambiant, vessie, os pubien, graisse\n"
        "• Variabilité : Dépend du constructeur (Siemens/Philips)"
    )
    axes[0].text(0.03, 0.05, info_before, transform=axes[0].transAxes,
                 fontsize=10.5, color='white', verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a202c', alpha=0.88, edgecolor='#e2e8f0'))
    axes[0].axis('off')

    # Côté DROIT : APRÈS
    im_after = axes[1].imshow(slice_preprocessed, cmap='gray', vmin=-3.0, vmax=3.0)
    axes[1].set_title("APRÈS PRÉTRAITEMENT\n(Tenseur Standardisé pour l'IA)", fontsize=15, fontweight='bold', color='#2b6cb0', pad=12)
    
    info_after = (
        "CARACTÉRISTIQUES PRÉTRAITÉES :\n"
        "• Dimensions spatiales : 96 x 96 x 16 voxels\n"
        "• Échelle d'intensité : Standardisée [-3.0σ, +3.0σ]\n"
        "• Volume total : 147 456 voxels (-95% d'empreinte mémoire)\n"
        "• Focus : Centré strictement sur la prostate\n"
        "• Invariance : Indépendant du constructeur et de la bobine"
    )
    axes[1].text(0.03, 0.05, info_after, transform=axes[1].transAxes,
                 fontsize=10.5, color='white', verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a202c', alpha=0.88, edgecolor='#e2e8f0'))
    axes[1].axis('off')

    plt.suptitle("COMPARAISON DIRECTE : VOLUME CLINIQUE BRUT vs ENTRÉE MODÈLE 3D", fontsize=16, fontweight='bold', y=0.97)
    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "presentation_before_after_direct.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Sauvegardé : {fig2_path}")


    # =========================================================================
    # FIGURE 3 : Harmonisation Multimodale (T2W, ADC, HBV)
    # =========================================================================
    print("Génération de Figure 3 : Harmonisation multimodale...")
    
    # Preprocess des 3 séquences
    prep_t2w = load_and_normalize_volume(t2w_path, target_shape=(96, 96, 16))
    prep_adc = load_and_normalize_volume(adc_path, target_shape=(96, 96, 16))
    prep_hbv = load_and_normalize_volume(hbv_path, target_shape=(96, 96, 16))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Ligne 1 : AVANT (Images brutes avec dimensions et résolutions hétérogènes)
    # T2W brut
    axes[0, 0].imshow(t2w_raw[slice_z], cmap='gray')
    axes[0, 0].set_title("T2W Brut\nMatrice : 384 x 384 | Voxel : 0.5 mm", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("AVANT\n(Acquisitions Cliniques Brutes)", fontsize=13, fontweight='bold', color='#c53030')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # ADC brut
    axes[0, 1].imshow(adc_raw[slice_z], cmap='gray')
    axes[0, 1].set_title("ADC Brut\nMatrice : 128 x 120 | Voxel : 2.0 mm", fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # HBV brut
    axes[0, 2].imshow(hbv_raw[slice_z], cmap='gray')
    axes[0, 2].set_title("HBV Brut\nMatrice : 128 x 120 | Voxel : 2.0 mm", fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # Ligne 2 : APRÈS (Toutes harmonisées à 96x96x16, normalisées [-3, 3])
    axes[1, 0].imshow(prep_t2w[10], cmap='gray', vmin=-3.0, vmax=3.0)
    axes[1, 0].set_title("T2W Prétraité\nMatrice : 96 x 96 | Clip [-3σ, +3σ]", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("APRÈS\n(Canaux Prêts pour Fusion)", fontsize=13, fontweight='bold', color='#2b6cb0')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    axes[1, 1].imshow(prep_adc[10], cmap='gray', vmin=-3.0, vmax=3.0)
    axes[1, 1].set_title("ADC Prétraité\nMatrice : 96 x 96 | Clip [-3σ, +3σ]", fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    axes[1, 2].imshow(prep_hbv[10], cmap='gray', vmin=-3.0, vmax=3.0)
    axes[1, 2].set_title("HBV Prétraité\nMatrice : 96 x 96 | Clip [-3σ, +3σ]", fontsize=12, fontweight='bold')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    plt.suptitle("HARMONISATION MULTIMODALE : MISE EN CONCORDANCE SPATIALE & D'INTENSITÉ POUR LA FUSION", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig3_path = OUTPUT_DIR / "presentation_multimodal_harmonization.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Sauvegardé : {fig3_path}")

if __name__ == "__main__":
    make_figures()
