"""
Visualisation des images IRM multiparamétriques (T2W, ADC, HBV) et du masque tumoral (csPCa).
Affiche :
1. Vue comparative multi-séquences sur la coupe de la lésion index (T2W, ADC, HBV, T2W + Masque).
2. Vue multi-coupes de la lésion (coupes axiales 13, 14, 15).
3. Visualisation du tenseur prétraité (96x96x16) après Masked Z-Score, tel que reçu par le réseau CNN 3D.
"""

import sys
from pathlib import Path

# Assure l'accès au module src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from src.data.preprocessing import load_and_normalize_volume

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_patient_10008"
MASK_PATH = Path("/home/binta-ndiayef/gdrive_mount/thesis_data/picai/picai_labels/csPCa_lesion_delineations/human_expert/Pooch25/10008_1000008.nii.gz")

def normalize_display(img_slice):
    """Normalisation min-max pour un affichage contrasté."""
    p1, p99 = np.percentile(img_slice, (1, 99))
    clipped = np.clip(img_slice, p1, p99)
    return (clipped - p1) / (p99 - p1 + 1e-8)

def visualize_mri():
    print("Chargement des volumes IRM du patient 10008...")
    t2w_img = sitk.ReadImage(str(DATA_DIR / "10008_1000008_t2w.mha"))
    adc_img = sitk.ReadImage(str(DATA_DIR / "10008_1000008_adc.mha"))
    hbv_img = sitk.ReadImage(str(DATA_DIR / "10008_1000008_hbv.mha"))
    mask_img = sitk.ReadImage(str(MASK_PATH))

    # Conversion en tableaux NumPy (Z, Y, X)
    t2w_arr = sitk.GetArrayFromImage(t2w_img)
    adc_arr = sitk.GetArrayFromImage(adc_img)
    hbv_arr = sitk.GetArrayFromImage(hbv_img)
    mask_arr = sitk.GetArrayFromImage(mask_img)

    # ── Figure 1 : Comparaison Multimodale mpMRI + Masque Délimitation ──────
    slice_idx = 14  # Coupe centrale de la lésion
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # T2W
    axes[0].imshow(normalize_display(t2w_arr[slice_idx]), cmap="gray")
    axes[0].set_title("T2-Weighted (T2W)\nAnatomie & Capsule Prostatique", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # ADC
    axes[1].imshow(normalize_display(adc_arr[slice_idx]), cmap="gray")
    axes[1].set_title("Apparent Diffusion (ADC)\nHypo-signal tumoral (noir)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # HBV
    axes[2].imshow(normalize_display(hbv_arr[slice_idx]), cmap="gray")
    axes[2].set_title("High b-value (HBV)\nHyper-signal tumoral (blanc)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # T2W + Masque Expert
    axes[3].imshow(normalize_display(t2w_arr[slice_idx]), cmap="gray")
    masked_roi = np.ma.masked_where(mask_arr[slice_idx] == 0, mask_arr[slice_idx])
    axes[3].imshow(masked_roi, cmap="autumn", alpha=0.6)
    axes[3].set_title("T2W + Délimitation Expert (csPCa)\nLésion index (Rouge)", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle("Patient PI-CAI #10008 — Visualisation Multimodale mpMRI (Coupe Axiale z=14)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "mri_multimodal_patient10008.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Sauvegardé : {fig1_path}")

    # ── Figure 2 : Évolution de la Lésion sur les 3 Coupes Clés (13, 14, 15) ─
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    slices = [13, 14, 15]

    for row_idx, s in enumerate(slices):
        # T2W
        axes[row_idx, 0].imshow(normalize_display(t2w_arr[s]), cmap="gray")
        axes[row_idx, 0].set_ylabel(f"Coupe z={s}", fontsize=12, fontweight="bold")
        if row_idx == 0:
            axes[row_idx, 0].set_title("T2W", fontsize=13, fontweight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # ADC
        axes[row_idx, 1].imshow(normalize_display(adc_arr[s]), cmap="gray")
        if row_idx == 0:
            axes[row_idx, 1].set_title("ADC", fontsize=13, fontweight="bold")
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # HBV
        axes[row_idx, 2].imshow(normalize_display(hbv_arr[s]), cmap="gray")
        if row_idx == 0:
            axes[row_idx, 2].set_title("HBV", fontsize=13, fontweight="bold")
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])

        # Overlay Masque
        axes[row_idx, 3].imshow(normalize_display(t2w_arr[s]), cmap="gray")
        m_roi = np.ma.masked_where(mask_arr[s] == 0, mask_arr[s])
        axes[row_idx, 3].imshow(m_roi, cmap="autumn", alpha=0.6)
        if row_idx == 0:
            axes[row_idx, 3].set_title("T2W + Vérité Terrain", fontsize=13, fontweight="bold")
        axes[row_idx, 3].set_xticks([])
        axes[row_idx, 3].set_yticks([])

    plt.suptitle("Extension Volumique de la Lésion csPCa à Travers les Coupes Axiales", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "mri_lesion_slices_progression.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Sauvegardé : {fig2_path}")

    # ── Figure 3 : Volume après Prétraitement Standard (Entrée du CNN) ───────
    # Passage dans la fonction de preprocessing officielle du projet
    prep_np = load_and_normalize_volume(
        DATA_DIR / "10008_1000008_t2w.mha",
        target_shape=(96, 96, 16),
    )  # Forme (16, 96, 96)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # Sélection de 8 coupes représentatives parmi les 16
    sampled_slices = np.linspace(2, 14, 8, dtype=int)
    for i, s in enumerate(sampled_slices):
        ax = axes[i // 4, i % 4]
        im = ax.imshow(prep_np[s], cmap="gray", vmin=-3.0, vmax=3.0)
        ax.set_title(f"Coupe {s+1}/16 (z-score)", fontsize=11, fontweight="bold")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.04, pad=0.08)
    cbar.set_label("Intensité Normalisée (Masked Z-Score écrêté dans [-3.0, +3.0])", fontsize=11)
    plt.suptitle("Volume T2W Prétraité (Entrée Tensorielle du SimpleCNN3D : 96x96x16)", fontsize=14, fontweight="bold", y=0.99)
    fig3_path = OUTPUT_DIR / "mri_preprocessed_input_tensor.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Sauvegardé : {fig3_path}")

if __name__ == "__main__":
    visualize_mri()
