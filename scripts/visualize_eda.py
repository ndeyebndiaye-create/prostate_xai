"""
Script de génération des graphiques d'Analyse Exploratoire des Données (EDA).
Analyse le marksheet clinique du dataset PI-CAI :
- Distribution des classes cibles (csPCa) selon ISUP >= 2 et ISUP >= 3
- Distribution des covariables cliniques : Âge, PSA, PSAD, Volume prostatique, PI-RADS, Centres
- Matrice de corrélation clinique
Sauvegarde les figures dans docs/figures/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Style visuel soigné pour publication / mémoire
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MARKSHEET_PATH = Path("/home/binta-ndiayef/gdrive_mount/thesis_data/picai/picai_labels/clinical_information/marksheet.csv")
if not MARKSHEET_PATH.exists():
    MARKSHEET_PATH = Path("https://raw.githubusercontent.com/DIAGNijmegen/picai_labels/main/clinical_information/marksheet.csv")

def run_eda():
    print(f"Chargement des données cliniques depuis {MARKSHEET_PATH}...")
    df = pd.read_csv(MARKSHEET_PATH)
    
    # 1. Préparation des variables cibles
    df["csPCa_isup2"] = (df["case_ISUP"].fillna(0).astype(int) >= 2).astype(int)
    df["csPCa_isup3"] = (df["case_ISUP"].fillna(0).astype(int) >= 3).astype(int)
    
    total_patients = len(df)
    n_pos_isup2 = df["csPCa_isup2"].sum()
    n_neg_isup2 = total_patients - n_pos_isup2
    
    print(f"Nombre total de cas : {total_patients}")
    print(f"csPCa (ISUP >= 2) : {n_pos_isup2} ({n_pos_isup2/total_patients*100:.1f}%) | Non-csPCa : {n_neg_isup2} ({n_neg_isup2/total_patients*100:.1f}%)")
    
    # ── Figure 1 : Distribution des Classes et Équilibre du Dataset ─────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Diagramme en barres
    palette = ["#2b5c8f", "#d95f02"]
    counts = [n_neg_isup2, n_pos_isup2]
    labels = [f"Non-csPCa (ISUP < 2)\nN = {n_neg_isup2}\n({n_neg_isup2/total_patients*100:.1f}%)",
              f"csPCa (ISUP >= 2)\nN = {n_pos_isup2}\n({n_pos_isup2/total_patients*100:.1f}%)"]
    
    bars = axes[0].bar(labels, counts, color=palette, width=0.55, edgecolor='black', alpha=0.85)
    axes[0].set_title("Distribution Binaire de la Cible (Seuil ISUP $\geq$ 2)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel("Nombre de Patients", fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajout des valeurs au dessus des barres
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Camembert Seuil ISUP >= 2
    explode = (0, 0.08)
    axes[1].pie(counts, explode=explode, labels=['Non-csPCa (71.7%)', 'csPCa (28.3%)'],
                colors=palette, autopct='%1.1f%%', shadow=True, startangle=140,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title(f"Ratio de Déséquilibre : {n_neg_isup2/n_pos_isup2:.2f} : 1", fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "eda_class_distribution.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Sauvegardé : {fig1_path}")
    
    # ── Figure 2 : Caractéristiques Cliniques Clés ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Âge du patient selon le statut csPCa
    sns.histplot(data=df, x='patient_age', hue='csPCa_isup2', multiple='stack',
                 palette=palette, bins=20, ax=axes[0, 0], kde=True)
    axes[0, 0].set_title("Distribution de l'Âge par Statut de Cancer", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Âge du patient (années)")
    axes[0, 0].set_ylabel("Nombre de patients")
    axes[0, 0].legend(title="Cancer", labels=["csPCa (ISUP $\geq$ 2)", "Non-csPCa"])
    
    # 2. PSA (valeur sérique) avec échelle logarithmique
    df_psa = df.dropna(subset=['psa']).copy()
    sns.boxplot(data=df_psa, x='csPCa_isup2', y='psa', palette=palette, ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title("Distribution du PSA (Échelle Log, hors extrêmes)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xticklabels(["Non-csPCa", "csPCa (ISUP $\geq$ 2)"])
    axes[0, 1].set_ylabel("PSA sérique (ng/mL)")
    axes[0, 1].set_xlabel("Classe")
    
    # 3. Distribution des scores PI-RADS
    if 'lesion_PIRADS' in df.columns:
        df_pirads = df.dropna(subset=['lesion_PIRADS']).copy()
        df_pirads['lesion_PIRADS'] = df_pirads['lesion_PIRADS'].astype(str)
        sns.countplot(data=df_pirads, x='lesion_PIRADS', hue='csPCa_isup2',
                      palette=palette, ax=axes[1, 0], order=['1', '2', '3', '4', '5'])
        axes[1, 0].set_title("Distribution des Scores PI-RADS selon le Statut", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Score PI-RADS (Lésion index)")
        axes[1, 0].set_ylabel("Nombre de cas")
        axes[1, 0].legend(title="Statut", labels=["Non-csPCa", "csPCa"])
    
    # 4. Distribution par Centre d'Imagerie Hospitalier
    if 'center' in df.columns:
        sns.countplot(data=df, x='center', hue='csPCa_isup2', palette=palette, ax=axes[1, 1])
        axes[1, 1].set_title("Répartition des Patients par Centre Médical", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Centre hospitalier")
        axes[1, 1].set_ylabel("Nombre d'examens")
        axes[1, 1].legend(title="Statut", labels=["Non-csPCa", "csPCa"])
    
    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "eda_clinical_covariates.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Sauvegardé : {fig2_path}")

    # ── Figure 3 : Matrice de Corrélation Clinique ─────────────────────────
    numeric_cols = ['patient_age', 'psa', 'psad', 'prostate_volume', 'case_ISUP', 'csPCa_isup2']
    valid_cols = [c for c in numeric_cols if c in df.columns]
    corr_df = df[valid_cols].apply(pd.to_numeric, errors='coerce').corr()
    
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmin=-0.5, vmax=1.0, center=0,
                square=True, linewidths=.7, cbar_kws={"shrink": .8}, annot=True, fmt=".2f")
    plt.title("Matrice de Corrélation des Variables Cliniques (PI-CAI)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    fig3_path = OUTPUT_DIR / "eda_correlation_matrix.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Sauvegardé : {fig3_path}")

if __name__ == "__main__":
    run_eda()
