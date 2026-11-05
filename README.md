# Explainable Multi-Modal Deep Learning for Prostate Cancer Aggressiveness (mpMRI)

Master's thesis project — AIMS Sénégal — Binta.

## Environment: local machine (VSCode), not Colab/Kaggle

This project moved from Colab to a local VSCode setup because the planned
architecture (vector DB for embeddings + relational DB for results + API +
frontend) needs multiple persistent services running together, which Colab
and Kaggle notebooks cannot do (single ephemeral kernel, no service
orchestration).

Data access: **Google Drive Desktop**. The Drive folder is synced locally by
the Drive Desktop app, so from the code's point of view it is just a normal
local path — no `drive.mount()`, no Colab-specific code anywhere in this repo.
Where that path actually points on your machine is set once, in `.env`
(see below), never hardcoded in source files.

## Status

- [x] Step 1 — Data downloaded to Drive (PI-CAI, via Zenodo), channel naming
      confirmed (`_0000`=T2W, `_0001`=ADC, `_0002`=HBV)
- [x] Step 2 — Unimodal baselines completed (5-fold CV, 1,180 patients): T2W (`0.6917 ± 0.0410`), ADC (`0.6405 ± 0.0353`), HBV (`0.6167 ± 0.0205`)
- [x] Step 3 — Simple multimodal fusion baseline completed (early concatenation, `configs/phase3_fusion_simple.yaml`: AUROC `0.6708 ± 0.0364`)
- [x] Step 4 — Proposed architecture implemented: per-modality 3D `SegResNet` encoders + `SimpleITK` physical-space alignment + cross-attention & modality gating (`configs/phase4_cross_attention.yaml`)
- [x] Step 5 — Clinical variables branch (`ClinicalEncoderMLP` on `[age, log(PSA), PSAD, prostate_volume]`) + bidirectional cross-attention configs C5, C6, C7 ready (`configs/phase5_c*.yaml`)
- [ ] Step 6 — Grad-CAM++, SHAP, uncertainty estimation
- [ ] Step 7 — Robustness / domain-shift experiments
- [ ] Step 8 — Storage layer (Qdrant + PostgreSQL) and API — only once a
      trained model actually produces embeddings/predictions worth storing

## Documentation & Guides de Référence

- 🗺️ **[Feuille de Route, Concordance Rédaction-Code & Progression](docs/FEUILLE_DE_ROUTE_ET_PROGRESSION.md)** : Suivi détaillé de toutes les étapes accomplies, concordance chapitre par chapitre avec le manuscrit de thèse (Chapitres 1 à 5) et plan d'action complet pour la suite (sur **PI-CAI** exclusivement).
- 📖 **[Guide Complet d'Exécution et Reproduction (Local / Serveur)](docs/GUIDE_COMPLET_EXECUTION_ET_REPRODUCTION.md)** : Instructions détaillées pas à pas pour installer, lancer les visualisations et exécuter les entraînements sur n'importe quel ordinateur ou serveur sans RunPod.
- 📓 **[Journal de Bord des Expérimentations](docs/JOURNAL_DE_BORD_EXPERIMENTATIONS.md)** : Registre chronologique de la thèse avec protocoles, métriques AUROC obtenues et décisions pour les étapes suivantes.
- 📊 **[Rapport Visuel EDA & Résultats](docs/EDA_AND_PIPELINE_GUIDE.md)** : Synthèse des distributions statistiques, affichage des coupes IRM axiales et courbes d'entraînement.

## Quick Start (Exécution Locale)

```bash
# 1. Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 2. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 3. Lancer la suite de tests unitaires (48 tests, 100% synthétiques)
pytest tests/ -v

# 4. Générer les visualisations graphiques
python scripts/visualize_eda.py               # Distributions cliniques & corrélations
python scripts/visualize_mri_slices.py        # Coupes axiales mpMRI & masque expert
python scripts/visualize_training_results.py  # Graphes de validation AUROC (5-Fold)

# 5. Lancer un entraînement baseline
python -m src.training.train_unimodal_baseline --config configs/phase1_t2w.yaml
```

## Repository structure

```
├── .env.example          # template for DATA_ROOT — copy to .env, never commit .env
├── docker-compose.yml     # Qdrant + PostgreSQL — for Step 8, not needed yet
├── configs/               # experiment configs (hyperparameters per run)
├── src/
│   ├── config.py          # loads DATA_ROOT from .env — single source of truth for paths
│   ├── data/               # dataset classes, path resolution, label loading
│   ├── models/             # encoders, fusion modules (grows with each step)
│   ├── training/           # training loops, CV orchestration
│   └── utils/              # metrics, seeding
└── tests/                 # unit tests on synthetic data only
```

## Design constraints (still apply, updated for local environment)

- **Patient-level classification only.** Label 0 = ISUP < 2 (bénin / indolent), Label 1 = ISUP ≥ 2 (cliniquement significatif / csPCa, Gleason 3+4+).
- **PROSTATEx is excluded** entirely — TCIA confirms it is fully absorbed into
  PI-CAI (same 344 patients), so it cannot serve as an external test set.
- **External validation candidate**: TCIA collection `Prostate-MRI-US-Biopsy`
  (to be confirmed patient-by-patient against PI-CAI before use — Step 7).
- **Senegal cohort** (Hôpital de Mbour / MADCaP) is a separate African
  external validation set, not a PI-CAI replacement — studies domain shift,
  not used for training.
- Build sequence (supervisor-approved): unimodal baselines → simple fusion
  baseline → proposed cross-attention architecture → + clinical variables →
  XAI → robustness experiments. This order exists specifically so any
  performance gain can be attributed to a specific change, not conflated.
