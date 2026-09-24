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
- [ ] Step 2 — Unimodal baselines: T2W, ADC, HBV separately (in progress)
- [ ] Step 3 — Simple multimodal fusion baseline (early concatenation)
- [ ] Step 4 — Proposed architecture: per-modality encoders + cross-attention
- [ ] Step 5 — Add clinical variables, compare image-only vs image+clinical
- [ ] Step 6 — Grad-CAM++, SHAP, uncertainty estimation
- [ ] Step 7 — Robustness / domain-shift experiments
- [ ] Step 8 — Storage layer (Qdrant + PostgreSQL) and API — only once a
      trained model actually produces embeddings/predictions worth storing

Steps 1-7 are pure ML (`src/`) and do not need Docker at all. Step 8
(`docker-compose.yml`, `backend/`) is scaffolded now but **not meant to run
yet** — there is nothing to store until a model exists.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install ML dependencies
pip install -r requirements.txt

# 3. Configure your local Drive Desktop path
cp .env.example .env
# then edit .env and set DATA_ROOT to your actual synced Drive folder, e.g.:
#   macOS:   /Users/<you>/Google Drive/My Drive/thesis_data
#   Windows: G:\My Drive\thesis_data
#   Linux:   /home/<you>/GoogleDrive/thesis_data
```

```bash
# 4. Run the test suite (uses synthetic data only, no dependency on DATA_ROOT)
pytest tests/ -v
```

```bash
# 5. Run a baseline training (reads DATA_ROOT from .env automatically)
python -m src.training.train_unimodal_baseline --modality t2w
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

- **Patient-level classification only.** Label 0 = ISUP ≤ 2, Label 1 = ISUP ≥ 3.
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
