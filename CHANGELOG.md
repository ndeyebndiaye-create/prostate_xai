# CHANGELOG — `prostate_xai`

Historique des décisions et résultats, par phase.
Format : `## [Phase] — Date — Résultat`

> Convention : **ne jamais modifier une entrée passée**. Ajouter de nouvelles
> entrées en bas. Les décisions négatives ("X n'a pas marché parce que Y") ont
> autant de valeur que les positives — elles évitent de ré-explorer la même impasse.

---

## [Init] — 2026-09-26 — Structure de base

**Décisions** :
- Données PI-CAI accédées via rclone (remote `gdrive_thesis:`) monté sur `~/gdrive_mount/thesis_data`
- Label binaire : ISUP ≥ 3 = csPCa (seuil EAU), défini une seule fois dans `picai_paths.py`
- Validation croisée stratifiée 5-fold au niveau patient (jamais au niveau coupe)
- PROSTATEx exclu de tout entraînement (chevauchement confirmé avec PI-CAI)

**Fichiers créés** :
- `src/config.py`, `src/data/picai_paths.py`, `src/data/unimodal_dataset.py`
- `src/models/simple_cnn3d.py`, `src/training/train_unimodal_baseline.py`
- `tests/test_picai_paths.py` (6 tests)

**Tests** : 6/6 passent ✅

---

## [Tâches 1-3] — 2026-09-26 — Baselines multimodales + logging

**Décisions** :
- `src/data/preprocessing.py` créé comme source unique de la logique de normalisation
  (masked z-score + center-crop/pad) — jamais dupliqué dans les datasets
- `PicaiMultimodalDataset` paramétré par liste de modalités (couvre Phase 2 et 3 sans modification)
- `results/baselines.csv` : append-only, 5 lignes fold + 1 ligne résumé par expérience
- Noms d'expérience : `phase1_{modality}_only`, `fusion_simple` (pour les 3 séquences)

**Fichiers créés** :
- `src/data/preprocessing.py`
- `src/data/multimodal_dataset.py`
- `src/utils/results_logger.py`
- `src/training/train_fusion_baseline.py`
- `scripts/compare_baselines.py`
- `tests/test_new_modules.py` (21 tests)

**Fichiers modifiés** :
- `src/data/unimodal_dataset.py` — délègue preprocessing, API identique
- `src/training/train_unimodal_baseline.py` — appelle `save_results()` à la fin

**Tests** : 27/27 passent ✅

**Résultats** : aucun (montage rclone non actif pendant l'implémentation)

---

## [Bonnes pratiques] — 2026-09-26 — Infrastructure de reproductibilité

**Décisions** :
- Option A retenue pour les configs : argparse reste, `--config YAML` est optionnel
  (priorité : CLI > YAML > defaults argparse)
- Seed globale (42 par défaut) couvre : random, numpy, torch, cuda, cudnn, PYTHONHASHSEED
- `git_hash` ajouté dans chaque ligne CSV → lien direct entre résultat et code
- Checkpoints sauvegardés dans `checkpoints/` (dans .gitignore) au meilleur AUROC par fold
- `evaluate()` factorisé dans `training_utils.py` (plus de duplication entre scripts)
- Confusion matrix (sensibilité/spécificité) loggée dans la console à chaque fold
- Guard anti-fuite : `assert_no_patient_overlap()` appelé avant chaque fold
- `torch.cuda.empty_cache()` entre les folds (gestion VRAM)

**Fichiers créés** :
- `src/utils/seed.py`
- `src/utils/fold_guard.py`
- `src/utils/checkpointing.py`
- `src/utils/training_utils.py`
- `src/utils/config_loader.py`
- `configs/phase1_t2w.yaml`, `configs/phase1_adc.yaml`, `configs/phase1_hbv.yaml`
- `configs/phase3_fusion_simple.yaml`
- `CHANGELOG.md` (ce fichier)

**Fichiers modifiés** :
- `src/utils/results_logger.py` — +colonnes git_hash, seed, config_file
- `src/training/train_unimodal_baseline.py` — intègre toutes les bonnes pratiques
- `src/training/train_fusion_baseline.py` — idem
- `tests/test_new_modules.py` — +14 tests (seed, fold_guard, training_utils, repro CSV)
- `.gitignore` — +checkpoints/, *.pt, *.pth

**Tests** : 40/40 passent ✅

---

## [Phase 1 T2W] — 2026-09-29 — Baseline unimodale T2W

**Contexte** : Entraînement baseline unimodal sur la séquence IRM T2W avec CV 5-fold stratifiée.
**Fichier de config** : `configs/phase1_t2w.yaml`
**Résultat** : AUROC = **0.717 ± 0.114** (5 folds, seed=42, git=bf67ce3)
- Fold 1: 0.534
- Fold 2: 0.714
- Fold 3: 0.752
- Fold 4: 0.694
- Fold 5: 0.891

**Fichiers générés** :
- `results/baselines.csv` (lignes 1 à 5 + ligne résumé)
- `checkpoints/phase1_t2w_only_fold*_best.pt`

**Prochaine étape** : Lancer la Phase 1 pour les modalités ADC (`configs/phase1_adc.yaml`) et HBV (`configs/phase1_hbv.yaml`).

---

## [Infra RunPod & Dataset PI-CAI] — 2026-11-03 — Récupération optimisée Google Drive & préparation GPU

**Contexte** : Déploiement sur serveur GPU (NVIDIA RTX PRO 4000 Blackwell 24 Go, PyTorch 2.8.0+cu128) et rapatriement du dataset PI-CAI depuis Google Drive (`gdrive_thesis:thesis_data/picai`) vers `/workspace/data/picai`.

**Décisions** :
- Filtrage `rclone copy` avec `--include "*_{t2w,adc,hbv,0000,0001,0002}.mha"` et `--fast-list --transfers 8 --tpslimit 8` : exclusion des volumes coronaux/sagittaux (`_cor.mha`, `_sag.mha`, 63 % du poids image non utilisé) et des ~10 000 masques `.nii.gz` (`anatomical_delineations`), réduisant le volume transféré de 20,02 GiB (16 354 fichiers) à **9,61 GiB (3 432 fichiers)** et évitant le blocage `rateLimitExceeded` de l'API Google Drive.
- Validation de la cohorte locale : **1 180 patients complets** (3 modalités T2W, ADC, HBV présentes, 0 patient incomplet ; 845 `label=0`, 335 `label=1` avec seuil `ISUP >= 2`, ratio `2.52:1`).
- Vérification de la suite de tests : **42/42 tests unitaires passent** ✅.
- Benchmark GPU mesuré sur Fold 1 T2W : **~34,5 s / epoch** (~17 min par fold de 30 epochs).

**Prochaine étape** : Exécuter les 4 entraînements complets (`phase1_t2w`, `phase1_adc`, `phase1_hbv`, `phase3_fusion_simple`) et consigner les AUROC finaux ci-dessous.

---

## [Phases 1 & 3 — Baselines C1 à C4] — 2026-11-04 — Résultats 5-Fold CV sur 1 180 patients PI-CAI

**Contexte** : Exécution complète des 4 baselines (30 epochs × 5 folds stratifiés au niveau patient, `seed=42`, `git=b8379ec`, `ISUP >= 2`, `SimpleCNN3D`, entrée `(96, 96, 16)`) sur les 1 180 patients complets de PI-CAI.

**Résultats (`results/baselines.csv`)** :
- **C1 — `phase1_t2w_only` (`t2w`)** : AUROC = **`0.6917 ± 0.0410`** (Folds : `0.7162`, `0.6893`, `0.6154`, `0.7347`, `0.7031`)
- **C2 — `phase1_adc_only` (`adc`)** : AUROC = **`0.6405 ± 0.0353`** (Folds : `0.5922`, `0.6172`, `0.6347`, `0.6659`, `0.6924`)
- **C3 — `phase1_hbv_only` (`hbv`)** : AUROC = **`0.6167 ± 0.0205`** (Folds : `0.6373`, `0.5979`, `0.6160`, `0.6418`, `0.5904`)
- **C4 — `fusion_simple` (`t2w+adc+hbv`, concaténation précoce)** : AUROC = **`0.6708 ± 0.0364`** (Folds : `0.6622`, `0.6720`, `0.6128`, `0.7266`, `0.6803`)

**Observations scientifiques (concordance avec la revue de littérature, Chapitres 2 & 3)** :
1. **Hiérarchie unimodale (`T2W > ADC > HBV`)** : reproduit exactement le constat de **Bertelli et al. (2022)** (Section 2.4.1 du mémoire) où le T2W seul (`0.6917`) domine l'ADC seul (`0.6405`) et le HBV seul (`0.6167`) grâce à sa résolution spatiale native supérieure (`0.28 mm` vs `1.75 mm`).
2. **Limite de la concaténation précoce (`C4 = 0.6708 < C1 = 0.6917`)** : confirme empiriquement l'analyse de **Bertelli et al. (2022)** (Section 2.4.3) et de **Manzoor et al. (2025)** (Section 2.5.1 / Section 4.4.1) : empiler directement les canaux en entrée d'un CNN unique sans recalage physique inter-séquence (`sitk.Resample` d'ADC/HBV vers la grille T2W) ni encodeurs dédiés dégrade légèrement le signal T2W au lieu d'exploiter la complémentarité multiparamétrique.

**Prochaine étape** : Aligner le pipeline de prétraitement et l'architecture (Phase 4 & 5) sur le protocole exact des Chapitres 3 et 4 du mémoire (recalage physique `SimpleITK` vers l'espace T2W, fenêtre `19 × 128 × 128`, encodeurs `SegResNet` 3D par modalité, MLP clinique `[âge, log(PSA), PSAD, volume]` et fusion par attention croisée bidirectionnelle).

---

## [Phases 4 & 5 — Architecture Proposée] — 2026-11-04 — Encodeurs SegResNet 3D, Recalage SimpleITK, MLP Clinique & Attention Croisée

**Contexte** : Implémentation de l'étape suivante conformément à la séquence de construction du projet et aux Chapitres 3 (Section 3.3.1, Tableau 3.1) et 4 (Sections 4.1 à 4.5) du mémoire.

**Décisions & Implémentations** :
- **Recalage physique (`src/data/preprocessing.py`)** : ajout de `resample_to_reference()` via `SimpleITK.Resample` pour aligner physiquement `ADC` et `HBV` sur la grille spatiale (origine, espacement, direction, taille) du volume `T2W` de chaque patient avant normalisation et recadrage (`align_to_t2w=True`).
- **Variables cliniques sans fuite (`src/data/multimodal_dataset.py`)** : ajout de `fit_transform_clinical_features()` qui extrait `[patient_age, log(PSA), PSAD, prostate_volume]` et calcule l'imputation et la standardisation z-score exclusivement sur le pli d'entraînement (`train_ids`) avant application au pli de validation (`val_ids`).
- **Encodeurs 3D & MLP Clinique (`src/models/segresnet_encoder.py`)** :
  - `SegResNetEncoder3D` basé sur `monai.networks.nets.SegResNet` (`spatial_dims=3, init_filters=16, blocks_down=(1,2,2,4)`), tête de segmentation supprimée, conservation de `last_feature_map` pour Grad-CAM++ 3D par modalité.
  - `ClinicalEncoderMLP` (`4 -> 64 -> 64` avec `LayerNorm + GELU + Dropout`).
- **Fusion Multimodale & Gating (`src/models/cross_attention_fusion.py`)** :
  - `MultimodalCrossAttentionNet` supportant **Model-U** (`shared_weights=False`) et **Model-S** (`shared_weights=True`), l'attention croisée bidirectionnelle Image $\leftrightarrow$ Clinique (`BidirectionalCrossAttention`), le bloc Transformer léger (`LightweightTransformerBlock`), le `ModalityGating` et le masque de disponibilité `availability_mask` (`Modality Dropout`).
- **Script d'entraînement & Configs YAML (`src/training/train_cross_attention.py`)** :
  - `configs/phase4_cross_attention.yaml` (Image seule : SegResNet 3D + Cross-Attention + Gating)
  - `configs/phase5_c5_concat_clinical.yaml` (Config C5 : SegResNet 3D + Clinique + Concaténation tardive)
  - `configs/phase5_c6_attention_clinical.yaml` (Config C6 : SegResNet 3D + Clinique + Self-Attention)
  - `configs/phase5_c7_cross_attention_clinical.yaml` (Config C7 : Architecture complète proposée)

**Tests** : 48/48 passent ✅ (`tests/test_cross_attention_phase4.py` ajouté).

**Prochaine étape** : Lancer les entraînements Phase 4 (`phase4_cross_attention.yaml`) et Phase 5 (`C5`, `C6`, `C7`) et comparer les AUROC obtenus aux baselines `C1`–`C4`.

---

## Template pour la prochaine entrée

```
## [Phase X] — AAAA-MM-JJ — Description courte

**Contexte** : ...
**Décision** : ...
**Résultat** : AUROC = X.XXX ± X.XXX (5 folds, seed=42, git=XXXXXXX)
**Observation** : ...
**Prochaine étape** : ...
```

