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

## Template pour la prochaine entrée

```
## [Phase X] — AAAA-MM-JJ — Description courte

**Contexte** : ...
**Décision** : ...
**Résultat** : AUROC = X.XXX ± X.XXX (5 folds, seed=42, git=XXXXXXX)
**Observation** : ...
**Prochaine étape** : ...
```

