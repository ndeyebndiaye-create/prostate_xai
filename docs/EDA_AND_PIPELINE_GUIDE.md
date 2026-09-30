# Guide d'Exploration des Données (EDA) & Exécution du Pipeline

> **Document de Référence pour la Thèse**
> Ce fichier documente la structure des données, les statistiques clés (EDA) et l'ensemble des commandes pour exécuter et reproduire les entraînements pas à pas.

---

## 1. Analyse Exploratoire des Données (EDA - Exploratory Data Analysis)

### A. Synthèse du Dataset PI-CAI
- **Origine** : Dataset public PI-CAI (Prostate Cancer Image Analysis Standardized Evaluation).
- **Nombre total de patients (Marksheet)** : **1 500 patients**
- **Séquences IRM multiparamétriques (mpMRI)** :
  - `T2W` (`_0000` ou `_t2w.mha`) : IRM pondérée en T2 (anatomie de la prostate)
  - `ADC` (`_0001` ou `_adc.mha`) : Carte du coefficient de diffusion apparent (densité cellulaire)
  - `HBV` (`_0002` ou `_hbv.mha`) : IRM de diffusion à haut b-value (détection des lésions)

### B. Distribution des Classes (Seuil ISUP $\ge$ 2 - Option Retenue)
- **Classe 0 (Non-csPCa)** : ISUP 0 et 1 (Bénin / Gleason 3+3) $\rightarrow$ **1075 patients (71.67%)**
- **Classe 1 (csPCa)** : ISUP $\ge$ 2 (Gleason 3+4, 4+3, 4+5, etc.) $\rightarrow$ **425 patients (28.33%)**
- **Ratio de Déséquilibre** : **2.53 : 1** (gestion par CrossEntropyLoss pondérée + StratifiedKFold)

---

## 2. Guide d'Exécution des Commandes (Step-by-Step)

### Étape 0 : Se placer dans le répertoire du projet et activer l'environnement
```bash
cd ~/Harness_memoire/prostate_xai
source .venv/bin/activate
```

---

### Étape 1 : Vérifier la balance et les patients complets (EDA rapide)
```bash
python scripts/analyze_dataset_balance.py
```
*Cette commande télécharge/lit le marksheet officiel et affiche le nombre de patients par classe.*

---

### Étape 2 : Lancer la Phase 1 — Baselines Unimodales (1 seule séquence)

1. **Entraînement T2W (Séquence Principale)** :
   ```bash
   python -m src.training.train_unimodal_baseline --config configs/phase1_t2w.yaml
   ```
   *(Option rapide 5 epochs pour test : ajouter `--n_epochs 5`)*

2. **Entraînement ADC** :
   ```bash
   python -m src.training.train_unimodal_baseline --config configs/phase1_adc.yaml
   ```

3. **Entraînement HBV** :
   ```bash
   python -m src.training.train_unimodal_baseline --config configs/phase1_hbv.yaml
   ```

---

### Étape 3 : Lancer la Phase 3 — Fusion Multimodale (Canaux empilés)

```bash
python -m src.training.train_fusion_baseline --config configs/phase3_fusion_simple.yaml
```
*Combine T2W + ADC + HBV sous forme d'un tenseur 3 canaux en entrée du CNN 3D.*

---

### Étape 4 : Visualiser le Tableau Comparatif des Résultats

```bash
python scripts/compare_baselines.py
```
*Lit automatiquement `results/baselines.csv` et affiche le classement des modèles par AUROC avec la moyenne $\pm$ écart-type sur les 5 folds.*

---

### Étape 5 : Lancer les Tests Unitaires pour tout valider sans données réelles
```bash
python -m pytest tests/ -v
```
*(40/40 tests doivent passer vert)*

---

## 3. Emplacement des Fichiers et Résultats

- **Fichier de logs des résultats** : `results/baselines.csv` (contient timestamp, nom d'expérience, fold, AUROC, seed, hash Git).
- **Checkpoints des meilleurs modèles** : `checkpoints/` (`.pt` sauvegardés par fold au meilleur AUROC).
- **Fichier d'explication du code ligne par ligne** : `explication_codebase.md`.
