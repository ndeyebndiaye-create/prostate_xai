# Guide Complet d'Exécution et de Reproduction du Projet (Local & Serveur)

> **Projet de Mémoire : Détection du Cancer Cliniquement Significatif de la Prostate (csPCa) par Imagerie Multimodale & IA Explicable (XAI)**  
> **Auteure** : Binta Ndiaye — AIMS Sénégal  
> Ce document est le manuel de référence pour **exécuter, reproduire et comprendre l'ensemble du projet sans dépendance exclusive à RunPod**. Il fonctionne sur machine locale (Ubuntu, macOS, Windows WSL2), sur serveur universitaire ou sur cluster de calcul.

---

## Sommaire
1. [Architecture et Prérequis Système](#1-architecture-et-prérequis-système)
2. [Installation Pas à Pas de l'Environnement](#2-installation-pas-à-pas-de-lenvironnement)
3. [Configuration de l'Accès aux Données (Local vs Google Drive)](#3-configuration-de-laccès-aux-données)
4. [Validation Immédiate de l'Installation (Tests Unitaires)](#4-validation-immédiate-de-linstallation)
5. [Analyse Exploratoire des Données (EDA) & Graphiques](#5-analyse-exploratoire-des-données-eda)
6. [Affichage et Inspection des Images IRM & Masques](#6-affichage-et-inspection-des-images-irm--masques)
7. [Lancement des Entraînements (Unimodal & Fusion)](#7-lancement-des-entraînements)
8. [Visualisation et Analyse des Résultats](#8-visualisation-et-analyse-des-résultats)
9. [Dépannage et FAQ](#9-dépannage-et-faq)

---

## 1. Architecture et Prérequis Système

* **Système d'Exploitation** : Linux (recommandé), macOS ou Windows avec WSL2.
* **Version Python** : Python $\ge$ 3.10 (testé et validé sous Python 3.14).
* **Ressources Matérielles** :
  * **GPU** : CUDA compatible (NVIDIA RTX 3080/4090/A100 ou équivalent) pour l'entraînement complet.
  * **CPU** : Possible pour l'EDA, l'affichage d'images, les tests unitaires et les entraînements légers (1-2 epochs).
* **Stockage** :
  * Mode streaming RAM : **< 2 Go** d'espace disque local requis.
  * Mode dataset complet local : ~60 Go d'espace disque.

---

## 2. Installation Pas à Pas de l'Environnement

### Étape 2.1 : Cloner le dépôt et se positionner à la racine
```bash
cd ~/Harness_memoire/prostate_xai
```

### Étape 2.2 : Créer et activer l'environnement virtuel
```bash
# Création du venv
python3 -m venv .venv

# Activation
source .venv/bin/activate
# (Sur Windows CMD : .venv\Scripts\activate.bat)
# (Sur Windows PowerShell : .venv\Scripts\Activate.ps1)
```

### Étape 2.3 : Mettre à jour `pip` et installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configuration de l'Accès aux Données

Le code supporte de manière transparente **deux modes d'accès aux données**.

### Option A : Données stockées sur disque local
Si vous avez téléchargé le dataset PI-CAI sur votre disque :
1. Créez votre fichier `.env` depuis le template :
   ```bash
   cp .env.example .env
   ```
2. Éditez `.env` et indiquez le chemin absolu vers votre dossier :
   ```env
   DATA_ROOT=/chemin/vers/votre/dossier/thesis_data
   ```
   L'arborescence attendue sous `thesis_data/` est :
   ```text
   thesis_data/
   └── picai/
       ├── images/
       │   └── <patient_id>/
       │       ├── <patient_id>_<study_id>_t2w.mha
       │       ├── <patient_id>_<study_id>_adc.mha
       │       └── <patient_id>_<study_id>_hbv.mha
       └── picai_labels/
           └── clinical_information/
               └── marksheet.csv
   ```

### Option B : Données sur Google Drive (Streaming RAM via `rclone`)
Si votre disque local est limité, le pipeline sait charger les volumes `.mha` **directement dans la mémoire RAM via `rclone` sans occuper d'espace disque** :
1. Installer `rclone` (ex: `sudo apt install rclone` ou `brew install rclone`).
2. Configurer un remote nommé `gdrive_thesis` pointant vers le Google Drive contenant le dossier `thesis_data` :
   ```bash
   rclone config
   ```
   *(Suivre l'assistant interactif pour associer votre compte Google Drive)*.
3. Vérifier l'accès :
   ```bash
   rclone lsf gdrive_thesis:thesis_data/picai/
   ```

---

## 4. Validation Immédiate de l'Installation

Avant de lancer de gros calculs, vérifiez que tout fonctionne grâce à la suite de tests unitaires automatisés. Les tests s'exécutent sur des tenseurs synthétiques générés en mémoire et ne nécessitent aucun téléchargement :

```bash
pytest tests/ -v
```
**Résultat attendu** : `42 passed in ~10s` (100% vert).

---

## 5. Analyse Exploratoire des Données (EDA)

Pour calculer et tracer toutes les statistiques sur les 1 500 patients du marksheet PI-CAI :

```bash
python scripts/visualize_eda.py
```

### Fichiers générés dans `docs/figures/` :
1. **`eda_class_distribution.png`** : 
   - Non-csPCa (ISUP < 2) : 1 075 patients (71,7%).
   - csPCa (ISUP $\ge$ 2) : 425 patients (28,3%).
   - Ratio de déséquilibre : **2,53 : 1**.
2. **`eda_clinical_covariates.png`** : 
   - Histogramme des âges des patients.
   - Boxplot du PSA sérique (échelle log).
   - Barplot des scores PI-RADS (1 à 5).
   - Répartition par centre hospitalier d'acquisition.
3. **`eda_correlation_matrix.png`** :
   - Matrice de corrélation de Pearson entre variables cliniques et sévérité du cancer.

---

## 6. Affichage et Inspection des Images IRM & Masques

Pour inspecter visuellement les volumes multiparamétriques et la segmentation expert :

```bash
python scripts/visualize_mri_slices.py
```

### Fichiers générés dans `docs/figures/` :
1. **`mri_multimodal_patient10008.png`** :
   - Affiche côte-à-côte sur la coupe tumorale axiale : **T2W**, **ADC** (hypo-signal noir), **HBV** (hyper-signal blanc), et **T2W + Masque expert** (lésion en rouge).
2. **`mri_lesion_slices_progression.png`** :
   - Extension volumique 3D de la lésion sur 3 coupes axiales adjacentes ($z=13, 14, 15$).
3. **`mri_preprocessed_input_tensor.png`** :
   - Aperçu des 16 coupes du tenseur prétraité normalisé par *Masked Z-score* $(96, 96, 16)$, exactement tel qu'il entre dans le réseau de neurones 3D.

---

## 7. Lancement des Entraînements

### 7.1 Test Rapide (Mode Vérification — 1 ou 2 epochs)
Pour vérifier que l'entraînement tourne sur votre machine sans attendre plusieurs heures :

```bash
# Test sur 1 fold et 2 epochs
python -m src.training.train_unimodal_baseline --config configs/phase1_t2w.yaml --n_epochs 2 --n_folds 1
```

### 7.2 Entraînement Complet des Baselines Unimodales (5-Fold CV, 30 epochs)
1. **Séquence T2W (Anatomique)** :
   ```bash
   python -m src.training.train_unimodal_baseline --config configs/phase1_t2w.yaml
   ```
2. **Séquence ADC (Diffusion Apparente)** :
   ```bash
   python -m src.training.train_unimodal_baseline --config configs/phase1_adc.yaml
   ```
3. **Séquence HBV (Haute valeur b)** :
   ```bash
   python -m src.training.train_unimodal_baseline --config configs/phase1_hbv.yaml
   ```

### 7.3 Entraînement de la Fusion Multimodale (Canaux empilés)
Combine T2W + ADC + HBV en entrée du CNN 3D (3 canaux d'entrée) :
```bash
python -m src.training.train_fusion_baseline --config configs/phase3_fusion_simple.yaml
```

> **Astuce GPU vs CPU** : Le code détecte automatiquement la présence d'un GPU (`torch.cuda.is_available()`). Si aucun GPU n'est présent, il bascule proprement sur CPU avec AMP désactivé.

---

## 8. Visualisation et Analyse des Résultats

### 8.1 Générer les graphiques d'évaluation des entraînements
Une fois l'entraînement terminé (ou pour exploiter les résultats existants dans `results/baselines.csv`) :

```bash
python scripts/visualize_training_results.py
```
**Sortie générée** : `docs/figures/training_baseline_results.png` présentant le barplot des AUROC par fold, la ligne moyenne (0.7172) et la bande d'écart-type ($\pm 0.1145$).

### 8.2 Tableau comparatif texte dans le terminal
```bash
python scripts/compare_baselines.py
```

### 8.3 Emplacement des fichiers de sauvegarde
- **Checkpoints des meilleurs modèles** : `./checkpoints/{experiment_name}/fold_{N}/best.pt`
- **Historique des métriques par epoch** : `./logs/{experiment_name}_fold{N}.csv`
- **Fichier de traçabilité officiel** : `./results/baselines.csv` (enregistre date, seed, AUROC, et hash Git).

---

## 9. Dépannage et FAQ

* **Q : J'ai une erreur `ModuleNotFoundError: No module named 'src'`**  
  **R** : Assurez-vous d'être bien positionné dans le dossier racine `prostate_xai` et utilisez `python -m ...` ou préfixez avec `PYTHONPATH=.`.

* **Q : Mon GPU manque de mémoire (CUDA Out of Memory)**  
  **R** : Réduisez le `batch_size` via la ligne de commande :  
  `python -m src.training.train_unimodal_baseline --config configs/phase1_t2w.yaml --batch_size 4`

* **Q : Les fichiers checkpoints ou logs ne se créent pas**  
  **R** : Vérifiez que vous avez les droits d'écriture dans le dossier du projet. Si vous utilisez RunPod ou un dossier spécifique, vous pouvez forcer le chemin de sortie via la variable d'environnement :  
  `export RUN_OUTPUT_ROOT="/mon/chemin/de/sortie"`
