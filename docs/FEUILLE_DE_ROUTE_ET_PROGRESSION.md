# Feuille de Route, Suivi de Progression et Plan d'Exécution Complet

> **Projet de Mémoire — AIMS Sénégal** : *Explainable Multi-Modal Deep Learning Framework for Prostate Cancer Aggressiveness Classification from Multiparametric MRI: Toward Clinically Trustworthy AI for Low-Resource African Settings*  
> **Auteure** : Ndeye Binta NDIAYE — **Superviseur** : Dr Aurelle Tchagna  
> **Dernière mise à jour** : 4 novembre 2026

---

## 1. Vue d'Ensemble de la Progression

- [x] **Étape 1 — Rapatriement et validation de la cohorte PI-CAI**
  - `9,61 Go` transférés depuis Google Drive (`3 432` volumes `.mha` axiaux `T2W`, `ADC`, `HBV` + `marksheet.csv`).
  - **1 180 patients complets** après dédoublonnage au niveau patient (`845` négatifs `ISUP <= 1`, `335` positifs `ISUP >= 2`, ratio `2,52 : 1`).
  - **PROSTATEx totalement exclu** (on travaille exclusivement sur PI-CAI).
- [x] **Étape 2 — Entraînement des Baselines Unimodales (`C1`, `C2`, `C3` — 5-Fold CV, 30 epochs)**
  - `C1` (`phase1_t2w_only`) : **AUROC = `0.6917 ± 0.0410`**
  - `C2` (`phase1_adc_only`) : **AUROC = `0.6405 ± 0.0353`**
  - `C3` (`phase1_hbv_only`) : **AUROC = `0.6167 ± 0.0205`**
- [x] **Étape 3 — Entraînement de la Baseline de Fusion Simple (`C4` — Concaténation précoce)**
  - `C4` (`fusion_simple`, `t2w+adc+hbv`) : **AUROC = `0.6708 ± 0.0364`**
  - Reproduit empiriquement les constats de **Bertelli et al. (2022)** et **Manzoor et al. (2025)** décrits au Chapitre 2 du mémoire.
- [x] **Étape 4 — Implémentation du Recalage Physique & de l'Architecture Proposée (Chapitres 3.3 & 4)**
  - Recalage physique `SimpleITK.Resample` (`ADC` et `HBV` vers la grille spatiale `T2W`) dans `src/data/preprocessing.py`.
  - Extraction et standardisation sans fuite des 4 variables cliniques `[patient_age, log(PSA), PSAD, prostate_volume]` dans `src/data/multimodal_dataset.py`.
  - Encodeurs 3D `SegResNet` (MONAI) + `ClinicalEncoderMLP` dans `src/models/segresnet_encoder.py`.
  - Fusion par Attention Croisée Bidirectionnelle + Transformer léger + `ModalityGating` + support `Model-U` / `Model-S` et `availability_mask` dans `src/models/cross_attention_fusion.py`.
  - Suite de tests unitaires portée à **48/48 tests passés** (`tests/test_cross_attention_phase4.py`).
- [ ] **Étape 5 — Exécution des Entraînements Phase 4 (Image seule) & Phase 5 (`C5`, `C6`, `C7` + Baselines Cliniques)** *(Prêt à lancer)*
- [ ] **Étape 6 — Études d'Ablation, Métriques Cliniques Verrouillées & Calibration (`ECE`, `Brier`)**
- [ ] **Étape 7 — Explicabilité Intégrée (`Grad-CAM++` 3D par modalité, `Pointing Accuracy`, Occlusion, `SHAP`) & Incertitude (`MC Dropout`)**
- [ ] **Étape 8 — Robustesse aux Dégradations Contrôlées (Expériences 1 à 3) & Généralisation Inter-Centres (`Leave-One-Center-Out` : Expérience 4)**

---

## 2. Résultats Déjà Acquis (Baselines `C1` à `C4`)

Tous les résultats ci-dessous sont enregistrés dans `results/baselines.csv` (`seed=42`, `git=b8379ec`, 5-Fold CV stratifiée au niveau patient sur les 1 180 patients complets de PI-CAI) :

| Code Tableau 3.1 | Expérience | Modalités | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **AUROC (Moy ± Écart-type)** |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | `phase1_t2w_only` | `T2W` seul | 0.7162 | 0.6893 | 0.6154 | 0.7347 | 0.7031 | **`0.6917 ± 0.0410`** |
| **C4** | `fusion_simple` | `T2W + ADC + HBV` (concat. précoce) | 0.6622 | 0.6720 | 0.6128 | 0.7266 | 0.6803 | **`0.6708 ± 0.0364`** |
| **C2** | `phase1_adc_only` | `ADC` seul | 0.5922 | 0.6172 | 0.6347 | 0.6659 | 0.6924 | **`0.6405 ± 0.0353`** |
| **C3** | `phase1_hbv_only` | `HBV` seul | 0.6373 | 0.5979 | 0.6160 | 0.6418 | 0.5904 | **`0.6167 ± 0.0205`** |

### Pourquoi ces 4 résultats valident votre revue de littérature (Chapitre 2) :
1. **Hiérarchie unimodale (`T2W > ADC > HBV`)** : Confirme **Bertelli et al. (2022)** (Section 2.4.1, p. 13) — le T2W seul (`0.6917`) domine l'ADC seul (`0.6405`) et le HBV seul (`0.6167`) grâce à sa résolution spatiale native supérieure (`0,28 mm/px` contre `1,75 mm/px`).
2. **Limite de la concaténation précoce (`C4 = 0.6708 < C1 = 0.6917`)** : Confirme **Bertelli et al. (2022)** (Section 2.4.3, p. 14) et **Manzoor et al. (2025)** (Section 2.5.1, p. 15 & Section 4.4.1, p. 29) — empiler naïvement les canaux en entrée d'un CNN unique sans recalage physique ni encodeurs dédiés ne permet pas d'exploiter la complémentarité multiparamétrique.

---

## 3. Ce Qu'il Nous Reste à Faire (Étapes 5 à 8)

### Étape 5 : Entraînements de l'Architecture Proposée (`Phase 4` et Configurations `C5`, `C6`, `C7`)

#### 5.1 Commandes d'entraînement prêtes à exécuter
```bash
cd /root/prostate_xai
export DATA_ROOT="/workspace/data"
export RUN_OUTPUT_ROOT="/workspace"

# Phase 4 (Image seule) : 3 Encodeurs SegResNet 3D + Recalage SimpleITK + Attention Croisée + Gating
python3 -m src.training.train_cross_attention --config configs/phase4_cross_attention.yaml

# Config C5 (Tableau 3.1) : SegResNet 3D + Variables Cliniques (MLP) + Concaténation tardive
python3 -m src.training.train_cross_attention --config configs/phase5_c5_concat_clinical.yaml

# Config C6 (Tableau 3.1) : SegResNet 3D + Variables Cliniques (MLP) + Self-Attention
python3 -m src.training.train_cross_attention --config configs/phase5_c6_attention_clinical.yaml

# Config C7 (Tableau 3.1 — Modèle proposé) : SegResNet 3D + Clinique + Attention Croisée Bidirectionnelle + Gating
python3 -m src.training.train_cross_attention --config configs/phase5_c7_cross_attention_clinical.yaml
```

#### 5.2 Baselines Cliniques Seules (Section 3.6.1 du mémoire, p. 25)
- **À développer** : `scripts/train_clinical_baselines.py`
  - **Régression Logistique** (dans l'esprit de l'équipe **Guerbet**, Section 2.2.1) sur `[patient_age, log(PSA), PSAD, prostate_volume]` sur les mêmes 5 folds (`seed=42`).
  - **XGBoost / Gradient Boosting** à complexité limitée sur les 4 variables cliniques.
  - **Analyse descriptive `lesion_PIRADS`** (uniquement comme référentiel clinique de comparaison, jamais comme entrée ni cible du réseau, conformément à la Section 3.2.2).

---

### Étape 6 : Études d'Ablation, Seuils Cliniques Verrouillés & Calibration (Sections 3.6.3, 3.8 & 4.2.1)

1. **Ablations architecturales (Section 3.6.3, p. 25 & Section 4.2.1, p. 29)** :
   - **Model-U** (`--shared_weights` désactivé, poids non partagés) vs **Model-S** (`--shared_weights`, poids partagés entre les 3 encodeurs `SegResNet`).
   - Sans mécanisme de gating (`--use_gating False`).
   - Ablation par retrait d'une modalité (`--modalities t2w adc`, `--modalities t2w hbv`, `--modalities adc hbv`).
   - Option de normalisation différenciée **Hevi AI** (Section 2.2.5 : z-score global sur le pli d'entraînement pour `ADC` vs z-score par examen pour `T2W` et `HBV`).
   - Évaluation de la fenêtre spatiale `19 × 128 × 128` (`--target_shape 128 128 19`, Section 3.3.3 & 4.1).
2. **Métriques cliniques et calibration probabiliste (Section 3.8, p. 26)** :
   - Verrouillage du seuil de décision sur validation (seuil à **sensibilité clinique cible 90 %** et **seuil de Youden**).
   - Calcul systématique de : **Sensibilité, Spécificité, Précision (VPP), VPN, F1-score**.
   - Calcul de l'**ECE (*Expected Calibration Error*)**, du **Score de Brier**, et tracé des **courbes de calibration** (`scripts/evaluate_calibration_and_metrics.py`).

---

### Étape 7 : Explicabilité Intégrée (XAI) & Quantification de l'Incertitude (Sections 4.6 & 4.7)

1. **Grad-CAM++ 3D indépendant par modalité (`src/xai/gradcam3d.py` — Section 4.6.1, p. 31-32)** :
   - Calcul du gradient du logit positif par rapport à `encoder.last_feature_map` pour produire 3 cartes séparées : `Heatmap_T2W`, `Heatmap_ADC`, `Heatmap_HBV` (jamais fusionnées en une seule carte).
   - **Validation quantitative (*Pointing Accuracy*)** : téléchargement des masques lésionnels PI-CAI (`csPCa_lesion_delineations/human_expert` et `AI`) et vérification que le voxel d'activation maximale tombe dans la lésion annotée (rapporté séparément pour les 220 annotations humaines et les annotations IA).
   - **Test de fidélité par occlusion** : masquage de la région maximale et mesure de la chute de probabilité prédite.
2. **SHAP sur la branche clinique (`src/xai/shap_clinical.py` — Section 4.6.2, p. 32)** :
   - Calcul des valeurs de Shapley sur le **logit** (avant sigmoïde/softmax, Éq. 4.6.2) pour quantifier la contribution locale $\phi_j$ de `patient_age`, `log(PSA)`, `PSAD` et `prostate_volume`.
3. **Incertitude par Monte Carlo Dropout & Règle d'Abstention (`src/xai/mc_dropout.py` — Section 4.7, p. 32)** :
   - Activation du dropout à l'inférence avec comparaison de **10, 25 et 50 passes stochastiques**.
   - Calcul de la variance prédictive, du taux d'abstention et du gain de performance sur les cas acceptés vs cas orientés vers une révision humaine.

---

### Étape 8 : Robustesse aux Conditions Africaines à Ressources Limitées & Généralisation Inter-Centres (Sections 3.4, 3.5 & 3.9)

1. **Pipeline de Dégradation Contrôlée $X' = g_\theta(X)$ (`src/data/degradation.py` — Section 3.4, p. 23-24)** :
   - **Bruit Rician** sur `T2W` et `HBV` : niveaux $\alpha \in \{0.03 \text{ (faible)}, 0.06 \text{ (moyen)}, 0.10 \text{ (fort)}\} \times \sigma_\Omega$.
   - **Flou gaussien 3D** sur `ADC` : $\sigma \in \{0.5, 1.0, 1.5\}$ voxel.
   - **Sous-échantillonnage spatial axial** suivi de réinterpolation (simulation d'acquisition 1,5T à résolution réduite).
   - **Modality Dropout** : suppression stochastique ou systématique d'une modalité avec injection explicite de `availability_mask` dans `MultimodalCrossAttentionNet`.
2. **Protocole Expérimental Contrôlé (Section 3.9, p. 26-27)** :
   - **Expérience 1 (Référence propre)** : entraînement et évaluation sur données PI-CAI propres.
   - **Expérience 2 (Test de stress sans adaptation)** : modèle entraîné sur données propres, évalué sur données dégradées ($\Delta_{\text{stress}} = \text{AUC}_{\text{Exp1}} - \text{AUC}_{\text{Exp2}}$).
   - **Expérience 3 (Adaptation au domaine)** : entraînement et évaluation avec dégradeur actif ($\Delta_{\text{adapt}} = \text{AUC}_{\text{Exp3}} - \text{AUC}_{\text{Exp2}}$ et $\Delta_{\text{recovery}}$).
   - **Expérience 4 (Généralisation inter-centres — *Leave-One-Center-Out*, Section 3.5 & 3.9)** : entraînement sur 2 centres hollandais (ex. `RUMC + PCNN`) et test sur le 3e centre jamais vu (`ZGT`), avec permutation des rôles (utilisant la colonne `center` de `marksheet.csv`).

---

## 4. Harmonisation Rédactionnelle à Reporter dans le Manuscrit LaTeX

- **Remplacer l'ancienne mention « 598 cas complets / PROSTATEx » par « 1 180 cas complets de PI-CAI »** (PROSTATEx étant totalement écarté) aux passages suivants du manuscrit :
  - **Résumé (p. iv)**
  - **Section 1.3 — Lieu de l'étude et population (p. 3)**
  - **Section 1.8 — Méthodologie technique / Données (p. 5)**
  - **Section 2.2.2 — BDAV_Y (p. 8)**
  - **Section 3.2.1 — Cohorte et dédoublonnage (p. 20-21)**
  - **Section 3.5 — Protocole de séparation des données (p. 24)**
  - **Section 3.10 — Gestion du nombre réduit de cas (p. 27)**
  - **Section 4.2.1 — Poids partagés ou non partagés (p. 29)**
  - **Section 4.4.4 — Bloc de fusion (p. 30)**
- **Compléter au Chapitre 5 les 3 familles de tableaux prévues à la Section 3.11 (p. 27)** au fur et à mesure de la fin des étapes 5 à 8 ci-dessus :
  1. *Tableau 5.1* : Performance diagnostique des configurations `C1` à `C7` + baselines cliniques (`AUROC, Sensibilité, Spécificité, F1, ECE`).
  2. *Tableau 5.2* : Robustesse sur les Expériences 1 à 4 (`AUROC propre, perte sous stress, récupération après adaptation, Leave-One-Center-Out`).
  3. *Tableau 5.3* : Fidélité et stabilité de l'explicabilité (`Pointing Accuracy` humain vs IA, chute sous occlusion, stabilité des cartes sous bruit).
