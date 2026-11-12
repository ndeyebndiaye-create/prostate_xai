# Documentation Détaillée des Étapes 4 à 8 : Architecture, Code, Commandes et Analyse des Résultats

> **Projet de Mémoire AIMS Sénégal**  
> *Explainable Multi-Modal Deep Learning Framework for Prostate Cancer Aggressiveness Classification from Multiparametric MRI: Toward Clinically Trustworthy AI for Low-Resource African Settings*  
> **Auteure** : Ndeye Binta NDIAYE  **Superviseur** : Dr Aurelle Tchagna  
> **Dossier de figures dédié (Phases 4 à 8)** : `docs/xai_and_robustness_visuals/` *(séparé de `docs/figures/` qui conserve les figures d'EDA et de prétraitement initial).*

---

## Table des Matières
1. [Architecture Globale du Code Créé (Étapes 4 à 8)](#1-architecture-globale-du-code-créé-étapes-4-à-8)
2. [Étape 4 — Prétraitement Physique SimpleITK & Encodeurs 3D SegResNet + Cross-Attention (Image Seule)](#2-étape-4--prétraitement-physique-simpleitk--encodeurs-3d-segresnet--cross-attention-image-seule)
3. [Étape 5.1 — Baselines Cliniques Seules (Régression Logistique Guerbet, Gradient Boosting & Référentiel PI-RADS)](#3-étape-51--baselines-cliniques-seules-régression-logistique-guerbet-gradient-boosting--référentiel-pi-rads)
4. [Étape 5.2 — Configurations Multimodales + Cliniques (`C5`, `C6`, `C7`)](#4-étape-52--configurations-multimodales--cliniques-c5-c6-c7)
5. [Étape 6 — Métriques Cliniques Verrouillées (@ Sensibilité 90 % & Youden) et Calibration (`ECE`, `Brier`)](#5-étape-6--métriques-cliniques-verrouillées--sensibilité-90---youden-et-calibration-ece-brier)
6. [Étape 7 — Explicabilité Intégrée (`Grad-CAM++ 3D` par Modalité, `SHAP` Exact sur Logit) & Incertitude (`MC Dropout`)](#6-étape-7--explicabilité-intégrée-grad-cam-3d-par-modalité-shap-exact-sur-logit--incertitude-mc-dropout)
7. [Étape 8 — Robustesse aux Dégradations Africaines (`1.5T`, Bruit Rician, Flou ADC, Séquence Manquante)](#7-étape-8--robustesse-aux-dégradations-africaines-15t-bruit-rician-flou-adc-séquence-manquante)

---

## 1. Architecture Globale du Code Créé (Étapes 4 à 8)

Le tableau ci-dessous récapitule chaque fichier créé ou enrichi dans le dépôt, son rôle scientifique dans le mémoire, et la section correspondante du manuscrit :

| Fichier du Projet | Rôle Technique & Scientifique | Section du Mémoire |
| :--- | :--- | :---: |
| `src/data/preprocessing.py` | Recalage spatial physique `resample_to_reference()` (`SimpleITK.Resample` d'`ADC`/`HBV` vers la grille `T2W`) + normalisation *Masked Z-score* + cache disque `.npy` et RAM (`lru_cache`) | Chap. 3, Sec. 3.3.1 |
| `src/data/multimodal_dataset.py` | Nettoyage et standardisation sans fuite (`fit_transform_clinical_features`) des 4 variables cliniques `[patient_age, log_psa, psad, prostate_volume]` (imputation médiane et z-score calculés strictement sur `train_ids`) | Chap. 3, Sec. 3.2.2 & 3.5 |
| `src/models/segresnet_encoder.py` | `SegResNetEncoder3D` (MONAI `SegResNet` 3D sans tête de segmentation, conserve `last_feature_map` avec `retain_grad()` pour Grad-CAM++ 3D) + `ClinicalEncoderMLP` (`4 -> 64 -> 64`, `LayerNorm + GELU + Dropout`) | Chap. 4, Sec. 4.2 & 4.3 |
| `src/models/cross_attention_fusion.py` | `MultimodalCrossAttentionNet` : supporte **Model-U** (`shared_weights=False`) et **Model-S** (`shared_weights=True`), `BidirectionalCrossAttention`, `LightweightTransformerBlock`, `ModalityGating` et `availability_mask` | Chap. 4, Sec. 4.4 & 4.5 |
| `src/training/train_cross_attention.py` | Script d'entraînement 5-Fold CV avec AMP FP16, reprise sur checkpoint atomique (`last.pt`/`best.pt`), logging epoch par epoch (`logs/*.csv`) et Modality Dropout optionnel | Chap. 3, Sec. 3.5 & 3.7 |
| `src/utils/calibration_metrics.py` | Verrouillage du seuil clinique à **90 % de sensibilité** (`find_threshold_at_sensitivity`) et **indice de Youden** (`find_youden_threshold`), calcul de Sensibilité, Spécificité, VPP, VPN, F1, **ECE** et **Score de Brier** | Chap. 3, Sec. 3.8 |
| `scripts/train_clinical_baselines.py` | Entraînement 5-Fold CV de la Régression Logistique (`clinical_logreg`) et du Gradient Boosting (`clinical_gboost`) sur les 4 variables cliniques + évaluation descriptive externe de `lesion_PIRADS` | Chap. 3, Sec. 3.6.1 |
| `scripts/evaluate_calibration_and_metrics.py` | Calcul automatique de toutes les métriques cliniques et de calibration sur les checkpoints `best.pt` sauvegardés | Chap. 3, Sec. 3.8 & Tab. 5.1 |
| `src/xai/gradcam3d.py` | Calcul de **Grad-CAM++ 3D indépendant par modalité** (`Heatmap_T2W`, `Heatmap_ADC`, `Heatmap_HBV`), **Pointing Accuracy** et **test de fidélité par occlusion** ($\Delta_{\text{prob}}$) | Chap. 4, Sec. 4.6.1 |
| `src/xai/shap_clinical.py` | Calcul **exact** des $2^4 = 16$ coalitions de Shapley (`SHAP`) sur le **logit positif** csPCa pour quantifier $\phi_{\text{âge}}, \phi_{\log(\text{PSA})}, \phi_{\text{PSAD}}, \phi_{\text{volume}}$ | Chap. 4, Sec. 4.6.2 (Éq. 4.6.2) |
| `src/xai/mc_dropout.py` | Inférence stochastique **Monte Carlo Dropout** ($T \in \{10, 25, 50\}$ passes) + évaluation de la **règle d'abstention sélective** | Chap. 4, Sec. 4.7 |
| `scripts/run_xai_and_uncertainty.py` | Pipeline exécutable complet pour l'Étape 7 (génère les visuels dans `docs/xai_and_robustness_visuals/` et le tableau `results/xai_and_uncertainty_summary.csv`) | Chap. 5, Tab. 5.3 |
| `src/data/degradation.py` | Opérateur de dégradation contrôlée $X' = g_\theta(X)$ : **Bruit Rician 3D** ($\alpha \in \{0.03, 0.06, 0.10\}$ sur `T2W`/`HBV`), **Flou gaussien 3D** ($\sigma \in \{0.5, 1.0, 1.5\}$ sur `ADC`) et **Sous-échantillonnage axial 1,5T** | Chap. 3, Sec. 3.4 |
| `scripts/run_robustness_and_loco.py` | Évaluation des tests de stress (Expérience 1 vs Expérience 2 + séquences manquantes) pour remplir le Tableau 5.2 | Chap. 3, Sec. 3.9 & Tab. 5.2 |

---

## 2. Étape 4 — Prétraitement Physique SimpleITK & Encodeurs 3D SegResNet + Cross-Attention (Image Seule)

### 2.1 Commande Exécutée
```bash
export TRITON_DEFAULT_FP_FUSION=1
python3 -m src.training.train_cross_attention --config configs/phase4_cross_attention.yaml
```

### 2.2 Ce que fait le code en détail
1. **Alignement physique inter-séquence (`resample_to_reference` dans `src/data/preprocessing.py`)** :
   - Les séquences `ADC` et `HBV` ont une résolution native grossière ($\approx 1,75 \text{ mm/px}$) comparée au `T2W` ($\approx 0,28 \text{ mm/px}$).
   - Avant tout découpage matriciel, `SimpleITK.Resample` projette `ADC` et `HBV` dans le repère physique exact (origine millimétrique, espacement voxel, cosinus directeurs) du volume `T2W` du patient.
   - Le tenseur $(96, 96, 16)$ normalisé par *masked z-score* est mis en cache dans un fichier `.npy` local afin de n'exécuter l'interpolation C++ qu'une seule fois par patient.
2. **Trois encodeurs 3D `SegResNet` dédiés (Model-U)** :
   - Chaque séquence ($m \in \{\text{T2W}, \text{ADC}, \text{HBV}\}$) traverse son propre encodeur résiduel 3D `SegResNetEncoder3D` (`init_filters=16, blocks_down=(1, 2, 2, 4)`), produisant une carte de caractéristiques 3D profonde $A^{(m)} \in \mathbb{R}^{128 \times 2 \times 12 \times 12}$ suivie d'un *Global Average Pooling* et d'une projection vers $\mathbb{R}^{128}$.
3. **Attention croisée T2W $\leftrightarrow$ Diffusion + Transformer + Gating** :
   - Sans variables cliniques (`use_clinical=False`), le token anatomique `T2W` sert d'ancre spatiale face aux tokens fonctionnels `(ADC, HBV)` dans `BidirectionalCrossAttention`, suivi de 2 blocs Transformer légers et du `ModalityGating`.

### 2.3 Résultats Obtenus (`results/baselines.csv` & `logs/phase4_segresnet_cross_attention_fold1..5.csv`)

| Expérience | Modalités | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **AUROC (Moy ± Écart-type)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`phase4_segresnet_cross_attention`** | `t2w+adc+hbv` (Image seule) | `0.6685` | `0.6456` | `0.6194` | `0.7183` | `0.6319` | **`0.6567 ± 0.0348`** |

**Interprétation scientifique** :
- Avec 3 encodeurs `SegResNet` 3D distincts + 2 couches Transformer + Cross-Attention entraînés *from scratch* sur 944 patients par pli sans l'ancrage régularisateur des biomarqueurs cliniques (`PSA`, `PSAD`, `volume`), le modèle image seule plus profond présente un léger sur-apprentissage sur les petits plis par rapport au `SimpleCNN3D` T2W seul (`0.6917`), tout en atteignant `0.7183` sur le Fold 4.
- Cela motive directement l'**Étape 5** : l'introduction des variables cliniques (`[âge, log(PSA), PSAD, volume]`) pour guider et régulariser l'attention des encodeurs 3D.

---

## 3. Étape 5.1 — Baselines Cliniques Seules (Régression Logistique Guerbet, Gradient Boosting & Référentiel PI-RADS)

### 3.1 Commande Exécutée
```bash
python3 -m scripts.train_clinical_baselines
```

### 3.2 Nettoyage et Prétraitement des Variables Cliniques (`fit_transform_clinical_features`)
Sur chacun des 5 plis (`seed=42`, exactement les mêmes 1 180 patients complets que `C1`–`C4`) :
1. **Filtrage des valeurs aberrantes** : toute valeur non numérique ou physiologiquement impossible ($\text{PSA} \le 0$, $\text{volume} \le 0$, $\text{PSAD} \le 0$, $\text{âge} \notin [18, 110]$) est convertie en `NaN`.
2. **Reconstruction du PSAD manquant** : $\text{PSAD} = \text{PSA} / \text{volume}$ lorsque `psad` est absent mais que `psa` et `prostate_volume` sont valides.
3. **Transformation logarithmique** : $\log(\text{PSA} + 10^{-3})$ pour stabiliser la queue de distribution droite du PSA sérique.
4. **Imputation par la médiane et Z-score sans fuite** : la médiane d'imputation, la moyenne $\mu_{\text{train}}$ et l'écart-type $\sigma_{\text{train}}$ sont ajustés **uniquement sur les 944 patients d'entraînement** du pli, puis appliqués aux 236 patients de validation.

### 3.3 Résultats Quantitatifs (`results/baselines.csv` & `results/clinical_and_calibration_metrics.csv`)

#### A. Performance AUROC par pli (sur les 1 180 patients complets)
| Modèle Clinique | Variables | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **AUROC (Moy ± Écart-type)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clinical_logreg`** *(Régression Logistique)* | `âge, log(PSA), PSAD, vol` | `0.7568` | `0.8052` | `0.6776` | `0.8130` | `0.7573` | **`0.7620 ± 0.0483`** |
| **`clinical_gboost`** *(Gradient Boosting)* | `âge, log(PSA), PSAD, vol` | `0.7159` | `0.7869` | `0.6659` | `0.7829` | `0.7678` | **`0.7439 ± 0.0465`** |

#### B. Métriques Cliniques à Seuil Verrouillé & Calibration Probabiliste (Moyenne 5 Folds)
| Modèle | Stratégie de Seuil | Seuil Verrouillé | **Sensibilité** | **Spécificité** | **VPP (Précision)** | **VPN** | **F1-score** | **Score de Brier** | **ECE** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clinical_logreg`** | **Sensibilité cible 90 % (`sens90`)** | `0.3190` | **`0.9075`** | `0.3751` | `0.3653` | **`0.9149`** | `0.5207` | `0.1996` | `0.1789` |
| **`clinical_logreg`** | **Indice de Youden (`youden`)** | `0.5060` | `0.6776` | `0.6923` | `0.4683` | `0.8449` | `0.5520` | `0.1996` | `0.1789` |
| **`clinical_gboost`** | **Sensibilité cible 90 % (`sens90`)** | `0.1884` | `0.7970` | `0.5538` | `0.4154` | `0.8736` | `0.5457` | **`0.1751`** | **`0.0574`** |
| **`clinical_gboost`** | **Indice de Youden (`youden`)** | `0.3076` | `0.6358` | **`0.7444`** | **`0.4929`** | `0.8396` | **`0.5540`** | **`0.1751`** | **`0.0574`** |

### 3.4 Analyse Scientifique Détaillée (pour le Chapitre 5 du Mémoire)
1. **Validation empirique du constat de l'équipe Guerbet (Chapitre 2, Section 2.2.1)** :
   - Sur 4 variables tabulaires continues à relation monotone avec le risque tumoral (principalement la densité de PSA `PSAD`), la **Régression Logistique linéaire (`0.7620`)** surpasse un modèle d'arbres boostés (`0.7439`) qui a tendance à sur-découper l'espace à 4 dimensions.
2. **Le dilemme clinique du PSA/PSAD seul (Sensibilité vs Spécificité)** :
   - Lorsqu'on verrouille le seuil de décision pour garantir la **sensibilité clinique de 90 %** exigée pour ne pas manquer de cancer agressif (`Sensibilité = 90,75 %`, `VPN = 91,49 %`), la **spécificité de la Régression Logistique chute à `37,51 %`**.
   - Concrètement, **62,5 % des patients sains (hyperplasie bénigne de la prostate ou prostatite élevant le PSA) subiraient une biopsie inutile** si l'on se fiait uniquement aux variables cliniques.
   - C'est la justification clinique fondamentale de la fusion **IRM multiparamétrique + Clinique (`C5`, `C6`, `C7`)** : combiner la haute valeur prédictive négative de la clinique avec la localisation tissulaire de l'IRM 3D pour réduire les faux positifs.

---

## 4. Étape 5.2 — Configurations Multimodales + Cliniques (`C5`, `C6`, `C7`)

### 4.1 Commandes d'Exécution
```bash
export TRITON_DEFAULT_FP_FUSION=1

# Configuration C5 (Tableau 3.1) : SegResNet 3D + MLP Clinique + Concaténation tardive
python3 -m src.training.train_cross_attention --config configs/phase5_c5_concat_clinical.yaml

# Configuration C6 (Tableau 3.1) : SegResNet 3D + MLP Clinique + Self-Attention Transformer
python3 -m src.training.train_cross_attention --config configs/phase5_c6_attention_clinical.yaml

# Configuration C7 (Tableau 3.1 — Architecture complète proposée) :
# SegResNet 3D + MLP Clinique + Attention Croisée Bidirectionnelle Image <-> Clinique + Modality Gating
python3 -m src.training.train_cross_attention --config configs/phase5_c7_cross_attention_clinical.yaml
```

### 4.2 Différences Architecturales Exactes entre `C5`, `C6` et `C7`
- **`C5` (`fusion_mode="concat"`)** : Les 3 vecteurs latents IRM ($\mathbf{t}_{\text{T2W}}, \mathbf{t}_{\text{ADC}}, \mathbf{t}_{\text{HBV}} \in \mathbb{R}^{128}$) et le vecteur clinique ($\mathbf{t}_{\text{clin}} \in \mathbb{R}^{128}$) sont simplement concaténés en un vecteur $\mathbb{R}^{512}$ projeté vers $\mathbb{R}^{128}$ sans mécanisme d'attention inter-modalité.
- **`C6` (`fusion_mode="attention"`)** : Les 4 tokens ($\text{T2W}, \text{ADC}, \text{HBV}, \text{clin}$) + le token `[CLS]` passent dans les 2 couches d'auto-attention (`LightweightTransformerBlock`) + `ModalityGating`, mais **sans l'étape préalable d'attention croisée bidirectionnelle** entre la clinique et l'image.
- **`C7` (`fusion_mode="cross_attention"`, modèle proposé au Chapitre 4)** :
  1. **Attention croisée bidirectionnelle (`BidirectionalCrossAttention`, Section 4.4.3)** : Le contexte clinique interroge les 3 séquences IRM ($\text{Clin} \to \text{IRM}$) et les 3 séquences IRM sont modulées par le profil clinique ($\text{IRM} \to \text{Clin}$).
  2. **Raffinement global (`LightweightTransformerBlock`, Section 4.4.4)** : Auto-attention à 4 têtes avec token `[CLS]` et encodages de type de modalité.
  3. **Pondération adaptative (`ModalityGating`, Section 4.4.5)** : Calcul des poids de confiance $\alpha_m \in [0, 1]$ ($\sum_m \alpha_m = 1$) pour chaque modalité avant la tête de classification finale.

### 4.3 Résultats Acquis — Configurations `C5`, `C6` et `C7 Baseline`

| Configuration | Modalités | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **AUROC (Moy ± Écart-type)** | Gain vs `C1` (`T2W`) | Gain vs `clinical_logreg` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`phase5_c5_concat_clinical` (`C5`)** | `t2w+adc+hbv+clin` | `0.7648` | `0.8098` | `0.6939` | `0.8253` | `0.7751` | **`0.7738 ± 0.0456`** | **`+8.21 pts`** (`0.6917`) | **`+1.18 pts`** (`0.7620`) |
| **`phase5_c6_attention_clinical` (`C6`)** | `t2w+adc+hbv+clin` | `0.7669` | `0.8140` | `0.6856` | `0.8327` | `0.7895` | **`0.7778 ± 0.0512`** | **`+8.61 pts`** (`0.6917`) | **`+1.58 pts`** (`0.7620`) |
| **`phase5_c7_cross_attention_clinical` (`C7 Baseline`)** | `t2w+adc+hbv+clin` | `0.7729` | `0.8060` | `0.6869` | `0.8330` | `0.7751` | **`0.7748 ± 0.0492`** | **`+8.31 pts`** (`0.6917`) | **`+1.28 pts`** (`0.7620`) |

### 4.4 Raffinement C7 Haute Résolution (`phase5_c7_cross_attention_clinical_hires` : $128 \times 128 \times 20$)
À la suite du diagnostic géométrique démontrant que le crop initial à $96 \times 96$ ($4{,}8 \times 4{,}8\text{ cm}$) tronquait la partie antérieure de la prostate et incluait le rectum postérieur, une configuration haute résolution avec recentrage antérieur a été mise en place :
- **Dimension spatiale cible** : $128 \times 128 \times 20$ voxels ($6{,}4 \times 6{,}4\text{ cm}$ axial et $6{,}0\text{ cm}$ craniocaudal).
- **Recentrage antérieur paramétrable** dans `center_crop_or_pad()` (`src/data/preprocessing.py`).
- **Orchestration complète en une commande** via `scripts/run_hires_pipeline_and_xai.sh`.

#### Mise à jour synchronisée des figures dans `docs/figures/` et `docs/xai_and_robustness_visuals/` :
L'exécution de `bash scripts/run_hires_pipeline_and_xai.sh` déclenche automatiquement la mise à jour des figures suivantes :
1. `docs/figures/training_all_models_comparison.png` : Intègre la nouvelle barre C7 Hi-Res en cyan foncé aux côtés de C1–C7.
2. `docs/figures/training_clinical_and_calibration_tradeoff.png` : Met à jour les compromis Sensibilité@90%, Spécificité, ECE et Brier.
3. `docs/figures/training_epoch_learning_curves.png` : Ajoute la courbe de convergence epoch par epoch de C7 Hi-Res.
4. `docs/figures/xai_gradcam3d_per_modality.png` : Affiche la prostate complète délimitée, sans voile diffus, avec masquage anatomique transparent ($<0.20$) et isolignes de contours (50 % jaune, 75 % rouge).
5. `docs/figures/xai_clinical_shap_summary.png` : Importance relative exacte des 4 variables cliniques.
6. `docs/figures/xai_mc_dropout_abstention.png` : Courbe de gain d'AUROC par abstention sélective.

---

## 5. Étape 6 — Métriques Cliniques Verrouillées (@ Sensibilité 90 % & Youden) et Calibration (`ECE`, `Brier`)

### 5.1 Commande d'Exécution (après la fin des entraînements)
```bash
python3 -m scripts.evaluate_calibration_and_metrics
```

### 5.2 Ce que calcule ce script
Il parcourt automatiquement tous les dossiers `checkpoints/<experience>/fold_<1..5>/best.pt`, extrait les probabilités de validation `best_probs` et les vrais labels `best_labels`, et calcule pour chaque pli et en moyenne (`summary_mean`) :
- **`auroc`** : Aire sous la courbe ROC.
- **`threshold`** : Seuil de décision verrouillé selon deux stratégies cliniques (`sens90` pour garantir $\ge 90\%$ de sensibilité, et `youden` maximisant $\text{Sensibilité} + \text{Spécificité} - 1$).
- **`sensitivity`, `specificity`, `ppv`, `npv`, `f1`** : Métriques diagnostiques complètes.
- **`ece` (*Expected Calibration Error*) & `brier` (*Score de Brier*)** : Évaluation de la fiabilité des probabilités prédites (Tableau 5.1 du mémoire).
- **Sortie** : `results/clinical_and_calibration_metrics.csv`.

---

## 6. Étape 7 — Explicabilité Intégrée (`Grad-CAM++ 3D` par Modalité, `SHAP` Exact sur Logit) & Incertitude (`MC Dropout`)

### 6.1 Commande d'Exécution
```bash
python3 -m scripts.run_xai_and_uncertainty --experiment phase5_c7_cross_attention_clinical --use_clinical
```

### 6.2 Détail Mathématique et Technique du Code (`src/xai/`)
1. **Grad-CAM++ 3D séparé par modalité (`src/xai/gradcam3d.py`)** :
   - Contrairement aux approches à concaténation précoce (`C4`) qui mélangent toutes les séquences dès la première couche et ne peuvent produire qu'une seule carte floue, notre architecture à encodeurs dédiés permet de calculer $\frac{\partial z_{\text{csPCa}}}{\partial A^{(m)}_{k,d,h,w}}$ séparément sur la dernière carte convolutive 3D de **chaque** encodeur (`T2W`, `ADC`, `HBV`).
   - Produit 3 cartes 3D distinctes : `Heatmap_T2W`, `Heatmap_ADC`, `Heatmap_HBV`.
   - Calcule le **test de fidélité par occlusion** ($\Delta_{\text{prob}}$ lorsque les 5 % de voxels les plus saillants sont masqués) et la **Pointing Accuracy**.
2. **SHAP Exact sur le Logit Clinique (`src/xai/shap_clinical.py`)** :
   - Évalue exactement les $2^4 = 16$ coalitions des 4 variables cliniques sur le **logit** de la classe positive (avant softmax, garantissant l'additivité exacte $\sum_{j=1}^4 \phi_j = f(x_{\text{img}}, x_{\text{clin}}) - f(x_{\text{img}}, \mathbf{0})$) en mettant en cache l'encodage 3D de l'IRM pour que le calcul soit instantané (< 10 ms par patient).
3. **Monte Carlo Dropout & Règle d'Abstention (`src/xai/mc_dropout.py`)** :
   - Réactive les couches `Dropout` à l'inférence (`enable_mc_dropout`) et compare $T \in \{10, 25, 50\}$ passes stochastiques.
   - Trace la courbe d'AUROC en fonction du taux d'abstention sélective $q \in \{0\%, 5\%, 10\%, 15\%, 20\%, 25\%\}$ (orientation des cas à forte variance prédictive $\sigma^2$ vers le radiologue).
4. **Dossier de sortie des visuels** :
   - Toutes les figures de cette étape sont enregistrées dans **`docs/xai_and_robustness_visuals/`** (et le tableau quantitatif dans `results/xai_and_uncertainty_summary.csv`).

---

## 7. Étape 8 — Robustesse aux Dégradations Africaines (`1.5T`, Bruit Rician, Flou ADC, Séquence Manquante)

### 7.1 Commande d'Exécution
```bash
python3 -m scripts.run_robustness_and_loco --experiment phase5_c7_cross_attention_clinical --use_clinical
```

### 7.2 Détail des Dégradations Simulées (`src/data/degradation.py`)
Pour simuler les conditions d'acquisition d'hôpitaux d'Afrique subsaharienne équipés d'IRM 1,5T plus anciennes ou à protocole incomplet :
1. **Bruit Rician 3D (`apply_rician_noise_3d`)** sur `T2W` et `HBV` :
   $$I_{\text{bruité}} = \sqrt{(I + n_1)^2 + n_2^2}, \quad n_1, n_2 \sim \mathcal{N}(0, (\alpha \sigma_\Omega)^2)$$
   avec 3 niveaux de sévérité : faible ($\alpha = 0,03$), moyen ($\alpha = 0,06$) et fort ($\alpha = 0,10$).
2. **Flou Gaussien 3D séparable (`apply_gaussian_blur_3d`)** sur `ADC` : $\sigma \in \{0.5, 1.0, 1.5\}$ voxel.
3. **Sous-échantillonnage axial 1,5T (`apply_axial_downsampling_3d`)** : réduction de la résolution spatiale d'un facteur 2 à 3 suivie d'une ré-interpolation trilinéaire.
4. **Séquence IRM manquante (*Modality Dropout* via `availability_mask`)** : évaluation du modèle lorsque `T2W`, `ADC` ou `HBV` est totalement absent ($\text{mask}_m = 0$), le module `ModalityGating` redistribuant automatiquement la masse d'attention ($\text{softmax}$) sur les séquences restantes et la clinique.
5. **Sortie** : `results/robustness_and_loco_table5_2.csv` (Tableau 5.2 du mémoire).
