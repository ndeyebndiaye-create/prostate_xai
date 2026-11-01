# Journal de Bord et Registre des Expérimentations (Thèse Prostate XAI)

> **Document Vivant de Suivi Scientifique**  
> Ce registre consigne rigoureusement l'historique complet de chaque phase du projet : hypothèses de recherche, protocoles d'exécution, commandes exactes, résultats quantitatifs obtenus, visualisations générées et décisions pour les étapes suivantes.  
> **Règle du projet** : Toute future expérimentation (Phases 4, 5, 6, 7) doit être enregistrée ici dès son exécution.

---

## Sommaire Chronologique
- [Fiche 01 — Phase 0 : Infrastructure, Streaming RAM et Tests Unitaires](#fiche-01--phase-0--infrastructure-streaming-ram-et-tests-unitaires)
- [Fiche 02 — Phase 2 : Analyse Exploratoire des Données (EDA) et Inspection Visuelle mpMRI](#fiche-02--phase-2--analyse-exploratoire-des-données-eda-et-inspection-visuelle-mpmri)
- [Fiche 03 — Phase 1 : Baseline Unimodale T2W (SimpleCNN3D, 5-Fold CV)](#fiche-03--phase-1--baseline-unimodale-t2w-simplecnn3d-5-fold-cv)
- [Fiche 04 — Phase 3 (À Venir) : Baseline Multimodale par Concaténation Précoce](#fiche-04--phase-3-à-venir--baseline-multimodale-par-concaténation-précoce)
- [Fiche 05 — Phase 4 (À Venir) : Encodeurs 3D Dédiés (SegResNet / MONAI)](#fiche-05--phase-4-à-venir--encodeurs-3d-dédiés-segresnet--monai)
- [Fiche 06 — Phase 5 (À Venir) : Fusion par Cross-Attention et Intégration Clinique](#fiche-06--phase-5-à-venir--fusion-par-cross-attention-et-intégration-clinique)
- [Fiche 07 — Phase 6 (À Venir) : Explicabilité (Grad-CAM++ 3D, SHAP) et Validation Radiologique](#fiche-07--phase-6-à-venir--explicabilité-grad-cam-3d-shap-et-validation-radiologique)

---

## Fiche 01 — Phase 0 : Infrastructure, Streaming RAM et Tests Unitaires

* **Date** : Septembre 2026
* **Objectif** : Mettre en place un socle logiciel robuste, modulaire et vérifiable sans nécessiter de stocker 60 Go de données sur chaque machine de test.
* **Méthodologie & Architecture** :
  * Création de `src/data/preprocessing.py` avec *Masked Z-Score* et *Center-Crop/Zero-Pad* $(96, 96, 16)$.
  * Développement du streaming RAM via `rclone cat` pour lire les fichiers `.mha` directement depuis Google Drive.
  * Création des garde-fous méthodologiques :
    * `fold_guard.py` : vérification stricte de non-chevauchement des patients ($\text{Train} \cap \text{Val} = \emptyset$).
    * `seed.py` : fixation déterministe de toutes les graines aléatoires (seed=42).
* **Validation** :
  * Commande : `pytest tests/ -v`
  * Résultat : **42/42 tests passés avec succès** (couverture complète du preprocessing, des datasets et de l'infrastructure).

---

## Fiche 02 — Phase 2 : Analyse Exploratoire des Données (EDA) et Inspection Visuelle mpMRI

* **Date** : Octobre 2026
* **Objectif** : Quantifier les distributions statistiques du dataset PI-CAI (1 500 patients) et vérifier la qualité des volumes IRM axiaux avec leurs annotations de vérités terrain.
* **Commande d'Exécution** :
  ```bash
  python scripts/visualize_eda.py
  python scripts/visualize_mri_slices.py
  ```
* **Résultats Quantitatifs (EDA)** :
  * **Nombre total de cas** : 1 500 patients
  * **Seuil csPCa retenu** : ISUP $\ge$ 2 (Gleason 3+4+)
  * **Classe 0 (Non-csPCa)** : 1 075 patients (**71.67%**)
  * **Classe 1 (csPCa)** : 425 patients (**28.33%**)
  * **Ratio de déséquilibre** : **2.53 : 1** $\rightarrow$ Justifie la pondération de la loss (`pos_weight = 2.5`).
* **Visualisations Clés Générées** :
  * `docs/figures/eda_class_distribution.png` : diagramme en barres et camembert du ratio de classes.
  * `docs/figures/eda_clinical_covariates.png` : distribution de l'âge, du PSA sérique, des scores PI-RADS (4 et 5 majoritaires pour csPCa) et répartition multi-centrique.
  * `docs/figures/eda_correlation_matrix.png` : matrice de corrélation de Pearson entre variables cliniques.
  * `docs/figures/mri_multimodal_patient10008.png` : comparaison synchronisée T2W / ADC / HBV / Masque expert (foyer d'hypo-signal ADC et d'hyper-signal HBV concordant avec l'annotation).
  * `docs/figures/mri_lesion_slices_progression.png` : extension 3D axiale ($z=13, 14, 15$).
  * `docs/figures/mri_preprocessed_input_tensor.png` : tenseur 3D $(96, 96, 16)$ normalisé par masked z-score.
  * `docs/figures/presentation_preprocessing_step_by_step.png` : découpage pas à pas du prétraitement (Brut -> Masked Z-score -> Clipping -> Center-Crop + Histogrammes).
  * `docs/figures/presentation_before_after_direct.png` : comparatif percutant AVANT vs APRÈS pour diapositives de présentation.
  * `docs/figures/presentation_multimodal_harmonization.png` : harmonisation spatiale et d'échelle des 3 modalités (T2W, ADC, HBV) pour la fusion.

---

## Fiche 03 — Phase 1 : Baseline Unimodale T2W (SimpleCNN3D, 5-Fold CV)

* **Date** : Septembre - Octobre 2026
* **Objectif** : Établir la performance de référence de la séquence anatomique principale (T2W) seule, servant de point de comparaison (*baseline*) pour quantifier le gain de la fusion multimodale.
* **Architecture & Hyperparamètres** :
  * **Modèle** : `SimpleCNN3D` (4 blocs convolutifs 3D + BatchNorm3D + MaxPool3D + tête Dense).
  * **Validation** : Stratified 5-Fold Cross-Validation au niveau patient.
  * **Entrée** : Tenseur 3D mono-canal $(1, 16, 96, 96)$.
  * **Epochs** : 30 epochs par fold (150 fold-epochs au total).
  * **Optimiseur** : AdamW (learning rate = $10^{-4}$, weight decay = $10^{-5}$).
  * **Fonction de coût** : `nn.CrossEntropyLoss(weight=[1.0, 2.5])`.
  * **Précision** : Mixte (AMP `torch.cuda.amp.GradScaler`).
* **Commande d'Exécution** :
  ```bash
  python -m src.training.train_unimodal_baseline --config configs/phase1_t2w.yaml
  ```
* **Résultats Obtenus (Traçabilité `results/baselines.csv`)** :

  | Fold | Validation AUROC | Meilleur Checkpoint |
  | :---: | :---: | :---: |
  | **Fold 1** | `0.5342` | `checkpoints/phase1_t2w_only_fold1_best.pt` |
  | **Fold 2** | `0.7143` | `checkpoints/phase1_t2w_only_fold2_best.pt` |
  | **Fold 3** | `0.7516` | `checkpoints/phase1_t2w_only_fold3_best.pt` |
  | **Fold 4** | `0.6944` | `checkpoints/phase1_t2w_only_fold4_best.pt` |
  | **Fold 5** | `0.8913` | `checkpoints/phase1_t2w_only_fold5_best.pt` |
  | **SYNTHÈSE** | **`0.7172 ± 0.1145`** | **Fold 5 optimal (`0.8913`)** |

* **Visualisation Générée** :
  * Commande : `python scripts/visualize_training_results.py`
  * Fichier : `docs/figures/training_baseline_results.png`
* **Interprétation Médicale et Décisions** :
  * L'AUROC moyenne de `0.717` démontre que le modèle capte du signal tumoral morphologique (nettement au-dessus de 0.50).
  * La forte variabilité (0.534 sur Fold 1 vs 0.891 sur Fold 5) s'explique par le fait que le T2W seul est souvent trompé par des prostatites ou de l'hyperplasie bénigne.
  * **Décision** : Passage à la Phase 3 (fusion T2W + ADC + HBV) pour éliminer les faux positifs grâce à la diffusion.

---

## Modèle de Fiche pour les Futures Expérimentations

Pour chaque nouvelle expérience à venir, dupliquez ce template :

```markdown
## Fiche XX — Phase X : [Nom de l'Expérience]

* **Date** : [YYYY-MM-DD]
* **Commit Git** : [Hash Git court]
* **Hypothèse Scientifique** : [Ce qu'on cherche à tester ou améliorer]
* **Modifications Apportées** : [Code modifié, nouveau modèle, configuration YAML]
* **Commande d'Exécution** :
  ```bash
  python -m ... --config configs/...
  ```
* **Résultats Quantitatifs** :
  * Table des métriques (AUROC par fold, Moyenne ± Écart-type)
  * Comparaison avec les baselines précédentes (Gain $\Delta\text{AUROC}$)
* **Visualisations Générées** :
  * Fichiers PNG dans `docs/figures/`
* **Conclusion & Décisions pour la suite** : [Ce qu'on retient pour la rédaction de la thèse]
```
