# Rapport d'Avancement Technique — Implémentation du Pipeline et des Baselines

**Projet** : Détection du Cancer Cliniquement Significatif de la Prostate (csPCa) par Imagerie Multimodale & IA Explicable (XAI)  
**Jeu de données** : PI-CAI (IRM biparamétrique/multiparamétrique : T2W, ADC, HBV)  
**Statut** : Phase d'implémentation logicielle post-architecture terminée — Validation par tests unitaires  

---

## 1. Contexte et Objectifs de la Phase

À la suite de la formalisation de l'architecture globale du projet, cette phase de développement s'est concentrée sur la mise en production du socle logiciel expérimental. Les objectifs techniques atteints sont :
1. La standardisation complète du pipeline de prétraitement et de chargement volumique 3D.
2. La mise en place de jeux de données unifiés (unimodaux et multimodaux).
3. L'implémentation des scripts d'entraînement des modèles de référence (*baselines*).
4. L'intégration de garde-fous stricts contre les biais méthodologiques (fuite de données, non-déterminisme).
5. La couverture du code par une suite complète de tests unitaires automatisés.

---

## 2. Pipeline de Données et Prétraitement 3D (`src/data/`)

Le prétraitement des volumes médicaux a été centralisé dans un module unique ([`preprocessing.py`](../src/data/preprocessing.py)) afin de garantir une stricte équivalence de traitement entre les modalités et les modèles.

### 2.1 Normalisation d'Intensité (*Masked Z-score*)
* **Principe** : Contrairement aux images naturelles, les intensités IRM n'ont pas d'unités physiques absolues et varient fortement selon le constructeur et la séquence.
* **Algorithme** :
  1. Calcul de la moyenne ($\mu$) et de l'écart-type ($\sigma$) restreint aux seuls voxels non nuls ($I(x,y,z) \neq 0$), excluant ainsi l'arrière-plan noir (air ambiant).
  2. Centrage-réduction : $\hat{I} = \frac{I - \mu}{\sigma + \epsilon}$.
  3. Écrêtage robuste (*intensity clipping*) sur l'intervalle $[-3.0, +3.0]$ pour éliminer l'influence des voxels hyper-intenses aberrants.

### 2.2 Harmonisation Spatiale Déterministe (*Center-Crop / Zero-Pad*)
* Les acquisitions cliniques brutes présentent des matrices de dimensions hétérogènes.
* Une fonction déterministe redimensionne chaque volume vers une forme spatiale cible homogène $(H=96, W=96, D=16)$ :
  * Si la dimension axiale ou transverse dépasse la cible : découpage centré (*center-crop*).
  * Si la dimension est inférieure : remplissage symétrique par des zéros (*zero-padding*).

### 2.3 Streaming RAM sans Empreinte Disque
* Un connecteur de streaming direct via `rclone` permet de lire et désérialiser les volumes `.mha` stockés sur le cloud (Google Drive) directement dans la mémoire vive (RAM) via des descripteurs temporaires.
* Cela permet l'exécution du code sur des machines aux ressources de stockage local limitées sans nécessiter de dupliquer localement l'intégralité du dataset PI-CAI.

### 2.4 Abstraction du Dataset Multimodal ([`multimodal_dataset.py`](../src/data/multimodal_dataset.py))
* La classe `PicaiMultimodalDataset` dérive de `torch.utils.data.Dataset` et supporte dynamiquement toute combinaison de modalités (`t2w`, `adc`, `hbv`).
* Les volumes 3D normalisés d'un même patient sont empilés le long de la dimension des canaux pour former un tenseur unifié de forme $(C, D, H, W)$ où $C \in \{1, 2, 3\}$.

---

## 3. Moteurs d'Entraînement des Baselines (`src/training/`)

Deux scripts modulaires ont été implémentés : [`train_unimodal_baseline.py`](../src/training/train_unimodal_baseline.py) (évaluation isolée par séquence) et [`train_fusion_baseline.py`](../src/training/train_fusion_baseline.py) (fusion précoce multi-canaux).

### 3.1 Validation Croisée Stratifiée au Niveau Patient (5-Fold CV)
* Le partitionnement des données utilise `StratifiedKFold` appliqué au niveau des **identifiants patients uniques**, et non des coupes 2D.
* Cela garantit deux impératifs :
  1. La conservation exacte du ratio de cas cliniquement significatifs ($\text{ISUP} \ge 3$) au sein de chaque fold.
  2. L'absence totale de corrélation intra-patient entre les sous-ensembles d'apprentissage et de validation.

### 3.2 Prise en Compte du Déséquilibre de Classes
* Dans le dataset PI-CAI, la majorité des examens ne présentent pas de lésion cliniquement significative (classe minoritaire positive).
* La fonction de coût `nn.CrossEntropyLoss` intègre un vecteur de pondération inversement proportionnel à la fréquence des classes :
  $$\text{weight}_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$

### 3.3 Efficacité Computationnelle et Gestion VRAM
* **Précision Mixte (AMP)** : Utilisation de `torch.amp.GradScaler` pour réduire la mémoire occupée et accélérer les calculs matriciels sur GPU sans perte de stabilité numérique.
* **Gestion du cache GPU** : Appel explicite à `torch.cuda.empty_cache()` à chaque transition de fold pour éviter l'accumulation de mémoire résiduelle.

---

## 4. Rigueur Expérimentale et Reproductibilité (`src/utils/`)

Une infrastructure outillée a été mise en place afin de prévenir les erreurs méthodologiques courantes en apprentissage profond médical :

| Module | Rôle Technique |
| :--- | :--- |
| [`fold_guard.py`](../src/utils/fold_guard.py) | **Garde-fou anti-fuite de données** : vérification formelle par intersection d'ensembles ($\text{train\_patients} \cap \text{val\_patients} = \emptyset$). Déclenche une exception bloquante immédiate en cas de chevauchement. |
| [`seed.py`](../src/utils/seed.py) | **Déterminisme strict** : synchronisation des graines pseudo-aléatoires sur Python `random`, NumPy, PyTorch (CPU et CUDA), `PYTHONHASHSEED` et désactivation des algorithmes non déterministes de `cuDNN`. |
| [`config_loader.py`](../src/utils/config_loader.py) | **Gestion des configurations** : chargement hiérarchique via fichiers YAML (`configs/`). Priorité : `Arguments CLI > Fichier YAML > Valeurs par défaut Argparse`. Évite tout paramètre codé en dur. |
| [`checkpointing.py`](../src/utils/checkpointing.py) | **Persistance d'état** : sérialisation atomique du modèle, de l'état de l'optimiseur AdamW et du scaler AMP correspondant à la meilleure itération par fold. |
| [`results_logger.py`](../src/utils/results_logger.py) | **Traçabilité logicielle** : journalisation CSV horodatée incluant systématiquement le **commit SHA Git courant**, assurant la correspondance parfaite entre code exécuté et métriques. |

---

## 5. Banc de Tests et Validation Logicielle (`tests/`)

Pour s'assurer de la robustesse de l'implémentation avant les campagnes expérimentales lourdes, une suite de tests unitaires complète a été développée ([`tests/test_new_modules.py`](../tests/test_new_modules.py)) :

* **40 tests unitaires automatisés**, s'exécutant sur des tenseurs synthétiques en mémoire (indépendants du système de fichiers distant).
* **Périmètre validé** :
  1. *Preprocessing* : conformité mathématique du masked z-score, symétrie du center-crop/pad, comportement aux bornes.
  2. *Datasets* : dimensions de sortie, intégrité du chargement unimodal vs multimodal, typage des tenseurs retournés.
  3. *Garde-fou* : validation du passage sur partitions disjointes et levée effective d'erreur sur partitions corrompues.
  4. *Reproductibilité* : cohérence des checkpoints générés et traçabilité des logs.

---

## 6. Synthèse de la Structure du Projet

```text
prostate_xai/
├── configs/                          # Fichiers de configuration déclaratifs (YAML)
│   ├── phase1_t2w.yaml
│   ├── phase1_adc.yaml
│   ├── phase1_hbv.yaml
│   └── phase3_fusion_simple.yaml
├── docs/                             # Documentation technique et guides méthodologiques
│   ├── EDA_AND_PIPELINE_GUIDE.md
│   └── RAPPORT_TECHNIQUE_IMPLEMENTATION.md
├── src/
│   ├── data/
│   │   ├── picai_paths.py            # Résolution des chemins et métadonnées PI-CAI
│   │   ├── preprocessing.py          # Z-score masqué, cropping/padding 3D, streaming RAM
│   │   ├── unimodal_dataset.py       # Dataset unicanal
│   │   └── multimodal_dataset.py     # Dataset multicanal unifié
│   ├── models/
│   │   └── simple_cnn3d.py           # Architecture 3D avec canaux configurables
│   ├── training/
│   │   ├── train_unimodal_baseline.py# Boucle CV 5-fold unimodale
│   │   └── train_fusion_baseline.py  # Boucle CV 5-fold multimodale (early fusion)
│   └── utils/
│       ├── seed.py                   # Déterminisme complet (CPU/GPU)
│       ├── fold_guard.py             # Assertion stricte anti-fuite de données
│       ├── checkpointing.py          # Sauvegarde/reprise d'états d'entraînement
│       ├── config_loader.py          # Parseur YAML + liaison avec git commit
│       ├── training_utils.py         # Fonctions communes d'évaluation et métriques
│       └── results_logger.py         # Export append-only des métriques
└── tests/
    ├── test_picai_paths.py
    └── test_new_modules.py           # 40 tests unitaires de validation
```

---

## 7. Prochaines Étapes Techniques

1. **Complétion des baselines unimodales et de fusion simple** : exécution systématique via les configurations déclaratives YAML définies.
2. **Implémentation de l'architecture cible** : intégration des mécanismes d'attention croisée (*cross-attention*) inter-modalités.
3. **Module d'explicabilité (XAI)** : implémentation et adaptation aux volumes 3D des méthodes d'attribution (Grad-CAM 3D, Integrated Gradients) avec validation sur les cartes de probabilité lésionnelles fournies par les radiologues dans PI-CAI.
