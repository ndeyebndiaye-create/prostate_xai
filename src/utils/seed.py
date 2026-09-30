"""
Seed manager — fixe toutes les sources d'aléatoire du projet en un seul appel.

Pourquoi c'est crucial pour une thèse :
  Dans 6 mois, l'encadrant peut demander de reproduire un chiffre précis.
  Si les seeds ne sont pas fixées de manière exhaustive, deux runs avec
  les mêmes hyperparamètres peuvent donner des AUROC légèrement différents.
  Ce fichier garantit la reproductibilité bit-à-bit sur CPU et GPU.

Sources d'aléatoire à couvrir :
  - random    : utilisé par Python built-in et certaines libs
  - numpy     : utilisé par scikit-learn (StratifiedKFold, etc.)
  - torch     : utilisé par les couches, l'initialisation des poids
  - cuda      : GPU multi-stream
  - cudnn     : optimisations GPU déterministes (plus lent, mais reproductible)
  - PYTHONHASHSEED : randomisation des hash Python (affecte l'ordre des sets/dicts)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fixe toutes les sources d'aléatoire pour garantir la reproductibilité.

    Doit être appelé au tout début de chaque script d'entraînement,
    avant toute initialisation de modèle ou chargement de données.

    Parameters
    ----------
    seed:
        Graine aléatoire. Valeur par défaut : 42 (convention du projet).
        Changer la seed = nouvelle expérience indépendante (ex. pour
        vérifier la stabilité des résultats).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn.deterministic = True : force des algorithmes déterministes pour
    # les opérations convolutives. Légèrement plus lent (~5-10%) mais
    # garantit la reproductibilité exacte bit-à-bit sur GPU.
    torch.backends.cudnn.deterministic = True

    # cudnn.benchmark = False : désactive la recherche automatique de
    # l'algorithme le plus rapide (qui est non-déterministe selon la taille
    # des batches). Doit être False quand deterministic=True.
    torch.backends.cudnn.benchmark = False

    # PYTHONHASHSEED : fixe le hasard de Python pour les opérations sur les
    # structures de données (set, dict). Affecte l'ordre d'itération dans
    # certains cas Python 3.7+.
    os.environ["PYTHONHASHSEED"] = str(seed)
