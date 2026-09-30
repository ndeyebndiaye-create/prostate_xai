"""
Sauvegarde des checkpoints (meilleur modèle par fold).

Pourquoi sauvegarder les checkpoints :
  - Reprendre l'entraînement après une coupure (serveur, batterie, etc.)
  - Charger le meilleur modèle d'un fold pour l'inférence ou l'XAI (Phase 6)
  - Comparer deux architectures en chargeant leurs checkpoints directement

Convention de nommage :
  `checkpoints/{experiment_name}_fold{N}_best.pt`
  Ex: `checkpoints/phase1_t2w_only_fold3_best.pt`

Le dossier `checkpoints/` est dans `.gitignore` (fichiers potentiellement
lourds — plusieurs centaines de MB selon l'architecture).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

_CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def save_checkpoint(
    model: nn.Module,
    experiment: str,
    fold: int,
    auroc: float,
) -> Path:
    """
    Sauvegarde le state_dict du modèle si c'est le meilleur AUROC pour ce fold.

    Appelé depuis train_one_fold() chaque fois que l'AUROC de validation
    s'améliore. Écrase le checkpoint précédent du même fold (on ne garde que
    le meilleur, pas l'historique complet des epochs).

    Parameters
    ----------
    model:
        Le modèle PyTorch à sauvegarder.
    experiment:
        Nom de l'expérience (ex. "phase1_t2w_only") — fait partie du nom de fichier.
    fold:
        Numéro du fold (1-based).
    auroc:
        AUROC atteint (loggé dans le message de confirmation).

    Returns
    -------
    Path du fichier .pt sauvegardé.
    """
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = _CKPT_DIR / f"{experiment}_fold{fold}_best.pt"
    torch.save(model.state_dict(), path)
    print(f"  [checkpoint] fold {fold} — new best AUROC {auroc:.4f} → {path.name}")
    return path


def load_checkpoint(model: nn.Module, experiment: str, fold: int) -> nn.Module:
    """
    Charge le meilleur checkpoint d'un fold dans un modèle existant.

    Utile pour l'inférence post-entraînement ou pour reprendre l'évaluation
    sans re-entraîner.

    Parameters
    ----------
    model:
        Modèle PyTorch avec la même architecture que celui sauvegardé.
    experiment:
        Nom de l'expérience (doit correspondre exactement à celui utilisé
        lors de `save_checkpoint`).
    fold:
        Numéro du fold (1-based).

    Returns
    -------
    Le modèle avec les poids chargés (même objet, modifié in-place).
    """
    path = _CKPT_DIR / f"{experiment}_fold{fold}_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {path}\n"
            f"Lancer l'entraînement avec experiment='{experiment}' d'abord."
        )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model
