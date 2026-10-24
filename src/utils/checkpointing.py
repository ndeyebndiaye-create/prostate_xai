"""
Sauvegarde des checkpoints (meilleur modèle par fold).

Pourquoi sauvegarder les checkpoints :
  - Reprendre l'entraînement après une coupure (serveur, batterie, etc.)
  - Charger le meilleur modèle d'un fold pour l'inférence ou l'XAI (Phase 6)
  - Comparer deux architectures en chargeant leurs checkpoints directement

Convention de nommage :
  `/workspace/checkpoints/{experiment_name}/fold_{N}/{last,best}.pt`
  Ex: `/workspace/checkpoints/phase1_t2w_only/fold_3/best.pt`

Le dossier `checkpoints/` est dans `.gitignore` (fichiers potentiellement
lourds — plusieurs centaines de MB selon l'architecture).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

OUTPUT_ROOT = Path(os.environ.get("RUN_OUTPUT_ROOT", "/workspace"))
_CKPT_DIR = OUTPUT_ROOT / "checkpoints"


def fold_checkpoint_paths(experiment: str, fold: int) -> tuple[Path, Path]:
    """Return the last and best checkpoint paths for one experiment fold."""
    fold_dir = _CKPT_DIR / experiment / f"fold_{fold}"
    return fold_dir / "last.pt", fold_dir / "best.pt"


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def save_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    experiment: str,
    fold: int,
    epoch: int,
    best_metric: float,
    best_probs: list[float],
    best_labels: list[int],
    is_best: bool,
) -> None:
    """Persist resumable training state after every epoch."""
    last_path, best_path = fold_checkpoint_paths(experiment, fold)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_probs": best_probs,
        "best_labels": best_labels,
    }
    _atomic_save(payload, last_path)
    if is_best:
        _atomic_save(payload, best_path)


def load_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    experiment: str,
    fold: int,
    device: torch.device,
) -> dict | None:
    """Restore the most recent fold state, or return None for a new fold."""
    last_path, _ = fold_checkpoint_paths(experiment, fold)
    if not last_path.exists():
        return None

    state = torch.load(last_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    return state


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
    _, path = fold_checkpoint_paths(experiment, fold)
    _atomic_save({"model_state_dict": model.state_dict()}, path)
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
    _, path = fold_checkpoint_paths(experiment, fold)
    legacy_path = _CKPT_DIR / f"{experiment}_fold{fold}_best.pt"
    if not path.exists() and legacy_path.exists():
        path = legacy_path
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {path}\n"
            f"Lancer l'entraînement avec experiment='{experiment}' d'abord."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model
