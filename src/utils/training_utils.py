"""
Utilitaires partagés d'entraînement — utilisés par tous les scripts de training.

Centralise deux fonctions qui étaient dupliquées entre train_unimodal_baseline.py
et train_fusion_baseline.py :
  - evaluate()                   : calcule AUROC + collecte probs/labels
  - log_fold_confusion_matrix()  : dérive sensibilité/spécificité depuis les probs

Règle : si evaluate() ou une fonction de métriques doit être modifiée, on le fait
ici une seule fois — les deux scripts de training en bénéficient automatiquement.

Pourquoi aller au-delà de l'AUROC :
  L'AUROC mesure la discrimination globale mais ne dit pas "combien de vrais
  cancers sont manqués" (sensibilité) ni "combien de faux positifs sont générés"
  (spécificité). Dans un contexte clinique, ces deux métriques sont aussi
  importantes que l'AUROC. Un modèle avec AUROC=0.75 mais sensibilité=0.3
  manquerait 70% des cancers — inacceptable en clinique.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[float], list[int]]:
    """
    Évalue le modèle sur un DataLoader et retourne l'AUROC + les prédictions brutes.

    Retourner les probs/labels (pas seulement l'AUROC) permet de calculer
    d'autres métriques (confusion matrix, sensibilité, spécificité) sans
    refaire un second passage sur le loader.

    Parameters
    ----------
    model:
        Modèle PyTorch en mode eval.
    loader:
        DataLoader de validation.
    device:
        Dispositif de calcul (CPU ou CUDA).

    Returns
    -------
    (auroc, all_probs, all_labels) :
        - auroc       : float, AUROC sur l'ensemble du loader
        - all_probs   : list[float], probabilité classe 1 pour chaque patient
        - all_labels  : list[int], label réel (0 ou 1) pour chaque patient
    """
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    auroc = roc_auc_score(all_labels, all_probs)
    return auroc, all_probs, all_labels


def log_fold_confusion_matrix(
    all_labels: list[int],
    all_probs: list[float],
    threshold: float = 0.5,
    fold: int | None = None,
) -> dict[str, float | int]:
    """
    Calcule et affiche la confusion matrix ainsi que sensibilité et spécificité.

    Utilise un seuil de décision fixe (0.5 par défaut) pour convertir les
    probabilités en prédictions binaires. Ce seuil est distinct de l'AUROC
    qui est calculé sur tous les seuils possibles.

    Parameters
    ----------
    all_labels:
        Labels réels (0 ou 1), un par patient.
    all_probs:
        Probabilités de la classe 1, une par patient.
    threshold:
        Seuil de décision pour binariser les probabilités. 0.5 = convention
        standard. Un seuil plus bas augmente la sensibilité (moins de cancers
        manqués) au détriment de la spécificité (plus de faux positifs).
    fold:
        Numéro du fold (optionnel, pour l'affichage).

    Returns
    -------
    dict contenant : tn, fp, fn, tp, sensitivity, specificity.
    """
    preds = [1 if p >= threshold else 0 for p in all_probs]
    tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-8)  # taux de vrais positifs (rappel classe 1)
    specificity = tn / (tn + fp + 1e-8)  # taux de vrais négatifs

    fold_label = f"fold {fold}" if fold is not None else "fold"
    print(
        f"  [{fold_label} | confusion matrix @ threshold={threshold}] "
        f"TP={tp}, TN={tn}, FP={fp}, FN={fn} | "
        f"sensitivity={sensitivity:.3f}, specificity={specificity:.3f}"
    )

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
