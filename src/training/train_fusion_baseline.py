"""
Fusion baseline training — simple channel-concatenation multimodal model.

    # Phase 2 : T2W + ADC (2 canaux)
    python -m src.training.train_fusion_baseline --modalities t2w adc

    # Phase 3 : les 3 séquences (3 canaux) — via argparse
    python -m src.training.train_fusion_baseline --modalities t2w adc hbv

    # Phase 3 — via YAML (reproductible, recommandé)
    python -m src.training.train_fusion_baseline --config configs/phase3_fusion_simple.yaml

Priorité : CLI explicite > YAML > defaults argparse.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import DATA_ROOT
from src.data.multimodal_dataset import PicaiMultimodalDataset
from src.data.picai_paths import find_complete_patients, load_marksheet
from src.models.simple_cnn3d import SimpleCNN3D
from src.utils.checkpointing import save_checkpoint
from src.utils.config_loader import apply_yaml_to_parser, get_git_hash
from src.utils.fold_guard import assert_no_patient_overlap, log_fold_class_distribution
from src.utils.results_logger import save_results
from src.utils.seed import set_seed
from src.utils.training_utils import evaluate, log_fold_confusion_matrix


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int,
    class_weight_pos: float,
    experiment: str = "",
    fold: int = 0,
) -> tuple[float, list[float], list[int]]:
    """
    Entraîne pour n_epochs, suit le meilleur AUROC, sauvegarde le checkpoint.

    Returns
    -------
    (best_auroc, best_probs, best_labels)
    """
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, class_weight_pos], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auroc = 0.0
    best_probs: list[float] = []
    best_labels: list[int] = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        auroc, probs, labels_list = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        print(
            f"  epoch {epoch + 1}/{n_epochs} — val AUROC: {auroc:.3f} "
            f"({epoch_time:.1f}s)"
        )

        if auroc > best_auroc:
            best_auroc = auroc
            best_probs = probs
            best_labels = labels_list
            if experiment:
                save_checkpoint(model, experiment, fold, auroc)

    return best_auroc, best_probs, best_labels


def run_fusion_cv(
    modalities: list[str],
    n_folds: int = 5,
    batch_size: int = 8,
    n_epochs: int = 30,
    class_weight_pos: float = 2.5,
    seed: int = 42,
    config_file: str = "argparse",
) -> list[float]:
    """
    CV 5-fold stratifiée pour la fusion multimodale (canaux empilés).

    Parameters
    ----------
    modalities:
        Liste des séquences à fusionner, ex. ["t2w","adc"] ou ["t2w","adc","hbv"].
    seed:
        Graine aléatoire — identique à celle de train_unimodal_baseline pour
        que les folds soient exactement les mêmes (comparaison valide).
    """
    # ── Reproductibilité ─────────────────────────────────────────────────────
    set_seed(seed)
    git_hash = get_git_hash()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modality_str = "+".join(modalities)
    print(
        f"Device: {device} | modalités: {modality_str} "
        f"({len(modalities)} canaux) | seed: {seed} | git: {git_hash}"
    )

    # ── Données ──────────────────────────────────────────────────────────────
    complete_ids, incomplete = find_complete_patients(DATA_ROOT)
    print(f"Patients complets (3 canaux): {len(complete_ids)} | incomplets: {len(incomplete)}")

    marksheet = load_marksheet(DATA_ROOT, complete_ids=complete_ids)
    patient_ids = marksheet["patient_id"].tolist()
    labels = marksheet["label"].tolist()

    # ── Nom d'expérience déterministe ─────────────────────────────────────────
    if set(modalities) == {"t2w", "adc", "hbv"}:
        experiment_name = "fusion_simple"
    else:
        experiment_name = f"fusion_{'_'.join(modalities)}"

    # ── CV 5-fold ─────────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aurocs: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        fold_num = fold_idx + 1
        print(f"\n=== Fold {fold_num}/{n_folds} — modalities: {modality_str} ===")

        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]

        # ── Guards anti-fuite ─────────────────────────────────────────────
        assert_no_patient_overlap(train_ids, val_ids)
        log_fold_class_distribution([labels[i] for i in train_idx], fold_num, "train")
        log_fold_class_distribution([labels[i] for i in val_idx], fold_num, "val")

        train_ds = PicaiMultimodalDataset(
            DATA_ROOT, train_ids, marksheet, modalities=modalities
        )
        val_ds = PicaiMultimodalDataset(
            DATA_ROOT, val_ids, marksheet, modalities=modalities
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        model = SimpleCNN3D(in_channels=len(modalities), n_classes=2).to(device)
        best_auroc, best_probs, best_labels = train_one_fold(
            model, train_loader, val_loader, device,
            n_epochs, class_weight_pos,
            experiment=experiment_name, fold=fold_num,
        )
        fold_aurocs.append(best_auroc)

        log_fold_confusion_matrix(best_labels, best_probs, fold=fold_num)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Résumé ────────────────────────────────────────────────────────────────
    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    print(f"\n=== Résumé — {modality_str} ===")
    print(f"AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}")

    config = {
        "n_folds": n_folds,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "class_weight_pos": class_weight_pos,
        "model": "SimpleCNN3D",
        "in_channels": len(modalities),
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "seed": seed,
        "git_hash": git_hash,
        "config_file": config_file,
    }
    save_results(experiment_name, modality_str, fold_aurocs, config)

    return fold_aurocs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fusion baseline PI-CAI — empile les modalités comme canaux."
    )

    # ── Étape 1 : définir tous les arguments ──────────────────────────────────
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Fichier YAML de config optionnel (Option A : CLI > YAML > defaults).",
    )
    parser.add_argument(
        "--modalities", nargs="+", choices=["t2w", "adc", "hbv"], default=None,
        help="Séquences à fusionner. Ex : --modalities t2w adc hbv. "
             "Requis (via CLI ou YAML).",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--class_weight_pos", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)

    # ── Étape 2-3 : pré-parse + injection YAML comme defaults ────────────────
    pre_args, _ = parser.parse_known_args()
    apply_yaml_to_parser(parser, pre_args.config)

    # ── Étape 4 : parse final ─────────────────────────────────────────────────
    args = parser.parse_args()

    if args.modalities is None:
        parser.error(
            "--modalities est requis. Ex : --modalities t2w adc hbv, "
            "ou l'inclure dans un YAML via --config."
        )

    run_fusion_cv(
        modalities=args.modalities,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        class_weight_pos=args.class_weight_pos,
        seed=args.seed,
        config_file=str(args.config) if args.config else "argparse",
    )
