"""
Unimodal baseline training, run as a script from the command line:

    python -m src.training.train_unimodal_baseline --modality t2w
    python -m src.training.train_unimodal_baseline --modality adc
    python -m src.training.train_unimodal_baseline --modality hbv

Replaces the earlier Colab-cell version — same logic, but as a proper
script so results are reproducible via `git log` + command line args,
not "whatever was in the notebook cell at the time".
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import DATA_ROOT
from src.data.picai_paths import find_complete_patients, load_marksheet
from src.data.unimodal_dataset import PicaiUnimodalDataset
from src.models.simple_cnn3d import SimpleCNN3D


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return roc_auc_score(all_labels, all_probs)


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int,
    class_weight_pos: float,
) -> float:
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, class_weight_pos], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # AMP is CUDA-only; on CPU (common on a laptop without a discrete GPU)
    # we fall back to plain full-precision training rather than silently
    # doing nothing useful with a GradScaler that has no CUDA to scale for.
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_auroc = 0.0
    for epoch in range(n_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        auroc = evaluate(model, val_loader, device)
        best_auroc = max(best_auroc, auroc)
        print(f"  epoch {epoch + 1}/{n_epochs} — val AUROC: {auroc:.3f}")

    return best_auroc


def run_baseline_cv(
    modality: str,
    n_folds: int = 5,
    batch_size: int = 8,
    n_epochs: int = 30,
    class_weight_pos: float = 2.5,
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    complete_ids, incomplete = find_complete_patients(DATA_ROOT)
    print(f"Complete patients (3 channels): {len(complete_ids)} | incomplete: {len(incomplete)}")

    marksheet = load_marksheet(DATA_ROOT, complete_ids=complete_ids)
    patient_ids = marksheet["patient_id"].tolist()
    labels = marksheet["label"].tolist()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aurocs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        print(f"\n=== Fold {fold + 1}/{n_folds} — modality: {modality} ===")
        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]

        train_ds = PicaiUnimodalDataset(DATA_ROOT, train_ids, marksheet, modality=modality)
        val_ds = PicaiUnimodalDataset(DATA_ROOT, val_ids, marksheet, modality=modality)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        model = SimpleCNN3D(in_channels=1, n_classes=2).to(device)
        fold_auroc = train_one_fold(
            model, train_loader, val_loader, device, n_epochs, class_weight_pos
        )
        fold_aurocs.append(fold_auroc)

    print(f"\n=== Summary — {modality} ===")
    print(f"AUROC: {np.mean(fold_aurocs):.3f} +/- {np.std(fold_aurocs):.3f}")
    return fold_aurocs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["t2w", "adc", "hbv"], required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=30)
    args = parser.parse_args()

    run_baseline_cv(
        modality=args.modality,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )
