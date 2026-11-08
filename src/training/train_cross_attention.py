"""
Training script for Step 4 & Step 5 (Chapter 4 Proposed Architecture):
Per-modality 3D SegResNet Encoders + Clinical MLP + Bidirectional Cross-Attention & Gating.

Usage:
    # Step 4 — Image-only SegResNet + Cross-Attention + Gating
    python3 -m src.training.train_cross_attention --config configs/phase4_cross_attention.yaml

    # Step 5 — Table 3.1 Configurations C5, C6, C7 (Image + Clinical variables)
    python3 -m src.training.train_cross_attention --config configs/phase5_c5_concat_clinical.yaml
    python3 -m src.training.train_cross_attention --config configs/phase5_c6_attention_clinical.yaml
    python3 -m src.training.train_cross_attention --config configs/phase5_c7_cross_attention_clinical.yaml

Priority: explicit CLI args > YAML config > argparse defaults.
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
from src.data.multimodal_dataset import (
    PicaiMultimodalDataset,
    fit_transform_clinical_features,
)
from src.data.picai_paths import find_complete_patients, load_marksheet
from src.models.cross_attention_fusion import MultimodalCrossAttentionNet
from src.utils.checkpointing import load_training_state, save_training_state
from src.utils.config_loader import apply_yaml_to_parser, get_git_hash
from src.utils.epoch_logger import log_epoch
from src.utils.fold_guard import assert_no_patient_overlap, log_fold_class_distribution
from src.utils.results_logger import save_results
from src.utils.seed import set_seed
from src.utils.training_utils import evaluate, log_fold_confusion_matrix


def _resolve_experiment_name(
    experiment_name: str | None,
    use_clinical: bool,
    fusion_mode: str,
    shared_weights: bool,
) -> str:
    if experiment_name:
        return experiment_name
    weight_tag = "shared" if shared_weights else "unshared"
    clin_tag = "_clin" if use_clinical else "_imgonly"
    return f"phase4_{fusion_mode}_{weight_tag}{clin_tag}"


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int,
    class_weight_pos: float,
    learning_rate: float,
    weight_decay: float,
    experiment: str = "",
    fold: int = 0,
    modality_dropout_prob: float = 0.0,
) -> tuple[float, list[float], list[int]]:
    """
    Train one fold for `n_epochs`, track best validation AUROC, and persist
    resumable atomic checkpoints (`last.pt` and `best.pt`).
    """
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, class_weight_pos], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = load_training_state(model, optimizer, scaler, experiment, fold, device)
    start_epoch = 0
    best_auroc = 0.0
    best_probs: list[float] = []
    best_labels: list[int] = []
    if state is not None:
        start_epoch = int(state["epoch"])
        best_auroc = float(state["best_metric"])
        best_probs = state["best_probs"]
        best_labels = state["best_labels"]
        print(f"  Reprise fold {fold} à l'epoch {start_epoch + 1}")

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        batch_count = 0

        n_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, start=1):
            if len(batch) == 3:
                images, clin, labels = batch
                images = images.to(device)
                clin = clin.to(device)
                labels = labels.to(device)
            else:
                images, labels = batch
                images = images.to(device)
                clin = None
                labels = labels.to(device)

            # Optional training-time Modality Dropout (Section 3.4.4)
            avail_mask = None
            if modality_dropout_prob > 0.0:
                b_sz, n_mod = images.shape[0], images.shape[1]
                n_tok = n_mod + (1 if clin is not None else 0)
                keep = (
                    torch.rand((b_sz, n_tok), device=device) >= modality_dropout_prob
                ).float()
                # Ensure at least one modality remains active per patient
                all_dropped = keep.sum(dim=1) == 0
                if all_dropped.any():
                    keep[all_dropped, 0] = 1.0
                avail_mask = keep

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images, clin=clin, availability_mask=avail_mask)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            batch_count += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch == start_epoch and (batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == n_batches):
                print(
                    f"    [epoch {epoch + 1} | batch {batch_idx}/{n_batches}] "
                    f"loss={loss.item():.4f} ({time.time() - epoch_start:.1f}s)",
                    flush=True,
                )

        auroc, probs, labels_list = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        print(
            f"  epoch {epoch + 1}/{n_epochs} — val AUROC: {auroc:.3f} "
            f"({epoch_time:.1f}s)"
        )

        is_best = auroc > best_auroc
        if is_best:
            best_auroc = auroc
            best_probs = probs
            best_labels = labels_list
        completed_epoch = epoch + 1
        mean_train_loss = total_loss / max(batch_count, 1)
        if experiment:
            log_epoch(
                experiment, fold, completed_epoch, mean_train_loss, auroc, best_auroc
            )
            save_training_state(
                model,
                optimizer,
                scaler,
                experiment,
                fold,
                completed_epoch,
                best_auroc,
                best_probs,
                best_labels,
                is_best,
            )

    return best_auroc, best_probs, best_labels


def run_cross_attention_cv(
    modalities: list[str],
    experiment_name: str | None = None,
    n_folds: int = 5,
    batch_size: int = 8,
    n_epochs: int = 30,
    class_weight_pos: float = 2.5,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    init_filters: int = 16,
    encoder_dim: int = 256,
    fusion_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    ffn_dim: int = 256,
    dropout: float = 0.15,
    shared_weights: bool = False,
    use_clinical: bool = False,
    fusion_mode: str = "cross_attention",
    use_gating: bool = True,
    align_to_t2w: bool = True,
    target_shape: tuple[int, int, int] = (96, 96, 16),
    modality_dropout_prob: float = 0.0,
    seed: int = 42,
    config_file: str = "argparse",
) -> list[float]:
    """
    5-fold patient-level stratified cross-validation for the proposed
    3D SegResNet + Cross-Attention & Gating architecture (Steps 4 & 5).
    """
    set_seed(seed)
    git_hash = get_git_hash()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = _resolve_experiment_name(
        experiment_name, use_clinical, fusion_mode, shared_weights
    )
    modality_str = "+".join(modalities) + ("+clin" if use_clinical else "")
    print(
        f"Device: {device} | exp: {exp_name} | modalités: {modality_str} | "
        f"fusion: {fusion_mode} | shared_weights: {shared_weights} | "
        f"align_to_t2w: {align_to_t2w} | shape: {target_shape} | "
        f"seed: {seed} | git: {git_hash}"
    )

    complete_ids, incomplete = find_complete_patients(DATA_ROOT)
    print(
        f"Patients complets (3 canaux): {len(complete_ids)} | "
        f"incomplets: {len(incomplete)}"
    )

    marksheet = load_marksheet(DATA_ROOT, complete_ids=complete_ids)
    patient_ids = marksheet["patient_id"].tolist()
    labels = marksheet["label"].tolist()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aurocs: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        fold_num = fold_idx + 1
        print(
            f"\n=== Fold {fold_num}/{n_folds} — {exp_name} ({modality_str}) ==="
        )

        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]

        assert_no_patient_overlap(train_ids, val_ids)
        log_fold_class_distribution([labels[i] for i in train_idx], fold_num, "train")
        log_fold_class_distribution([labels[i] for i in val_idx], fold_num, "val")

        train_clin, val_clin = None, None
        if use_clinical:
            train_clin, val_clin, _ = fit_transform_clinical_features(
                marksheet, train_ids, val_ids
            )

        train_ds = PicaiMultimodalDataset(
            DATA_ROOT,
            train_ids,
            marksheet,
            modalities=modalities,
            target_shape=target_shape,
            align_to_t2w=align_to_t2w,
            clinical_features=train_clin,
        )
        val_ds = PicaiMultimodalDataset(
            DATA_ROOT,
            val_ids,
            marksheet,
            modalities=modalities,
            target_shape=target_shape,
            align_to_t2w=align_to_t2w,
            clinical_features=val_clin,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=(device.type == "cuda"),
        )

        model = MultimodalCrossAttentionNet(
            modalities=modalities,
            init_filters=init_filters,
            encoder_dim=encoder_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            shared_weights=shared_weights,
            use_clinical=use_clinical,
            fusion_mode=fusion_mode,  # type: ignore[arg-type]
            use_gating=use_gating,
            n_classes=2,
        ).to(device)

        best_auroc, best_probs, best_labels = train_one_fold(
            model,
            train_loader,
            val_loader,
            device,
            n_epochs,
            class_weight_pos,
            learning_rate,
            weight_decay,
            experiment=exp_name,
            fold=fold_num,
            modality_dropout_prob=modality_dropout_prob,
        )
        fold_aurocs.append(best_auroc)

        log_fold_confusion_matrix(best_labels, best_probs, fold=fold_num)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs))
    print(f"\n=== Résumé — {exp_name} ({modality_str}) ===")
    print(f"AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")

    config = {
        "n_folds": n_folds,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "class_weight_pos": class_weight_pos,
        "model": "MultimodalCrossAttentionNet",
        "fusion_mode": fusion_mode,
        "shared_weights": shared_weights,
        "use_clinical": use_clinical,
        "align_to_t2w": align_to_t2w,
        "seed": seed,
        "git_hash": git_hash,
        "config_file": config_file,
    }
    save_results(exp_name, modality_str, fold_aurocs, config)
    return fold_aurocs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4 & 5: 3D SegResNet + Cross-Attention & Clinical Fusion."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["t2w", "adc", "hbv"],
        default=["t2w", "adc", "hbv"],
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--class_weight_pos", type=float, default=2.5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--init_filters", type=int, default=16)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--shared_weights", action="store_true", default=False)
    parser.add_argument("--use_clinical", action="store_true", default=False)
    parser.add_argument(
        "--fusion_mode",
        choices=["cross_attention", "attention", "concat"],
        default="cross_attention",
    )
    parser.add_argument("--use_gating", type=bool, default=True)
    parser.add_argument("--align_to_t2w", type=bool, default=True)
    parser.add_argument(
        "--target_shape", nargs=3, type=int, default=[96, 96, 16]
    )
    parser.add_argument("--modality_dropout_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    pre_args, _ = parser.parse_known_args()
    apply_yaml_to_parser(parser, pre_args.config)
    args = parser.parse_args()

    run_cross_attention_cv(
        modalities=args.modalities,
        experiment_name=args.experiment_name,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        class_weight_pos=args.class_weight_pos,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        init_filters=args.init_filters,
        encoder_dim=args.encoder_dim,
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        shared_weights=bool(args.shared_weights),
        use_clinical=bool(args.use_clinical),
        fusion_mode=args.fusion_mode,
        use_gating=bool(args.use_gating),
        align_to_t2w=bool(args.align_to_t2w),
        target_shape=tuple(args.target_shape),
        modality_dropout_prob=args.modality_dropout_prob,
        seed=args.seed,
        config_file=str(args.config) if args.config else "argparse",
    )
