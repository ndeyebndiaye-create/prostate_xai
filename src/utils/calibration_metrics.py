"""
Clinical & Calibration Metrics (Chapter 3, Section 3.8):
- Threshold locking on validation set:
  * Target clinical sensitivity (default 90%, Section 3.8.1)
  * Youden's J index (maximizing Sensitivity + Specificity - 1)
- Full diagnostic confusion metrics: Sensitivity, Specificity, PPV (Precision), NPV, F1-score
- Probabilistic calibration: Expected Calibration Error (ECE) & Brier Score
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve


def find_threshold_at_sensitivity(
    labels: list[int] | np.ndarray,
    probs: list[float] | np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    """
    Find the highest decision threshold on `probs` that achieves at least
    `target_sensitivity` on `labels` (maximizing specificity subject to
    sensitivity >= target_sensitivity).
    """
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) == 0:
        return float(np.min(y_prob))
    # Among valid indices, pick the one with lowest FPR (highest specificity)
    best_idx = valid[np.argmin(fpr[valid])]
    thresh = float(thresholds[best_idx])
    if not np.isfinite(thresh):
        thresh = 0.5
    return thresh


def find_youden_threshold(
    labels: list[int] | np.ndarray,
    probs: list[float] | np.ndarray,
) -> float:
    """
    Find the decision threshold maximizing Youden's J statistic:
        J = Sensitivity + Specificity - 1 = TPR - FPR.
    """
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    thresh = float(thresholds[best_idx])
    if not np.isfinite(thresh):
        thresh = 0.5
    return thresh


def compute_expected_calibration_error(
    labels: list[int] | np.ndarray,
    probs: list[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with `n_bins` uniform bins in [0, 1].
        ECE = sum_{b=1}^{B} (|I_b| / N) * |acc(I_b) - conf(I_b)|
    """
    y_true = np.asarray(labels, dtype=np.float64)
    y_prob = np.asarray(probs, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            ece += (count / n) * abs(acc - conf)
    return float(ece)


def compute_clinical_and_calibration_metrics(
    labels: list[int] | np.ndarray,
    probs: list[float] | np.ndarray,
    threshold: float | None = None,
    threshold_strategy: str = "sens90",
    target_sensitivity: float = 0.90,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute the full diagnostic and calibration metrics required by Table 5.1
    (Section 3.8 of the thesis):
    - auroc
    - threshold
    - sensitivity (Recall)
    - specificity
    - ppv (Precision / Positive Predictive Value)
    - npv (Negative Predictive Value)
    - f1
    - brier
    - ece (Expected Calibration Error)
    """
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)

    auroc = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) >= 2
        else 0.5
    )

    if threshold is None:
        if threshold_strategy == "sens90":
            threshold = find_threshold_at_sensitivity(
                y_true, y_prob, target_sensitivity=target_sensitivity
            )
        elif threshold_strategy == "youden":
            threshold = find_youden_threshold(y_true, y_prob)
        else:
            threshold = 0.5

    preds = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn + 1e-8))
    specificity = float(tn / (tn + fp + 1e-8))
    ppv = float(tp / (tp + fp + 1e-8))
    npv = float(tn / (tn + fn + 1e-8))
    f1 = float(2.0 * tp / (2.0 * tp + fp + fn + 1e-8))
    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return {
        "auroc": auroc,
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "brier": brier,
        "ece": ece,
    }
