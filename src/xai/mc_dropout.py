"""
Monte Carlo Dropout Uncertainty Quantification & Selective Abstention Rule
(Chapter 4, Section 4.7 of the thesis manuscript).

Implements:
- Activation of Dropout layers at inference time (`enable_mc_dropout`) while
  keeping normalization layers in evaluation mode.
- Stochastic forward passes for T in {10, 25, 50} (Section 4.7).
- Predictive mean probability `p_bar`, predictive variance `sigma^2`, and
  predictive entropy `H(p_bar)`.
- Selective abstention evaluation (`evaluate_selective_abstention`): measures
  diagnostic AUROC / sensitivity / specificity on retained cases when referring
  the top-q% most uncertain patients to human radiologist review.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Put the model in eval mode, then explicitly enable all `nn.Dropout` layers
    so stochastic dropout masks are sampled at each forward pass (Gal & Ghahramani, 2016).
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    images: torch.Tensor,
    clin: torch.Tensor | None = None,
    n_passes: int = 25,
    availability_mask: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """
    Run `n_passes` stochastic forward passes with MC Dropout active.

    Parameters
    ----------
    model : nn.Module
    images : torch.Tensor of shape `[B, C, D, H, W]`
    clin : optional torch.Tensor of shape `[B, 4]`
    n_passes : int, e.g. 10, 25, or 50 (Section 4.7)

    Returns
    -------
    dict with numpy arrays of shape `(B,)`:
        - `"mean_prob"` : mean positive-class probability across T passes
        - `"std_prob"`  : standard deviation of positive-class probability
        - `"var_prob"`  : predictive variance sigma^2
        - `"entropy"`   : predictive binary entropy H(mean_prob)
        - `"all_probs"` : array of shape `(n_passes, B)`
    """
    enable_mc_dropout(model)
    device = next(model.parameters()).device
    x = images.to(device)
    c = clin.to(device) if clin is not None else None
    mask = availability_mask.to(device) if availability_mask is not None else None

    pass_probs: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_passes):
            if c is not None or mask is not None:
                logits = model(x, clin=c, availability_mask=mask)
            else:
                logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pass_probs.append(probs)

    model.eval()
    stacked = np.stack(pass_probs, axis=0)  # (T, B)
    mean_prob = stacked.mean(axis=0)
    var_prob = stacked.var(axis=0, ddof=0)
    std_prob = np.sqrt(var_prob)
    p_clip = np.clip(mean_prob, 1e-8, 1.0 - 1e-8)
    entropy = -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))

    return {
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "var_prob": var_prob,
        "entropy": entropy,
        "all_probs": stacked,
    }


def evaluate_selective_abstention(
    labels: list[int] | np.ndarray,
    mean_probs: list[float] | np.ndarray,
    uncertainties: list[float] | np.ndarray,
    abstention_rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25),
) -> list[dict[str, float]]:
    """
    Evaluate the Selective Abstention Rule (Section 4.7):
    Patients are sorted by increasing uncertainty (`var_prob` or `std_prob`).
    For each abstention rate `q` (fraction of highest-uncertainty cases referred
    to a human radiologist), compute the AUROC and accuracy on the `(1 - q)`
    retained cases.
    """
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(mean_probs, dtype=np.float64)
    u = np.asarray(uncertainties, dtype=np.float64)
    n_total = len(y_true)

    order = np.argsort(u)  # ascending uncertainty (most confident first)
    curve: list[dict[str, float]] = []

    for q in abstention_rates:
        n_keep = max(2, int(round(n_total * (1.0 - q))))
        keep_idx = order[:n_keep]
        y_k = y_true[keep_idx]
        p_k = y_prob[keep_idx]
        auroc_k = (
            float(roc_auc_score(y_k, p_k)) if len(np.unique(y_k)) >= 2 else float("nan")
        )
        acc_k = float(np.mean((p_k >= 0.5).astype(np.int64) == y_k))
        curve.append(
            {
                "abstention_rate": float(q),
                "retained_fraction": float(n_keep / max(n_total, 1)),
                "n_retained": int(n_keep),
                "n_referred": int(n_total - n_keep),
                "auroc_retained": auroc_k,
                "accuracy_retained": acc_k,
                "uncertainty_threshold": float(u[keep_idx[-1]]),
            }
        )
    return curve
