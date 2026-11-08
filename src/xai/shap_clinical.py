"""
Exact Shapley Value Attribution (SHAP) on the Clinical Branch Logit
(Chapter 4, Section 4.6.2, Equation 4.6.2 of the thesis manuscript).

Since the clinical branch takes exactly M = 4 structured variables:
    [patient_age, log_psa, psad, prostate_volume]
there are only 2^4 = 16 coalitions S subseteq {1, 2, 3, 4}.
We therefore compute the EXACT Shapley values phi_j on the positive-class
logit f(x_img, x_clin) (before softmax/sigmoid, preserving linear additivity
as required by Section 4.6.2) without any sampling approximation error,
and without having to re-run the 3D SegResNet encoders across the 16 coalitions!
"""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import torch
import torch.nn as nn

from src.data.multimodal_dataset import CLINICAL_FEATURE_NAMES


def compute_exact_clinical_shap_on_logit(
    model: nn.Module,
    images: torch.Tensor,
    clin: torch.Tensor,
    baseline_clin: torch.Tensor | None = None,
    target_class: int = 1,
) -> dict[str, float]:
    """
    Compute exact Shapley values `phi_j` on the logit `z_{target_class}` for the
    4 clinical features `[patient_age, log_psa, psad, prostate_volume]`, holding
    the patient's 3D MRI representation fixed.

    Because the clinical features are z-score standardized on the training fold,
    `baseline_clin = [0, 0, 0, 0]` corresponds to the population mean of the
    training cohort (or a user-supplied reference vector).

    Satisfies the exact efficiency property:
        f(x_img, x_clin) - f(x_img, baseline_clin) = sum_{j=1}^4 phi_j

    Returns
    -------
    dict[str, float]
        Keys: `"patient_age"`, `"log_psa"`, `"psad"`, `"prostate_volume"`,
        `"base_logit"`, `"full_logit"`.
    """
    model.eval()
    device = next(model.parameters()).device
    x = images.detach().to(device)
    c = clin.detach().to(device)
    if c.ndim == 1:
        c = c.unsqueeze(0)
    if baseline_clin is None:
        base_c = torch.zeros_like(c, device=device)
    else:
        base_c = baseline_clin.detach().to(device)
        if base_c.ndim == 1:
            base_c = base_c.unsqueeze(0)

    n_features = c.shape[1]
    feature_names = (
        CLINICAL_FEATURE_NAMES
        if n_features == len(CLINICAL_FEATURE_NAMES)
        else [f"feat_{i}" for i in range(n_features)]
    )

    # Pre-encode the 3D MRI volumes ONCE (avoids 16x 3D CNN forward passes)
    with torch.no_grad():
        img_token_list = model.encode_modalities(x)
        cached_img_tokens = torch.stack(img_token_list, dim=1)  # [1, M, D]

        def _forward_from_cached_img(clin_batch: torch.Tensor) -> torch.Tensor:
            bsz = clin_batch.shape[0]
            img_toks = cached_img_tokens.expand(bsz, -1, -1).clone()
            h_clin = model.clinical_encoder(clin_batch)
            t_clin = model.clinical_proj(h_clin).unsqueeze(1)  # [B, 1, D]

            if model.fusion_mode == "concat":
                all_mod = torch.cat([img_toks, t_clin], dim=1)
                fused = model.concat_proj(all_mod.flatten(1))
                if model.gating is not None:
                    gated, _ = model.gating(all_mod, None)
                    fused = fused + gated
                return model.classifier(fused)[:, target_class]

            if model.fusion_mode == "cross_attention" and model.cross_attn is not None:
                img_toks, t_clin = model.cross_attn(img_toks, t_clin)

            mod_tokens = torch.cat([img_toks, t_clin], dim=1)
            cls_tok = model.cls_token.expand(bsz, -1, -1)
            seq = torch.cat([cls_tok, mod_tokens], dim=1)
            seq = seq + model.token_type_embeddings[:, : seq.shape[1], :]
            for block in model.transformer_blocks:
                seq = block(seq)
            cls_out = seq[:, 0, :]
            refined_mod = seq[:, 1:, :]
            if model.gating is not None:
                gated, _ = model.gating(refined_mod, None)
                final_repr = cls_out + gated
            else:
                final_repr = cls_out
            return model.classifier(final_repr)[:, target_class]

        # Build all 2^M = 16 coalition inputs in a single batch!
        all_subsets: list[tuple[int, ...]] = []
        coalition_tensors: list[torch.Tensor] = []
        for r in range(n_features + 1):
            for subset in combinations(range(n_features), r):
                all_subsets.append(subset)
                vec = base_c.clone()
                for idx in subset:
                    vec[0, idx] = c[0, idx]
                coalition_tensors.append(vec)

        batch_coalitions = torch.cat(coalition_tensors, dim=0)  # [16, 4]
        logits_all = _forward_from_cached_img(batch_coalitions).cpu().numpy()

    subset_to_val = {
        frozenset(sub): float(logits_all[i]) for i, sub in enumerate(all_subsets)
    }

    shap_values: dict[str, float] = {}
    fact_n = math.factorial(n_features)
    all_indices = set(range(n_features))

    for j in range(n_features):
        others = all_indices - {j}
        phi_j = 0.0
        for r in range(n_features):
            weight = (math.factorial(r) * math.factorial(n_features - r - 1)) / fact_n
            for S in combinations(others, r):
                S_set = frozenset(S)
                S_with_j = frozenset(S + (j,))
                phi_j += weight * (subset_to_val[S_with_j] - subset_to_val[S_set])
        shap_values[feature_names[j]] = float(phi_j)

    shap_values["base_logit"] = subset_to_val[frozenset()]
    shap_values["full_logit"] = subset_to_val[frozenset(all_indices)]
    return shap_values
