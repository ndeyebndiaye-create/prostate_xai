"""
Proposed Multi-Modal 3D SegResNet + Cross-Attention & Modality Gating Architecture
(Chapter 4, Sections 4.1 to 4.5, and Table 3.1 Configurations C5, C6, C7).

Supports:
- Step 4 (Image-only dedicated 3D SegResNet encoders + Cross-Attention + Gating,
  `use_clinical=False`)
- Step 5 (Image + Clinical MLP + Bidirectional Cross-Attention + Gating,
  `use_clinical=True`, Configurations C5, C6, C7 in Table 3.1)
- Ablation flags (Section 3.6.3 & 4.2.1):
  - `shared_weights=False` (Model-U, default) vs `shared_weights=True` (Model-S)
  - `fusion_mode="cross_attention"` (C7) vs `"attention"` (C6) vs `"concat"` (C5)
  - `use_gating=True` vs `False`
  - `availability_mask` for Modality Dropout (Section 3.4.4)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.models.segresnet_encoder import ClinicalEncoderMLP, SegResNetEncoder3D

FusionMode = Literal["cross_attention", "attention", "concat"]


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between image modality tokens and a context
    token (clinical token when `use_clinical=True`, or cross-modality context
    when `use_clinical=False`), as formulated in Chapter 4, Section 4.4.3:

        T_img^new  = LN(T_img  + Drop(MHA(Q=T_img,  K=t_ctx, V=t_ctx)))
        t_ctx^new  = LN(t_ctx  + Drop(MHA(Q=t_ctx,  K=T_img, V=T_img)))
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.img_to_ctx = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ctx_to_img = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_ctx = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        img_tokens: torch.Tensor,
        ctx_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        img_tokens : torch.Tensor of shape `[B, M, D]`
        ctx_tokens : torch.Tensor of shape `[B, K_ctx, D]`
        """
        delta_img, _ = self.img_to_ctx(query=img_tokens, key=ctx_tokens, value=ctx_tokens)
        new_img = self.norm_img(img_tokens + self.drop(delta_img))

        delta_ctx, _ = self.ctx_to_img(query=ctx_tokens, key=img_tokens, value=img_tokens)
        new_ctx = self.norm_ctx(ctx_tokens + self.drop(delta_ctx))
        return new_img, new_ctx


class LightweightTransformerBlock(nn.Module):
    """
    Single Transformer encoder layer following Chapter 4, Section 4.4.4:
        tokens -> Self-Attention (4 heads) -> Residual + LayerNorm
               -> FFN (128 -> 256 -> 128)  -> Residual + LayerNorm
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            query=tokens,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
        )
        tokens = self.norm1(tokens + self.drop(attn_out))
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + self.drop(ffn_out))
        return tokens


class ModalityGating(nn.Module):
    """
    Conditional Modality Gating mechanism (Chapter 4, Section 4.4.5):
    Computes adaptive importance weights over modality tokens and supports
    explicit masking of missing modalities (`availability_mask`, Section 3.4.4).
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        modality_tokens: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        modality_tokens : `[B, K, D]` (excludes `[CLS]`)
        availability_mask : optional `[B, K]` in `{0, 1}` (1 = available, 0 = dropped)

        Returns
        -------
        gated_summary : `[B, D]`
        gate_weights : `[B, K]`
        """
        logits = self.scorer(modality_tokens).squeeze(-1)  # [B, K]
        if availability_mask is not None:
            mask = availability_mask.to(dtype=torch.bool, device=logits.device)
            # Guard against all-zero mask row
            all_zero = ~mask.any(dim=-1, keepdim=True)
            safe_mask = mask | all_zero
            logits = logits.masked_fill(~safe_mask, -1e4)

        gate_weights = torch.softmax(logits, dim=-1)  # [B, K]
        gated_summary = torch.sum(modality_tokens * gate_weights.unsqueeze(-1), dim=1)
        return gated_summary, gate_weights


class MultimodalCrossAttentionNet(nn.Module):
    """
    Full Multi-Modal 3D SegResNet + Cross-Attention Architecture (Chapter 4).

    Parameters
    ----------
    modalities : list[str]
        Active image modalities, default `["t2w", "adc", "hbv"]`.
    init_filters : int
        Initial filters for each 3D SegResNet encoder (16 or 32, Section 4.2).
    blocks_down : tuple[int, ...]
        Residual blocks per downsampling stage, default `(1, 2, 2, 4)`.
    encoder_dim : int
        Output dimension of each 3D SegResNet encoder before token projection (256).
    fusion_dim : int
        Common token dimension `d` (128, Section 4.4.2 & 4.4.4).
    num_heads : int
        Number of attention heads (4, Section 4.4.4).
    num_layers : int
        Number of lightweight Transformer blocks (2, Section 4.4.4).
    ffn_dim : int
        Hidden dimension of Transformer FFN (256, Section 4.4.4).
    dropout : float
        Dropout rate (0.15, within the 0.10–0.20 range of Section 4.4.4).
    shared_weights : bool
        If False (default), builds **Model-U** (3 dedicated encoders).
        If True, builds **Model-S** (1 shared encoder across modalities, Section 4.2.1).
    use_clinical : bool
        If True, enables the `ClinicalEncoderMLP` branch (Section 4.3).
    clinical_dim : int
        Number of clinical features (4: `[patient_age, log(PSA), PSAD, prostate_volume]`).
    fusion_mode : {"cross_attention", "attention", "concat"}
        - `"cross_attention"`: bidirectional cross-attention + Transformer + gating (C7)
        - `"attention"`: self-attention Transformer + gating without cross-attn step (C6)
        - `"concat"`: late concatenation of modality/clinical embeddings (C5)
    use_gating : bool
        Whether to apply `ModalityGating` (Section 4.4.5).
    n_classes : int
        Number of output logits (2 for CrossEntropyLoss compatibility).
    """

    def __init__(
        self,
        modalities: list[str] | None = None,
        init_filters: int = 16,
        blocks_down: tuple[int, ...] = (1, 2, 2, 4),
        encoder_dim: int = 256,
        fusion_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.15,
        shared_weights: bool = False,
        use_clinical: bool = False,
        clinical_dim: int = 4,
        fusion_mode: FusionMode = "cross_attention",
        use_gating: bool = True,
        n_classes: int = 2,
    ):
        super().__init__()
        if modalities is None:
            modalities = ["t2w", "adc", "hbv"]
        self.modalities = list(modalities)
        self.n_modalities = len(self.modalities)
        self.shared_weights = shared_weights
        self.use_clinical = use_clinical
        self.fusion_mode = fusion_mode
        self.use_gating = use_gating
        self.fusion_dim = fusion_dim

        # ── 4.2 : 3D SegResNet Encoders (Model-U vs Model-S) ─────────────────
        if shared_weights:
            shared_enc = SegResNetEncoder3D(
                in_channels=1,
                init_filters=init_filters,
                blocks_down=blocks_down,
                out_dim=encoder_dim,
                dropout=dropout,
            )
            self.shared_encoder = shared_enc
            self.encoders = nn.ModuleDict({m: shared_enc for m in self.modalities})
        else:
            self.shared_encoder = None
            self.encoders = nn.ModuleDict(
                {
                    m: SegResNetEncoder3D(
                        in_channels=1,
                        init_filters=init_filters,
                        blocks_down=blocks_down,
                        out_dim=encoder_dim,
                        dropout=dropout,
                    )
                    for m in self.modalities
                }
            )

        # Projection of each image branch from encoder_dim (256) -> fusion_dim (128)
        self.img_projections = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(encoder_dim, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.GELU(),
                )
                for m in self.modalities
            }
        )

        # ── 4.3 : Clinical Encoder MLP (optional, enabled in Step 5) ─────────
        if use_clinical:
            self.clinical_encoder = ClinicalEncoderMLP(
                in_features=clinical_dim, hidden_dim=64, dropout=dropout
            )
            self.clinical_proj = nn.Sequential(
                nn.Linear(64, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )
        else:
            self.clinical_encoder = None
            self.clinical_proj = None

        n_tokens_without_cls = self.n_modalities + (1 if use_clinical else 0)

        # Learnable [CLS] token and modality-type embeddings (Section 4.4.2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.token_type_embeddings = nn.Parameter(
            torch.zeros(1, 1 + n_tokens_without_cls, fusion_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.token_type_embeddings, std=0.02)

        # ── 4.4.3 : Bidirectional Cross-Attention ────────────────────────────
        if fusion_mode == "cross_attention":
            self.cross_attn = BidirectionalCrossAttention(
                embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout
            )
        else:
            self.cross_attn = None

        # ── 4.4.4 : Lightweight Transformer Fusion Blocks ────────────────────
        if fusion_mode in ("cross_attention", "attention"):
            self.transformer_blocks = nn.ModuleList(
                [
                    LightweightTransformerBlock(
                        embed_dim=fusion_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList()

        # Late concatenation projection for Config C5 (`fusion_mode="concat"`)
        if fusion_mode == "concat":
            self.concat_proj = nn.Sequential(
                nn.Linear(n_tokens_without_cls * fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.concat_proj = None

        # ── 4.4.5 : Modality Gating ──────────────────────────────────────────
        self.gating = ModalityGating(embed_dim=fusion_dim) if use_gating else None

        # ── 4.5 : Classification Head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes),
        )

        # Stored on each forward pass for XAI / analysis
        self.last_gate_weights: torch.Tensor | None = None
        self.last_feature_maps: dict[str, torch.Tensor] = {}

    def encode_modalities(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Encode stacked multimodal tensor `x` of shape `[B, C, D, H, W]` where
        `C == len(self.modalities)` into projected tokens `[B, fusion_dim]`.
        """
        if x.shape[1] != self.n_modalities:
            raise ValueError(
                f"Expected {self.n_modalities} input channels for modalities "
                f"{self.modalities}, got shape {tuple(x.shape)}"
            )

        tokens: list[torch.Tensor] = []
        self.last_feature_maps = {}
        for idx, modality in enumerate(self.modalities):
            x_mod = x[:, idx : idx + 1, ...]  # [B, 1, D, H, W]
            enc = self.encoders[modality]
            h_mod = enc(x_mod)  # [B, encoder_dim]
            if enc.last_feature_map is not None:
                self.last_feature_maps[modality] = enc.last_feature_map
            t_mod = self.img_projections[modality](h_mod)  # [B, fusion_dim]
            tokens.append(t_mod)
        return tokens

    def forward(
        self,
        x: torch.Tensor,
        clin: torch.Tensor | None = None,
        availability_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape `[B, C, D, H, W]`
        clin : optional torch.Tensor of shape `[B, 4]` (required when `use_clinical=True`)
        availability_mask : optional torch.Tensor of shape `[B, K]` (`1` available, `0` dropped)
        """
        batch_size = x.shape[0]
        img_token_list = self.encode_modalities(x)
        img_tokens = torch.stack(img_token_list, dim=1)  # [B, M, D]

        if self.use_clinical:
            if clin is None:
                raise ValueError("`clin` tensor `[B, 4]` is required when use_clinical=True")
            h_clin = self.clinical_encoder(clin)  # [B, 64]
            t_clin = self.clinical_proj(h_clin).unsqueeze(1)  # [B, 1, D]
        else:
            t_clin = None

        # Apply availability mask zeroing on input tokens if provided (Section 3.4.4)
        if availability_mask is not None:
            mask_f = availability_mask.to(dtype=img_tokens.dtype, device=img_tokens.device)
            img_mask = mask_f[:, : self.n_modalities].unsqueeze(-1)
            img_tokens = img_tokens * img_mask
            if t_clin is not None and mask_f.shape[1] > self.n_modalities:
                clin_mask = mask_f[:, self.n_modalities : self.n_modalities + 1].unsqueeze(-1)
                t_clin = t_clin * clin_mask

        # ── Case 1 : Late Concatenation (Config C5) ──────────────────────────
        if self.fusion_mode == "concat":
            all_mod_tokens = (
                torch.cat([img_tokens, t_clin], dim=1) if t_clin is not None else img_tokens
            )
            flat = all_mod_tokens.flatten(1)
            fused = self.concat_proj(flat)
            if self.gating is not None:
                gated, weights = self.gating(all_mod_tokens, availability_mask)
                self.last_gate_weights = weights
                fused = fused + gated
            return self.classifier(fused)

        # ── Case 2 : Cross-Attention (C7) or Self-Attention (C6) ─────────────
        if self.fusion_mode == "cross_attention" and self.cross_attn is not None:
            if t_clin is not None:
                # Section 4.4.3 : Bidirectional Image <-> Clinical Cross-Attention
                img_tokens, t_clin = self.cross_attn(img_tokens, t_clin)
            elif self.n_modalities > 1:
                # Step 4 (image-only): T2W anatomical anchor <-> Diffusion (ADC, HBV)
                t2w_tok = img_tokens[:, :1, :]
                diff_toks = img_tokens[:, 1:, :]
                diff_toks, t2w_tok = self.cross_attn(diff_toks, t2w_tok)
                img_tokens = torch.cat([t2w_tok, diff_toks], dim=1)

        modality_tokens = (
            torch.cat([img_tokens, t_clin], dim=1) if t_clin is not None else img_tokens
        )  # [B, K, D]

        cls_tok = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        seq = torch.cat([cls_tok, modality_tokens], dim=1)  # [B, 1+K, D]
        seq = seq + self.token_type_embeddings[:, : seq.shape[1], :]

        for block in self.transformer_blocks:
            seq = block(seq)

        cls_out = seq[:, 0, :]  # [B, D]
        refined_mod_tokens = seq[:, 1:, :]  # [B, K, D]

        if self.gating is not None:
            gated_summary, gate_weights = self.gating(refined_mod_tokens, availability_mask)
            self.last_gate_weights = gate_weights
            final_repr = cls_out + gated_summary
        else:
            self.last_gate_weights = None
            final_repr = cls_out

        return self.classifier(final_repr)
