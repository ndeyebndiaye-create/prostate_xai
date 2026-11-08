"""
Step 7 — Explainable AI (XAI) & Uncertainty Quantification Package (Chapter 4, Sections 4.6 & 4.7):
- `gradcam3d.py`: Per-modality 3D Grad-CAM++ (`Heatmap_T2W`, `Heatmap_ADC`, `Heatmap_HBV`),
  Pointing Accuracy against PI-CAI lesion masks, and Occlusion Fidelity drop.
- `shap_clinical.py`: Exact Shapley values (`phi_j`) on the positive-class logit for the
  4 structured clinical variables `[patient_age, log_psa, psad, prostate_volume]`.
- `mc_dropout.py`: Monte Carlo Dropout stochastic inference (10, 25, 50 passes) +
  selective abstention rule for high-uncertainty cases.
"""
