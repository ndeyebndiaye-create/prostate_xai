#!/usr/bin/env bash
# ==============================================================================
# Script Maître d'Exécution : Entraînement C7 Haute Résolution & XAI Nette
# Projet : Prostate XAI — Mémoire AIMS Sénégal
# ==============================================================================

set -eo pipefail

echo "================================================================="
echo " [Étape 1/4] Entraînement C7 Haute Résolution (128x128x20)"
echo " FOV anatomique étendu : 64 mm x 64 mm x 60 mm (Prostate complète)"
echo "================================================================="

export TRITON_DEFAULT_FP_FUSION=1

# Lancer l'entraînement 5-fold CV
python3 -m src.training.train_cross_attention \
    --config configs/phase5_c7_cross_attention_clinical_hires.yaml

echo "================================================================="
echo " [Étape 2/4] Calcul des Métriques Cliniques & Calibration (Tableau 5.1)"
echo "================================================================="

python3 -m scripts.evaluate_calibration_and_metrics

echo "================================================================="
echo " [Étape 3/4] Génération des Visuels XAI Grad-CAM++ Nettes & Contrastées"
echo " (Masquage transparent < 0.20 + Seuil 70% + Contours de lésion)"
echo "================================================================="

python3 -m scripts.run_xai_and_uncertainty \
    --experiment phase5_c7_cross_attention_clinical_hires \
    --use_clinical

echo "================================================================="
echo " [Étape 4/4] Régénération des Figures d'Entraînement & Comparaison"
echo " (training_all_models_comparison.png, calibration_tradeoff, learning_curves)"
echo "================================================================="

python3 -m scripts.visualize_training_results

echo "================================================================="
echo " PIPELINE TERMINÉ AVEC SUCCÈS !"
echo " Les figures haute définition sont prêtes et à jour dans :"
echo "   - docs/figures/"
echo "   - docs/xai_and_robustness_visuals/"
echo "================================================================="
