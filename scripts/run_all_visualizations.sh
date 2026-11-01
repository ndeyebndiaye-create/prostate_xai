#!/usr/bin/env bash
# Exécute l'ensemble des scripts de visualisation pour le mémoire
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

echo "=========================================================="
echo "   GÉNÉRATION COMPLÈTE DES FIGURES ET GRAPHES (XAI)"
echo "=========================================================="

echo ""
echo "[1/3] Génération des graphiques EDA (Distributions & Corrélations)..."
python3 scripts/visualize_eda.py

echo ""
echo "[2/3] Génération des coupes IRM multiparamétriques & masques..."
python3 scripts/visualize_mri_slices.py

echo ""
echo "[3/3] Génération des graphiques de résultats d'entraînement..."
python3 scripts/visualize_training_results.py

echo ""
echo "=========================================================="
echo " SUCCÈS : Toutes les figures sont dans docs/figures/ :"
ls -lh docs/figures/*.png
echo "=========================================================="
