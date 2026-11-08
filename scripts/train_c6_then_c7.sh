#!/usr/bin/env bash
set -eo pipefail

echo "================================================================="
echo "[pipeline] Étape 1 : Attente de la fin de synchronisation rclone"
echo "================================================================="

# Wait until any running rclone copy process finishes
while pgrep -f "rclone copy" > /dev/null; do
    COUNT=$(find /workspace/data/picai -type f 2>/dev/null | wc -l || echo 0)
    SIZE=$(du -sh /workspace/data/picai 2>/dev/null | cut -f1 || echo "0")
    echo "[rclone] Téléchargement en cours : $COUNT fichiers (~$SIZE)..."
    sleep 10
done

echo "[pipeline] Synchronisation terminée !"
du -sh /workspace/data/picai

# Vérifier la présence du marksheet et compter les patients complets
python3 -c "
from src.config import DATA_ROOT
from src.data.picai_paths import find_complete_patients, load_marksheet
c, inc = find_complete_patients(DATA_ROOT / 'picai')
print(f'Cohorte locale : {len(c)} patients complets, {len(inc)} incomplets.')
assert len(c) >= 1000, f'Erreur : seulement {len(c)} patients complets trouvés !'
"

export TRITON_DEFAULT_FP_FUSION=1

echo "================================================================="
echo "[pipeline] Étape 2 : Lancement de l'Entraînement C6 (Self-Attention)"
echo "================================================================="
python3 -m src.training.train_cross_attention --config configs/phase5_c6_attention_clinical.yaml

echo "================================================================="
echo "[pipeline] Étape 3 : Lancement de l'Entraînement C7 (Cross-Attention + Gating)"
echo "================================================================="
python3 -m src.training.train_cross_attention --config configs/phase5_c7_cross_attention_clinical.yaml

echo "================================================================="
echo "[pipeline] Étape 4 : Calcul de la Calibration et des Métriques"
echo "================================================================="
python3 -m scripts.evaluate_calibration_and_metrics

echo "================================================================="
echo "[pipeline] Étape 5 : Mise à Jour des Graphiques et Courbes"
echo "================================================================="
python3 -m scripts.visualize_training_results

echo "================================================================="
echo "[pipeline] Entraînements C6 et C7 terminés avec succès !"
echo "================================================================="
