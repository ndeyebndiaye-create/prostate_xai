#!/usr/bin/env bash
set -Eeuo pipefail

STATUS_FILE="/workspace/TRAINING_STATUS"
rm -f "${STATUS_FILE}"
trap 'printf "failed\n" > "${STATUS_FILE}"' ERR

cd /workspace/prostate_xai
export DATA_ROOT="/workspace/data"
export RUN_OUTPUT_ROOT="/workspace"

python3 -m src.training.train_unimodal_baseline --config configs/phase1_t2w.yaml
python3 -m src.training.train_unimodal_baseline --config configs/phase1_adc.yaml
python3 -m src.training.train_unimodal_baseline --config configs/phase1_hbv.yaml
python3 -m src.training.train_fusion_baseline --config configs/phase3_fusion_simple.yaml

trap - ERR
printf "success\n" > "${STATUS_FILE}"
