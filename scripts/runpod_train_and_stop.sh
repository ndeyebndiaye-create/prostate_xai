#!/usr/bin/env bash
set -euo pipefail

# Run this controller on the Ubuntu PC. Do not put credentials in the script.
RUNPOD_HOST="YOUR_POD_IP"
RUNPOD_PORT="YOUR_SSH_PORT"
RUNPOD_USER="root"
SSH_KEY="${HOME}/.ssh/id_ed25519"
REMOTE_ROOT="/workspace"
REMOTE_PROJECT="${REMOTE_ROOT}/prostate_xai"
BACKUP_ROOT="./runpod_backup"

if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
  echo "Set RUNPOD_POD_ID before running this script." >&2
  exit 2
fi
command -v ssh >/dev/null
command -v rsync >/dev/null
command -v runpodctl >/dev/null

SSH=(ssh -p "${RUNPOD_PORT}" -i "${SSH_KEY}" -o BatchMode=yes)

sync_outputs() {
  mkdir -p "${BACKUP_ROOT}"/{checkpoints,logs,results}
  for output in checkpoints logs results; do
    if "${SSH[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" \
      "test -d '${REMOTE_ROOT}/${output}'"; then
      rsync -avz --partial --append-verify \
        -e "ssh -p ${RUNPOD_PORT} -i ${SSH_KEY} -o BatchMode=yes" \
        "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_ROOT}/${output}/" \
        "${BACKUP_ROOT}/${output}/"
    fi
  done
}

"${SSH[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" bash -s <<'REMOTE'
set -euo pipefail
cd /workspace/prostate_xai
command -v tmux >/dev/null
if tmux has-session -t prostate-train 2>/dev/null; then
  echo "tmux session prostate-train already exists." >&2
  exit 1
fi
rm -f /workspace/TRAINING_STATUS
tmux new-session -d -s prostate-train \
  'cd /workspace/prostate_xai && bash scripts/runpod_train.sh'
REMOTE

echo "Training launched in tmux. Syncing outputs every 5 minutes."
while true; do
  sync_outputs
  status=$("${SSH[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" \
    "cat '${REMOTE_ROOT}/TRAINING_STATUS' 2>/dev/null || true")
  case "${status}" in
    success)
      break
      ;;
    failed)
      echo "Training failed. Pod left running; inspect tmux and logs." >&2
      exit 1
      ;;
  esac
  sleep 300
done

# Pull once more after training has exited, then verify a checkpoint before stop.
sync_outputs
best_checkpoint="$(find "${BACKUP_ROOT}/checkpoints" -type f -name best.pt -print -quit)"
if [[ -z "${best_checkpoint}" ]]; then
  echo "Training succeeded, but no best.pt was found in the local backup; Pod left running." >&2
  exit 1
fi
python3 scripts/check_backup.py "${best_checkpoint}"
runpodctl stop pod "${RUNPOD_POD_ID}"
