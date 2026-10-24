#!/usr/bin/env bash
set -euo pipefail

# Edit these connection details; keep SSH keys and tokens out of this file.
RUNPOD_HOST="YOUR_POD_IP"
RUNPOD_PORT="YOUR_SSH_PORT"
RUNPOD_USER="root"
SSH_KEY="${HOME}/.ssh/id_ed25519"
REMOTE_ROOT="/workspace"
BACKUP_ROOT="./runpod_backup"

sync_outputs() {
  mkdir -p "${BACKUP_ROOT}"/{checkpoints,logs,results}
  local ssh_cmd
  ssh_cmd="ssh -p ${RUNPOD_PORT} -i ${SSH_KEY} -o BatchMode=yes"
  for output in checkpoints logs results; do
    rsync -avz --partial --append-verify \
      -e "${ssh_cmd}" \
      "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_ROOT}/${output}/" \
      "${BACKUP_ROOT}/${output}/"
  done
}

if [[ "${1:-}" == "--watch" ]]; then
  while true; do
    if ! sync_outputs; then
      echo "Sync failed; will retry in 5 minutes." >&2
    fi
    sleep 300
  done
else
  sync_outputs
fi
