#!/usr/bin/env bash
# Push the working tree (code only) to amdserver, where the data and GPU live.
# Usage: scripts/sync_to_server.sh [host] [remote_dir]
set -euo pipefail
HOST=${1:-amdserver}
DEST=${2:-Programming/Robot}
cd "$(dirname "$0")/.."
rsync -a --delete-after \
  --exclude /.git --exclude /data --exclude /runs --exclude /exports \
  --exclude CURRENT_MACHINE.md --exclude requirements.txt \
  --exclude __pycache__ --exclude '*.egg-info' --exclude /user-work/core-papers \
  ./ "$HOST:$DEST/"
echo "synced -> $HOST:$DEST"
