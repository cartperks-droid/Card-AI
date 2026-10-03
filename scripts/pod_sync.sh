#!/usr/bin/env bash
# Train on a RunPod pod while the Mac makes labels (user, 2026-10-03). Run from the project folder on the Mac.
#   bash scripts/pod_sync.sh <ip> <ssh port> code   # once: copy the code (no labels, runs, videos or engine checkout)
#   bash scripts/pod_sync.sh <ip> <ssh port>        # loop: label shards up, the run's log and checkpoints down
# The pod keeps everything under /workspace/card_engine (its persistent volume, which refuses chown, hence
# --no-owner --no-group); the run comes back to
# data/training_pod/ so it never mixes with a local run.
set -euo pipefail
ip=$1 port=$2
remote=root@$ip:/workspace/card_engine
ssh_cmd="ssh -p $port -o StrictHostKeyChecking=accept-new"
if [ "${3:-}" = code ]; then
  ls CLAUDE.md >/dev/null  # refuse to copy anything but the project folder
  rsync -az --no-owner --no-group -e "$ssh_cmd" --exclude .git --exclude .venv --exclude __pycache__ --exclude 'data/labels/' \
    --exclude 'data/training*' --exclude data/tablebase --exclude 'sim_js/vendor' --exclude 'sim_js/node_modules' \
    --exclude '*.MOV' --exclude '*.mov' --exclude '*.mp4' --exclude '*.checkpoint' ./ "$remote/"
  echo "code copied to $remote"
  exit 0
fi
mkdir -p data/training_pod
while true; do
  # New-engine shards only (old ones were moved to data/labels/old_engine); partial files are still being written.
  rsync -az --no-owner --no-group -e "$ssh_cmd" --exclude 'partial_*' --include 'store/***' --include 'snapshots/***' \
    --include rule_changes.json --exclude '*' data/labels/ "$remote/data/labels/" || true
  rsync -az --no-owner --no-group -e "$ssh_cmd" "$remote/data/training/log.jsonl" "$remote/data/training/*.checkpoint" data/training_pod/ 2>/dev/null || true
  tail -1 data/training_pod/log.jsonl 2>/dev/null || true
  sleep 300
done
