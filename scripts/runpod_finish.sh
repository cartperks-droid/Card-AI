#!/usr/bin/env bash
# Stop a RunPod training run at a set local time: end training/labelling, archive the results to the
# persistent volume, then stop the pod (stop, not terminate, so /workspace survives).
#
#   TZ=America/New_York nohup scripts/runpod_finish.sh 11:00 > /workspace/finish.log 2>&1 &
#
# Checkpoints and shards are written atomically, so at worst the steps since the last 1,000-step checkpoint are lost.
set -euo pipefail
at=${1:?usage: runpod_finish.sh HH:MM [archive_dir]}
out=${2:-/workspace}
repo=$(cd "$(dirname "$0")/.." && pwd)

target=$(date -d "$at" +%s)
(( target > $(date +%s) )) || target=$(date -d "tomorrow $at" +%s)
echo "pod ${RUNPOD_POD_ID:-?}: finishing at $(date -d "@$target") ($(( (target - $(date +%s)) / 60 )) min)"
sleep $(( target - $(date +%s) ))

pkill -INT -f 'card_engine.training.(train|labels)' || true
for _ in $(seq 60); do pgrep -f 'card_engine.training.(train|labels)' > /dev/null || break; sleep 5; done
pkill -KILL -f 'card_engine.training.(train|labels)' || true

archive="$out/card-ai-results-$(date +%Y%m%d-%H%M).tar.gz"
tar -C "$repo" -czf "$archive" $(cd "$repo" && ls -d data/training data/labels data/tablebase 2>/dev/null)
echo "archived $(du -h "$archive" | cut -f1) to $archive"
sync

runpodctl stop pod "$RUNPOD_POD_ID"
