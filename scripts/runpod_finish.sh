#!/usr/bin/env bash
# On the pod: at a set time, end training and labelling, archive the run to the volume, then stop the pod
# (stop, not terminate, so /workspace survives). The time is anything `date -d` reads, in the pod's clock (UTC):
#   nohup setsid scripts/runpod_finish.sh "+3 hours" > /workspace/finish.log 2>&1 < /dev/null &
# Checkpoints and shards are written atomically: at worst the steps since the last 1,000-step checkpoint are lost.
set -euo pipefail
at=${1:?usage: runpod_finish.sh TIME [archive_dir]}
out=${2:-/workspace}
repo=$(cd "$(dirname "$0")/.." && pwd)
export $(tr '\0' '\n' < /proc/1/environ | grep ^RUNPOD_ | xargs)  # SSH sessions lack the pod's own variables

target=$(date -d "$at" +%s)
(( target > $(date +%s) )) || target=$(date -d "tomorrow $at" +%s)
echo "pod ${RUNPOD_POD_ID:-?}: finishing at $(date -d "@$target") ($(( (target - $(date +%s)) / 60 )) min)"
sleep $(( target - $(date +%s) ))

pkill -INT -f 'card_engine.training.(train|labels|hard)' || true
for _ in $(seq 60); do pgrep -f 'card_engine.training.(train|labels|hard)' > /dev/null || break; sleep 5; done
pkill -KILL -f 'card_engine.training.(train|labels|hard)' || true

archive="$out/card-ai-results-$(date +%Y%m%d-%H%M).tar.gz"
# The runs (trainer.pt included) and the tablebase; label shards go to the Mac through pod_sync.sh, and a copy here
# would not fit the 30 GB volume.
tar -C "$repo" -czf "$archive" $(cd "$repo" && ls -d data/training* data/tablebase 2>/dev/null)
echo "archived $(du -h "$archive" | cut -f1) to $archive"
sync
runpodctl stop pod "$RUNPOD_POD_ID"
