# Training on RunPod (continuation notes)

State on 2026-10-04 and how the pieces fit. Read this first when picking the run up in a new session.

## Who does what
- **Pod** (RTX 4090, `/workspace/card_engine`, its persistent volume): trains only. The trainer runs in tmux session `train`:
  ```sh
  python -m card_engine.training.train --batch-size 512 --weight-decay 0.05 --dropout 0.1 --freeze-language-at 400000 --device cuda 2>&1 | tee -a data/training/train.out
  ```
  `--device cuda` is required: without it the trainer falls back to MPS or the CPU. It ran at about 25.7 steps/s, 98% GPU, 8 GB VRAM.
- **Mac** (`/Users/xavier.vallee/Documents/New project/card_engine`): makes labels and syncs.
  - Labels, **one** loop only (two runs pick the same shard numbers and waste work):
    ```sh
    caffeinate -i bash -c 'while true; do python3 -m card_engine.training.labels --shards 10000 --rows 2000 --workers 14; done' >> labels.out 2>&1 &
    ```
    Before starting it, check no older run is left: `pgrep -fl card_engine.training.labels`. On 2026-10-04 three runs were found at once (two orphans from earlier tabs).
  - Sync: `bash scripts/pod_sync.sh <ip> <port>` loops by itself every 5 minutes. It sends finished shards up (`data/labels/store`, snapshots, rule declarations) and brings the pod's `log.jsonl` and `*.checkpoint` down to `data/training_pod/`. So the latest model is always on the Mac; nothing else needs extracting.
  - After a code change: `bash scripts/pod_sync.sh <ip> <port> code`. The running trainer keeps its loaded code until restarted.
- The pod reloads the label store every 1,000 steps (about 40 s), so synced shards reach training within about 6 minutes.

## Connecting
`ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`. The IP and port change whenever the pod restarts: copy them from the RunPod Connect tab. Claude's cloud sessions cannot reach the pod (their network blocks raw SSH), so commands run from the Mac.

## Checking progress
On the Mac, from the project folder (reads the log pod_sync brought down):
```sh
python3 scripts/perf.py           # last 6 evaluations; add a number for more
```
- `val_rows` and `upsets` are the validation set and its upset slice. Shards whose seed is divisible by 25 join validation, so new labels change it and the metrics move with the data. A one-evaluation dip when `val_rows` jumps is not a regression.
- On 2026-10-04, step ~1.13M: accuracy 0.969 (baseline 0.903), upsets 0.81, KL 0.044, deterministic 0.988, random-battle error 0.117, train accuracy 0.973. Rows rose from 11.8M to 17.5M once the Mac's labels were synced.

## Stopping the pod on a timer
`scripts/runpod_finish.sh TIME` (on the pod) waits, stops training and labelling cleanly, archives `data/training`, `data/labels` and `data/tablebase` to `/workspace/card-ai-results-<date>.tar.gz`, then runs `runpodctl stop pod`. TIME is in the pod's clock, UTC (`"+3 hours"`, `"15:20"`).
```sh
cd /workspace/card_engine && nohup setsid scripts/runpod_finish.sh "+3 hours" > /workspace/finish.log 2>&1 < /dev/null &
```
- It reads `RUNPOD_POD_ID` from the container's first process, because SSH sessions lack it.
- Stop keeps `/workspace` (storage still bills); terminate deletes it. Terminate only once the checkpoints are on the Mac.
- To see what is scheduled: `ps -eo pid,etime,args | grep -E 'sleep|finish|shutdown|runpodctl' | grep -v grep`. `sleep infinity` is RunPod's own keep-alive.
- On 2026-10-04 a 3-hour timer was set at 12:19 UTC (`/workspace/finish.sh`, a copy of this script written over SSH): the pod stops about 15:19 UTC.

## Using the model on the Mac
pod_sync keeps the pod's checkpoints in `data/training_pod/`. Point the tools at them:
```sh
python3 -m card_engine.training.predict --checkpoint data/training_pod/model.checkpoint \
    --ally "Vampire Lord@GaPl" Set "Good Boy@Pl/Storm" Archer --ally-red Stormcaller@Galaxy \
    --enemy "Immortal Witch" Archer "Good Boy" Set --enemy-blue Fate@Crystal --simulate
python3 -m card_engine.training.generate --checkpoint data/training_pod/model.checkpoint \
    --enemy "Immortal Witch" Archer "Good Boy" Set --pool own
```
Generator masks (borders, mutations, support tiers, minimum rarity) and the custom pool are described in `docs/training.md`.
`best.checkpoint` is the lowest validation KL so far; `model.checkpoint` is the latest. Team syntax is in `card_engine/teams.py`; both tools are described in `docs/training.md`.

## Restarting after the pod was stopped
1. Start the pod in the RunPod console and note the new IP and port.
2. In tmux on the pod: `cd /workspace/card_engine && tmux new -s train`, then the trainer command above. It resumes from `data/training/`.
3. On the Mac: restart `pod_sync.sh` with the new IP and port.
