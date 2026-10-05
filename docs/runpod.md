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
    Fixed-stat battles (`--fixed`, see `docs/training.md`) run as their own loop, since they number their shards separately. The pod's trainer reads them only once it runs code from 2026-10-04 or later.
    Before starting it, check no older run is left: `pgrep -fl card_engine.training.labels`. On 2026-10-04 three runs were found at once (two orphans from earlier tabs).
  - Sync: `bash scripts/pod_sync.sh <ip> <port>` loops by itself every 5 minutes. It sends finished shards up (`data/labels/store`, snapshots, rule declarations) and brings the pod's `log.jsonl` and `*.checkpoint` down to `data/training_pod/`. So the latest model is always on the Mac; nothing else needs extracting.
  - After a code change: `bash scripts/pod_sync.sh <ip> <port> code`. The running trainer keeps its loaded code until restarted.
- The pod reloads the label store every 1,000 steps (about 40 s), so synced shards reach training within about 6 minutes.

## Loading labels on the pod
The pod's `/workspace` disk is slow per file: reading 76,657 shard files took about 38 minutes (2026-10-05) whatever the CPU did, against 5 minutes on the Mac. Start the pod's trainer with `--pack-labels`: the first load also writes the checked rows into one large file in `data/labels/packs/`, and every later restart reads that file plus only the shards that are newer. Packs written under older rules are deleted automatically.

## Labelling on the pod too
The pod's CPUs sit nearly idle while its GPU trains, and its engine is built (`sim_js/setup.sh`), so it labels as well (2026-10-05). Its shards are numbered from their own ranges, `--first-seed 50000000` for battles and fixed-stat battles and `--first-seed 5000000` for hard examples, so they never share a name with the Mac's (the Mac numbers on from its own highest seed, and `pod_sync.sh` only sends shards up). Size the workers to the pod's CPU allowance, not `nproc`: this pod reports 128 cores but may use 13.6 (`/sys/fs/cgroup/cpu/cpu.cfs_quota_us` ÷ `cpu.cfs_period_us`), and past it every process is paused, the trainer too, whatever its `nice` (2026-10-05: 120 workers dropped the trainer from 6.7 to 3.3 steps/s). Three loops of 3 workers, under `nice -n 19`, leave the trainer about 4 CPUs. Its hard examples are mined against the deep run's weight average (`data/training_deep/ema.checkpoint`), the model training there. The Mac's local run does not see the pod's shards.

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

## When the stopped pod's GPU is gone
A stopped pod's volume stays on its machine. If that machine's GPU is taken ("Your Pod's GPUs are no longer available"), the volume is out of reach until it frees up (2026-10-04). Create pods with a **Network Volume** to avoid this: it attaches to any pod.

To move on without the old volume, use the Mac's copy of the model (`data/training_pod/`, at most one sync behind). Only the optimiser state stays behind, and `--init-from` keeps the weights and step and warms a fresh optimiser up over 1,000 steps.
1. New pod (RTX 4090, PyTorch template, SSH). From the Mac:
   ```sh
   bash scripts/pod_sync.sh NEW_IP NEW_PORT code
   ssh -p NEW_PORT root@NEW_IP 'mkdir -p /workspace/card_engine/data/training'
   scp -P NEW_PORT data/training_pod/model.checkpoint data/training_pod/best.checkpoint data/training_pod/log.jsonl root@NEW_IP:/workspace/card_engine/data/training/
   ```
2. On the pod, build DaddyDrago's engine (the trainer reads his card data; the code sync leaves the engine checkout out):
   ```sh
   cd /workspace/card_engine
   curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs build-essential
   bash sim_js/setup.sh
   ```
3. From the Mac: `bash scripts/pod_sync.sh NEW_IP NEW_PORT` (the first pass uploads every shard).
4. On the pod, in tmux: the trainer command above plus `--init-from data/training/model.checkpoint`.
5. Terminate the old pod once the new one trains: a stopped pod still bills for its volume.
