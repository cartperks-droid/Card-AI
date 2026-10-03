# Training on RunPod

Budget: $35 total. Stop the pod as soon as training ends; a running pod bills every hour.

## Pod
RTX 4090 (24 GB), a PyTorch template (torch >= 2.2), container disk 20 GB, volume at `/workspace` big enough for `data/labels/store/` plus checkpoints. Enable SSH. Note the pod's `HOST` and `PORT` from the Connect tab.

## Setup on the pod
```sh
cd /workspace && git clone https://github.com/cartperks-droid/card-ai.git Card-AI && cd Card-AI
git checkout claude/trusting-davinci-02lh2e
pip install -e . && bash scripts/runpod_setup.sh
```

## Upload from the Mac (run in the repo root on the Mac)
```sh
HOST=...; PORT=...
rsync -avz --progress -e "ssh -p $PORT" data/labels/store/ root@$HOST:/workspace/Card-AI/data/labels/store/
rsync -avz --progress -e "ssh -p $PORT" data/training/    root@$HOST:/workspace/Card-AI/data/training/
```

## Train (on the pod, inside tmux so it survives disconnects)
```sh
tmux new -s train
python -m card_engine.training.train --device cuda --batch-size 512 --weight-decay 0.05 --dropout 0.1 --freeze-language-at 230000 --lr 1e-4 --upset-weight 2 --eval-every 5000
```
Training resumes from the checkpoints in `data/training/`.

## Send new label shards while training runs (on the Mac)
The Mac only generates labels; the pod trains. Rerun this whenever new shards exist. It copies only shards the pod lacks and never deletes anything on the pod. The trainer reloads shards every `--reload-every` steps (default 1000).
```sh
rsync -avz --ignore-existing --progress -e "ssh -p $PORT" data/labels/store/ root@$HOST:/workspace/Card-AI/data/labels/store/
```
To repeat it automatically every 10 minutes: `while true; do <the rsync above>; sleep 600; done`

## Download checkpoints (on the Mac)
```sh
rsync -avz --progress -e "ssh -p $PORT" root@$HOST:/workspace/Card-AI/data/training/ data/training/
```
Do this periodically, then **stop or terminate the pod** (RunPod console, or `runpodctl pod stop <id>`). Stopped pods still charge for volume storage; terminate once the checkpoints are safe on the Mac.
