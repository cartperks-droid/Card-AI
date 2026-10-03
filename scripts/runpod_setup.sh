#!/usr/bin/env bash
# One-time setup on a RunPod pod (PyTorch template). Run from the repo root.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq build-essential
pip install -q numpy
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA"; print(torch.cuda.get_device_name(0))'
python -c 'from card_engine.simulator.native import build_library; print(build_library())'
mkdir -p data/labels/store data/training
