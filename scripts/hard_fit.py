"""How well a checkpoint fits the hard examples it trains on, against the held-out ones (memorisation check).

    python3 scripts/hard_fit.py data/training_probe/model.checkpoint data/training_local/model.checkpoint

Reads only the hard_* shards, so it runs beside a trainer. Training KL far below validation KL, with the gap
widening between checkpoints, means the hard rows are being memorised rather than learned.
"""

import sys
import tempfile
from pathlib import Path

import torch

from card_engine.model.checkpoint import load_checkpoint
from card_engine.training.predict import default_device
from card_engine.training.train import Inputs, card_table, evaluate, latest_label_dir, load_split

store = latest_label_dir()
device = default_device()
with tempfile.TemporaryDirectory() as hard_dir:
    for path in store.glob("hard_*.npz"):
        (Path(hard_dir) / path.name).symlink_to(path.resolve())
    train_rows, val_rows = load_split(hard_dir, device)
train_rows = {k: v[train_rows["hard"] > 0] for k, v in train_rows.items()}
val_rows = {k: v[val_rows["hard"] > 0] for k, v in val_rows.items()}
inputs = Inputs(device)
print("checkpoint                                 step       train_rows  train_kl  train_acc  val_rows  val_kl  val_acc  val_upset")
for checkpoint in sys.argv[1:] or ["data/training_probe/model.checkpoint"]:
    model, metadata = load_checkpoint(checkpoint, map_location=device)
    model.eval()
    with torch.no_grad():
        table = card_table(model, inputs.data.description_tokens)
    t, v = evaluate(model, inputs, table, train_rows), evaluate(model, inputs, table, val_rows)
    print(f"{checkpoint:<42} {metadata.get('step', 0):<10,} {len(train_rows['target']):<11,} {t['kl']:.3f}     "
          f"{t['accuracy']:.3f}      {len(val_rows['target']):<9,} {v['kl']:.3f}   {v['accuracy']:.3f}    {v['upset_accuracy']:.3f}")
