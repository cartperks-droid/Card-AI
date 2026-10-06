"""How well a checkpoint fits the hard examples it trains on, against the held-out ones (memorisation check).

    python3 scripts/hard_fit.py data/training_probe/model.checkpoint data/training_local/model.checkpoint
    python3 scripts/hard_fit.py --kind found data/training_18p/ema.checkpoint     # the annealed search's finds

Reads only the hard_* (or found_*) shards, so it runs beside a trainer. With few shards there may be no held-out ones
yet (seeds divisible by 25); the training rows still say whether the model fits what it has seen. Training KL far below validation KL, with the gap
widening between checkpoints, means the hard rows are being memorised rather than learned.
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the project root holds card_engine
from card_engine.model.checkpoint import load_checkpoint
from card_engine.training.predict import default_device
from card_engine.training.train import Inputs, card_table, evaluate, latest_label_dir, load_split

args = sys.argv[1:]
kind = "hard"
if args[:1] == ["--kind"]:
    kind, args = args[1], args[2:]
store = latest_label_dir()
device = default_device()
with tempfile.TemporaryDirectory() as hard_dir:
    for path in store.glob(f"{kind}_*.npz"):
        (Path(hard_dir) / path.name).symlink_to(path.resolve())
    train_rows, val_rows = load_split(hard_dir, device)
train_rows = {k: v[train_rows["hard"] > 0] for k, v in train_rows.items()} if train_rows is not None else None
val_rows = {k: v[val_rows["hard"] > 0] for k, v in val_rows.items()} if val_rows is not None else None
if train_rows is None:
    raise SystemExit(f"No valid {kind}_* rows in {store}")
wins = train_rows["target"][:, 0] > 0.5
print(f"{kind} rows: {len(wins):,} for training, {int(wins.sum()):,} of them won by the attacker; "
      f"held out: {0 if val_rows is None else len(val_rows['target']):,}")
inputs = Inputs(device)
def won_rows_prediction(model, table, rows):
    """The model's mean attacker win chance on the rows the attacker won (target > 0.5): near the targets if it
    knows these wins, near 0 if it still believes the bigger stats."""
    won = rows["target"][:, 0] > 0.5
    picked = {k: v[won] for k, v in rows.items()}
    out = []
    with torch.no_grad():
        for start in range(0, int(won.sum()), 4096):
            part = {k: v[start:start + 4096] for k, v in picked.items()}
            out.append(model(**inputs(part, table)).float().softmax(-1)[:, 0])
    return (float(torch.cat(out).mean()), float(picked["target"][:, 0].mean())) if out else (float("nan"), float("nan"))


print("checkpoint                                 step       train_kl  train_acc  won: model / engine  val_kl  val_acc  val_upset")
for checkpoint in args or ["data/training_probe/model.checkpoint"]:
    model, metadata = load_checkpoint(checkpoint, map_location=device)
    model.eval()
    with torch.no_grad():
        table = card_table(model, inputs.data.description_tokens)
    t = evaluate(model, inputs, table, train_rows)
    v = evaluate(model, inputs, table, val_rows) if val_rows is not None else {"kl": float("nan"), "accuracy": float("nan"),
                                                                              "upset_accuracy": float("nan")}
    won_model, won_engine = won_rows_prediction(model, table, train_rows)
    print(f"{checkpoint:<42} {metadata.get('step', 0):<10,} {t['kl']:.3f}     {t['accuracy']:.3f}      "
          f"{won_model:.2f} / {won_engine:.2f}          {v['kl']:.3f}   {v['accuracy']:.3f}    {v['upset_accuracy']:.3f}")
