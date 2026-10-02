"""Train the general win predictor on simulator labels (side A initiates), resumable.

Each step encodes every card description once (289 sequences) and gathers the slot vectors,
so the description encoder trains end to end at a fraction of the per-slot cost. Targets are
the outcome probabilities (exact tablebase results or Monte Carlo frequencies), renormalised
over finished outcomes. The newest label
directory (rule fingerprint) is used; when rules change, training continues on the new labels.
Checkpoints hold the model (model.checkpoint) plus optimizer/step state for exact resumption.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..model import BattleModel, load_model_data
from ..model.checkpoint import load_checkpoint, save_checkpoint
from .labels import FIELDS, SHARD_DIR

RUN_DIR = Path(__file__).resolve().parents[2] / "data" / "training"
VALIDATION_EVERY = 25  # shard seeds divisible by this are held out


def latest_label_dir(root=SHARD_DIR, min_shards=100):
    """The newest rules' labels once they have `min_shards` shards (else the newest that does, else the largest)."""
    dirs = [d for d in Path(root).iterdir() if d.is_dir() and any(d.glob("shard_*.npz"))]
    if not dirs:
        raise FileNotFoundError(f"No label shards under {root}")
    newest = lambda d: max(p.stat().st_mtime for p in d.glob("shard_*.npz"))
    ready = [d for d in dirs if len(list(d.glob("shard_*.npz"))) >= min_shards]
    return max(ready, key=newest) if ready else max(dirs, key=lambda d: len(list(d.glob("shard_*.npz"))))


_STAT_FAVOURITE = {}


def stat_favourite(path, catalog=None):
    """Per row of a shard: the side (0/1) with more sqrt(HP x ATK) after supports, the naive baseline."""
    key = (str(path), Path(path).stat().st_mtime)
    if key not in _STAT_FAVOURITE:
        from ..catalog import load_catalog
        from .labels import compile_spec
        catalog = catalog or load_catalog()
        shard = np.load(path)
        favourite = []
        for i in range(len(shard["probs"])):
            battle = compile_spec(catalog, {k: shard[k][i].tolist() for k in FIELDS})
            strength = [sum((f.hp * f.attack) ** .5 for f in battle.teams[side][:4]) for side in (0, 1)]
            favourite.append(0 if strength[0] >= strength[1] else 1)
        _STAT_FAVOURITE[key] = np.array(favourite)
    return _STAT_FAVOURITE[key]


def load_split(directory, device):
    """(train, validation) tensors; validation = shards whose seed is divisible by VALIDATION_EVERY."""
    split = {"train": [], "val": []}
    favourites = []
    for path in sorted(Path(directory).glob("shard_*.npz")):
        seed = int(path.stem.split("_")[1])
        validation = seed % VALIDATION_EVERY == 0
        split["val" if validation else "train"].append(np.load(path))
        if validation:
            favourites.append(stat_favourite(path))
    out = {}
    for name, parts in split.items():
        if not parts:
            out[name] = None
            continue
        arrays = {key: np.concatenate([p[key] for p in parts]) for key in (*FIELDS, "probs")}
        finished = arrays["probs"][:, :2].sum(1)  # A win, B win (ties cannot happen; unfinished mass is dropped)
        keep = finished > 0
        tensors = {key: torch.as_tensor(arrays[key][keep].astype(np.int64), device=device) for key in FIELDS}
        tensors["target"] = torch.as_tensor(arrays["probs"][keep, :2] / finished[keep, None], dtype=torch.float32, device=device)
        if name == "val":
            tensors["favourite"] = torch.as_tensor(np.concatenate(favourites)[keep], device=device)
        out[name] = tensors
    return out["train"], out["val"]


class Inputs:
    """Turns label rows into model keyword inputs."""

    def __init__(self, device):
        self.data = load_model_data(device=device)
        self.device = device

    def __call__(self, rows, card_table):
        cards = rows["cards"]
        index = cards - 1
        identity = self.data.identity_keys[index]
        identity = torch.where(rows["arts"] > 0, self.data.art_identity_keys[rows["arts"]], identity)
        metadata = dict(border_ids=rows["borders"], mutation_ids=rows["mutations"], pack_ids=self.data.pack_ids[index],
                        identity_keys=identity, red_support_ids=rows["red"], blue_support_ids=rows["blue"],
                        support_tiers=torch.stack([rows["red_tier"], rows["blue_tier"]], -1))
        if self.data.class_weights is not None:
            metadata["class_weights"] = self.data.class_weights[index]
        return dict(card_embeddings=card_table[index], **metadata)


def card_table(model, tokens):
    return model.description(tokens)


def evaluate(model, inputs, tokens, rows, batch_size=4096):
    model.eval()
    total = {"loss": 0.0, "accuracy": 0.0, "brier": 0.0, "baseline": 0.0, "upsets": 0.0, "upset_accuracy": 0.0,
             "kl": 0.0, "decisive": 0.0, "decisive_accuracy": 0.0, "probabilistic_error": 0.0}
    count = rows["target"].shape[0]
    with torch.no_grad():
        table = card_table(model, tokens)
        for start in range(0, count, batch_size):
            part = {k: v[start:start + batch_size] for k, v in rows.items()}
            logits = model(**inputs(part, table))
            target = part["target"]
            total["loss"] += float(-(target * logits.log_softmax(-1)).sum())
            total["accuracy"] += float((logits.argmax(-1) == target.argmax(-1)).sum())
            total["brier"] += float((logits.softmax(-1) - target).square().sum())
            # KL = loss minus the targets' own entropy: the part of the loss the model could still remove.
            total["kl"] += float((target * (target.clamp_min(1e-12).log() - logits.log_softmax(-1))).sum())
            decisive = target.max(-1).values >= 0.999  # deterministic battles: should be right every time
            total["decisive"] += float(decisive.sum())
            total["decisive_accuracy"] += float((decisive & (logits.argmax(-1) == target.argmax(-1))).sum())
            # Probabilistic battles: mean |predicted - estimated| A-win probability (the target is the estimate itself).
            total["probabilistic_error"] += float(((logits.softmax(-1)[:, 0] - target[:, 0]).abs() * (~decisive)).sum())
            winner = target.argmax(-1)
            upset = (part["favourite"] != winner) if "favourite" in part else torch.zeros_like(winner, dtype=torch.bool)  # abilities overturned the stat favourite
            total["baseline"] += float((~upset).sum())
            total["upsets"] += float(upset.sum())
            total["upset_accuracy"] += float((upset & (logits.argmax(-1) == winner)).sum())
    model.train()
    result = {k: v / count for k, v in total.items()
              if k not in ("upsets", "upset_accuracy", "decisive", "decisive_accuracy", "probabilistic_error")}
    result["probabilistic_error"] = total["probabilistic_error"] / max(1.0, count - total["decisive"])
    result["probabilistic_rows"] = count - total["decisive"]
    result["decisive_accuracy"] = total["decisive_accuracy"] / max(1.0, total["decisive"])
    result["upset_accuracy"] = total["upset_accuracy"] / max(1.0, total["upsets"])
    result["upset_rows"] = total["upsets"]
    return result


def set_dropout(model, p):
    """Dropout only at the GELU upscale of each feedforward (user); none on residuals or attention."""
    from ..model.networks import TransformerBlock
    for block in model.modules():
        if isinstance(block, TransformerBlock):
            for layer in block.feedforward:
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = p
            block.residual_dropout.p = 0.0
            block.attention.dropout = 0.0


def freeze_language(model):
    """Stop training the description transformer; its card vectors become fixed."""
    for parameter in model.description.parameters():
        parameter.requires_grad_(False)
    model.description.eval()


def weight_norm(model):
    return math.sqrt(sum(float(p.detach().square().sum()) for p in model.parameters()))


def train(*, steps=None, batch_size=512, lr=3e-4, warmup=1000, weight_decay=0.05, dropout=0.1, freeze_language_at=None,
          eval_every=1000,
          checkpoint_every=1000, reload_every=1000, device=None, run_dir=RUN_DIR, label_root=SHARD_DIR):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path, state_path, log_path = run_dir / "model.checkpoint", run_dir / "trainer.pt", run_dir / "log.jsonl"
    if model_path.exists() and state_path.exists():
        model, _ = load_checkpoint(model_path, map_location=device)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    else:
        model, state = BattleModel().to(device), None
    set_dropout(model, dropout)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    step, best = 0, None
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["weight_decay"] = weight_decay
        step, best = state["step"], state.get("best")
    inputs = Inputs(device)
    tokens = inputs.data.description_tokens
    label_dir = latest_label_dir(label_root)
    train_rows, val_rows = load_split(label_dir, device)
    # Grokking probes: fixed subsets of the training and validation rows, so the curves stay comparable
    # (the full validation set grows with new shards and changes with the rules).
    probe_gen = torch.Generator(device="cpu").manual_seed(0)
    def probe(rows, n):
        picks = torch.randperm(rows["target"].shape[0], generator=probe_gen)[:n].to(device)  # one draw for every field
        return {k: v[picks] for k, v in rows.items()}
    train_probe, val_probe = probe(train_rows, 8192), (probe(val_rows, 8192) if val_rows is not None else None)
    probe_labels = label_dir.name
    log = log_path.open("a")
    frozen_table = None

    def current_table():
        nonlocal frozen_table
        if freeze_language_at is not None and step >= freeze_language_at:
            if frozen_table is None:
                freeze_language(model)
                with torch.no_grad():
                    frozen_table = card_table(model, tokens)
                print(json.dumps({"step": step, "language_frozen": True}), flush=True)
            return frozen_table
        return card_table(model, tokens)
    generator = torch.Generator(device="cpu").manual_seed(step)
    started, last = time.time(), time.time()
    try:
        while steps is None or step < steps:
            if step and step % reload_every == 0:  # new shards, or new rules
                label_dir = latest_label_dir(label_root)
                train_rows, val_rows = load_split(label_dir, device)
            count = train_rows["target"].shape[0]
            picks = torch.randint(count, (batch_size,), generator=generator).to(device)
            batch = {k: v[picks] for k, v in train_rows.items()}
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, (step + 1) / warmup)
            logits = model(**inputs(batch, current_table()))
            loss = -(batch["target"] * logits.log_softmax(-1)).sum(-1).mean()
            total = loss + model.strategy.identity_penalty()
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                now = time.time()
                record = {"step": step, "loss": round(loss.item(), 4), "steps_per_s": round(100 / (now - last), 2),
                          "train_rows": count, "labels": label_dir.name}
                last = now
                if step % eval_every == 0 and val_rows is not None:
                    record["val"] = {k: round(v, 4) for k, v in evaluate(model, inputs, tokens, val_rows).items()}
                    record["val_rows"] = int(val_rows["target"].shape[0])
                    if label_dir.name != probe_labels:  # new rules: new probes
                        train_probe, val_probe, probe_labels = probe(train_rows, 8192), probe(val_rows, 8192), label_dir.name
                    record["grok"] = {"train_probe": {k: round(v, 4) for k, v in evaluate(model, inputs, tokens, train_probe).items()
                                                      if k in ("kl", "accuracy", "decisive_accuracy")},
                                      "val_probe": {k: round(v, 4) for k, v in evaluate(model, inputs, tokens, val_probe).items()
                                                    if k in ("kl", "accuracy", "upset_accuracy", "decisive_accuracy", "probabilistic_error")},
                                      "weight_norm": round(weight_norm(model), 2)}
                    kl = record["val"]["kl"]
                    if best is None or kl < best["kl"] or best.get("labels") != label_dir.name:
                        best = {"kl": kl, "step": step, "labels": label_dir.name}
                        save_checkpoint(model, run_dir / "best.checkpoint", metadata={**best, "val": record["val"]})
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            if step % checkpoint_every == 0:
                save_checkpoint(model, model_path, metadata={"step": step, "labels": label_dir.name,
                                                             "objective": "A initiates; outcome frequencies"})
                tmp = state_path.with_suffix(".tmp")
                torch.save({"optimizer": optimizer.state_dict(), "step": step, "best": best}, tmp)
                tmp.replace(state_path)
    finally:
        log.close()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1, help="at the GELU upscale only")
    parser.add_argument("--freeze-language-at", type=int, help="step after which the description transformer is frozen")
    parser.add_argument("--device")
    parser.add_argument("--reload-every", type=int, default=1000)
    parsed = parser.parse_args()
    train(steps=parsed.steps, batch_size=parsed.batch_size, lr=parsed.lr, device=parsed.device, reload_every=parsed.reload_every,
          weight_decay=parsed.weight_decay, dropout=parsed.dropout, freeze_language_at=parsed.freeze_language_at)
