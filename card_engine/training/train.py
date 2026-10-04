"""Train the general win predictor on simulator labels (side A initiates), resumable.

Each step encodes every card description once (289 sequences) and gathers the slot vectors,
so the description encoder trains end to end at a fraction of the per-slot cost. Targets are
the outcome probabilities (exact tablebase results or Monte Carlo frequencies), renormalised
over finished outcomes. The newest label
directory (rule fingerprint) is used; when rules change, training continues on the new labels.
Checkpoints hold the model (model.checkpoint) plus optimizer/step state for exact resumption.
"""

import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..model import BattleModel, load_model_data
from ..model.checkpoint import load_checkpoint, save_checkpoint
from .labels import FIELDS, FIXED_FIELDS, SHARD_DIR, fixed_arrays, possible_rows, shard_paths

RUN_DIR = Path(__file__).resolve().parents[2] / "data" / "training"
VALIDATION_EVERY = 25  # shard seeds divisible by this are held out


def latest_label_dir(root=SHARD_DIR):
    """The label store (every rules version's shards; rows are filtered by training.flags when loading)."""
    store = Path(root) / "store"
    if not any(store.glob("shard_*.npz")):
        raise FileNotFoundError(f"No label shards in {store}")
    return store


_SHARD_CACHE = {}  # path -> (mtime, rules, validation, rows): each shard is read and checked once per rules version


def load_split(directory, device):
    """(train, validation) tensors of the rows still valid under the current rules (training.flags);
    validation = shards whose seed is divisible by VALIDATION_EVERY.

    Shards are read and checked once, then cached: a reload reads only new shards, and rechecks the others only
    when the rules change. Card fields stay int16 as stored (a quarter of int64's memory); Inputs widens each batch.
    """
    from ..catalog import load_catalog
    from .flags import _pool_cards, entity_hashes, load_changes, valid_rows
    catalog = load_catalog()
    current, changes = entity_hashes(catalog), load_changes()
    rules = json.dumps([sorted(current.items()), changes])
    pool_cards = None
    split, seen = {"train": [], "val": []}, {}
    for path in sorted(shard_paths(directory)):
        mtime = path.stat().st_mtime
        cached = _SHARD_CACHE.get(path)
        if cached is None or cached[:2] != (mtime, rules):
            validation = int(path.stem.split("_")[1]) % VALIDATION_EVERY == 0
            with np.load(path) as shard:
                arrays = {key: shard[key] for key in (*FIELDS, "probs")}
                arrays.update(fixed_arrays(shard, len(arrays["probs"])))
                snapshot_id = str(shard["snapshot"])
            pool_cards = _pool_cards(catalog) if pool_cards is None else pool_cards
            mask = valid_rows(arrays, snapshot_id, current, catalog=catalog, changes=changes, pool_cards=pool_cards)
            finished = arrays["probs"][:, :2].sum(1)  # A win, B win (ties cannot happen; unfinished mass is dropped)
            keep = mask & (finished > 0) & possible_rows(arrays["cards"])
            rows = {key: arrays[key][keep].astype(np.int16) for key in FIELDS}
            rows.update({key: arrays[key][keep] for key in FIXED_FIELDS})
            rows["target"] = (arrays["probs"][keep, :2] / finished[keep, None]).astype(np.float32)
            if validation:
                rows["favourite"] = stat_favourite({key: torch.as_tensor(value) for key, value in rows.items()
                                                    if key != "target"}).numpy().astype(np.int8)
            cached = (mtime, rules, validation, rows)
        seen[path] = cached
        if len(cached[3]["target"]):
            split["val" if cached[2] else "train"].append(cached[3])
    _SHARD_CACHE.clear()
    _SHARD_CACHE.update(seen)  # shards that disappeared, or old rules' entries, are dropped
    out = {name: {key: torch.as_tensor(np.concatenate([part[key] for part in parts]), device=device) for key in parts[0]}
           if parts else None for name, parts in split.items()}
    return out["train"], out["val"]


def stat_favourite(rows):
    """Per row: the side (0/1) with more sqrt(HP x ATK) after supports, the naive baseline (ties go to A)."""
    stats = card_stats(rows, card_stat_tables())  # float64 on the CPU, so near-ties resolve as the simulator's stats
    strength = (stats[..., 0] * stats[..., 1]).sqrt().sum(-1)
    return (strength[:, 1] > strength[:, 0]).long()


JURASSIC_WORLD = 13  # blue: Prehistoric cards gain value% stats per Prehistoric card on the team


@functools.cache
def card_stat_tables():
    """(base, red, prehistoric, jurassic) for the model's card stats, from DaddyDrago's engine (drago.stat_tables).

    base [card, border, mutation, 2]: the card's (HP, ATK); red [card, mutation, support, tier, 2]: the red support's
    multiplier on that card (support 0 = none); prehistoric [card]: Prehistoric pack membership; jurassic [tier]:
    Jurassic World's fraction per Prehistoric card. Stat effects of abilities (entry multipliers, deck passives such
    as General Moon Zoo, awakened Toys) are left to the model, like every other ability.
    """
    from ..catalog import load_catalog
    from ..simulator.drago import stat_tables
    return tuple(torch.as_tensor(table, dtype=torch.float64) for table in stat_tables(load_catalog()))


def card_stats(rows, tables):
    """[N, 2, 4, 2]: each card's (HP, ATK) as it enters the battle, from label rows and card_stat_tables(). A
    fixed-stat battle (fixed_side 0 or 1) starts that side's four cards at fixed_stats instead."""
    base, red, prehistoric, jurassic = tables
    rows = {key: value.long() if key in FIELDS else value for key, value in rows.items()}
    cards, mutations = rows["cards"], rows["mutations"]
    red = red[cards, mutations, rows["red"][..., None], rows["red_tier"][..., None]]
    member = prehistoric[cards]
    bonus = torch.where(rows["blue"] == JURASSIC_WORLD, jurassic[rows["blue_tier"]], 0.0)
    blue = 1 + member * (bonus * member.sum(-1))[..., None]
    stats = base[cards, rows["borders"], mutations] * red * blue[..., None]
    if "fixed_side" in rows:
        side = rows["fixed_side"].to(stats.device).long()
        fixed = rows["fixed_stats"].to(stats)[:, None, None, :]  # [N, 1, 1, 2]
        chosen = (side[:, None] == torch.arange(2, device=stats.device))[..., None, None]  # [N, 2, 1, 1]
        stats = torch.where(chosen, fixed, stats)
    return stats


class Inputs:
    """Turns label rows into model keyword inputs."""

    def __init__(self, device):
        self.data = load_model_data(device=device)
        self.device = device
        self.stat_tables = tuple(table.to(device=device, dtype=torch.float32) for table in card_stat_tables())

    def __call__(self, rows, card_table):
        rows = {**{key: rows[key].long() for key in FIELDS},  # stored as int16
                **{key: rows[key] for key in FIXED_FIELDS if key in rows}}
        cards = rows["cards"]
        index = cards - 1
        identity = self.data.identity_keys[index]
        identity = torch.where(rows["arts"] > 0, self.data.art_identity_keys[rows["arts"]], identity)
        metadata = dict(border_ids=rows["borders"], mutation_ids=rows["mutations"], pack_ids=self.data.pack_ids[index],
                        identity_keys=identity, red_support_ids=rows["red"], blue_support_ids=rows["blue"],
                        support_tiers=torch.stack([rows["red_tier"], rows["blue_tier"]], -1))
        if self.data.class_weights is not None:
            metadata["class_weights"] = self.data.class_weights[index]
        metadata["card_stats"] = card_stats(rows, self.stat_tables)
        return dict(card_embeddings=card_table[index], **metadata)


def card_table(model, tokens):
    return model.description(tokens)


def evaluate(model, inputs, table, rows, batch_size=4096):
    """Scores rows against a card table computed in eval mode (see `train`'s `eval_table`)."""
    model.eval()
    total = {"loss": 0.0, "accuracy": 0.0, "brier": 0.0, "baseline": 0.0, "upsets": 0.0, "upset_accuracy": 0.0,
             "kl": 0.0, "decisive": 0.0, "decisive_accuracy": 0.0, "probabilistic_error": 0.0}
    count = rows["target"].shape[0]
    with torch.no_grad():
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
          eval_every=1000, init_from=None,
          checkpoint_every=1000, reload_every=1000, device=None, run_dir=RUN_DIR, label_root=SHARD_DIR):
    """Train in run_dir, resuming its model and optimizer if both are there. Otherwise init_from (a model checkpoint,
    e.g. one downloaded from another machine) gives the starting weights and step, with a fresh optimizer whose learning
    rate warms up again; with neither, training starts from random weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path, state_path, log_path = run_dir / "model.checkpoint", run_dir / "trainer.pt", run_dir / "log.jsonl"
    warm_from = 0  # step the learning-rate warmup counts from (a fresh optimizer warms up again)
    if model_path.exists() and state_path.exists():
        model, _ = load_checkpoint(model_path, map_location=device)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    elif init_from is not None:
        model, metadata = load_checkpoint(init_from, map_location=device)
        state, warm_from = None, int(metadata["step"])
    else:
        model, state = BattleModel().to(device), None
    set_dropout(model, dropout)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    step, best = warm_from, None
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["weight_decay"] = weight_decay
        step, best, warm_from = state["step"], state.get("best"), state.get("warm_from", 0)
    inputs = Inputs(device)
    tokens = inputs.data.description_tokens
    from .flags import snapshot
    label_dir = latest_label_dir(label_root)
    train_rows, val_rows = load_split(label_dir, device)
    if train_rows is None:
        raise SystemExit("No labels are valid under the current rules: see `python -m card_engine.training.flags status`")
    rules_id = snapshot()  # the current rules; probes and the best checkpoint reset when it changes
    # Grokking probes: fixed subsets of the training and validation rows, so the curves stay comparable
    # (the full validation set grows with new shards and changes with the rules).
    probe_gen = torch.Generator(device="cpu").manual_seed(0)
    def probe(rows, n):
        picks = torch.randperm(rows["target"].shape[0], generator=probe_gen)[:n].to(device)  # one draw for every field
        return {k: v[picks] for k, v in rows.items()}
    train_probe, val_probe = probe(train_rows, 8192), (probe(val_rows, 8192) if val_rows is not None else None)
    probe_labels = rules_id
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

    def eval_table():
        """Card vectors for evaluation: the frozen table, or one eval-mode pass shared by all three evaluations."""
        if frozen_table is not None:
            return frozen_table
        model.eval()
        with torch.no_grad():
            table = card_table(model, tokens)
        model.train()
        return table
    generator = torch.Generator(device="cpu").manual_seed(step)
    started, last = time.time(), time.time()
    try:
        while steps is None or step < steps:
            if step and step % reload_every == 0:  # new shards, or new rules
                label_dir = latest_label_dir(label_root)
                fresh = load_split(label_dir, device)
                if fresh[0] is None:  # every row held back (an undeclared engine change): keep the rows already loaded
                    print(json.dumps({"step": step, "labels_held_back": "no rows are valid under the current rules; "
                                      "training continues on the loaded rows until the change is declared (flags status)"}),
                          flush=True)
                else:
                    train_rows, val_rows = fresh
                    rules_id = snapshot()
            count = train_rows["target"].shape[0]
            picks = torch.randint(count, (batch_size,), generator=generator).to(device)
            batch = {k: v[picks] for k, v in train_rows.items()}
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, (step - warm_from + 1) / warmup)
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
                          "train_rows": count, "labels": rules_id}
                last = now
                if step % eval_every == 0 and val_rows is not None:
                    table = eval_table()
                    record["val"] = {k: round(v, 4) for k, v in evaluate(model, inputs, table, val_rows).items()}
                    record["val_rows"] = int(val_rows["target"].shape[0])
                    if rules_id != probe_labels:  # new rules: new probes
                        train_probe, val_probe, probe_labels = probe(train_rows, 8192), probe(val_rows, 8192), rules_id
                    record["grok"] = {"train_probe": {k: round(v, 4) for k, v in evaluate(model, inputs, table, train_probe).items()
                                                      if k in ("kl", "accuracy", "decisive_accuracy")},
                                      "val_probe": {k: round(v, 4) for k, v in evaluate(model, inputs, table, val_probe).items()
                                                    if k in ("kl", "accuracy", "upset_accuracy", "decisive_accuracy", "probabilistic_error")},
                                      "weight_norm": round(weight_norm(model), 2)}
                    kl = record["val"]["kl"]
                    if best is None or kl < best["kl"] or best.get("labels") != rules_id:
                        best = {"kl": kl, "step": step, "labels": rules_id}
                        save_checkpoint(model, run_dir / "best.checkpoint", metadata={**best, "val": record["val"]})
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            if step % checkpoint_every == 0:
                save_checkpoint(model, model_path, metadata={"step": step, "labels": rules_id,
                                                             "objective": "A initiates; outcome frequencies"})
                tmp = state_path.with_suffix(".tmp")
                torch.save({"optimizer": optimizer.state_dict(), "step": step, "best": best, "warm_from": warm_from}, tmp)
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
    parser.add_argument("--init-from", help="model checkpoint to start from when the run directory has no trainer state "
                        "(its step is kept; the optimizer starts fresh)")
    parser.add_argument("--device")
    parser.add_argument("--reload-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000,
                        help="steps between evaluations; each scores the whole validation set, so it grows with the data")
    parser.add_argument("--run-dir", default=RUN_DIR, help="checkpoints and log; a new directory starts from random weights")
    parsed = parser.parse_args()
    train(steps=parsed.steps, batch_size=parsed.batch_size, lr=parsed.lr, device=parsed.device, reload_every=parsed.reload_every,
          weight_decay=parsed.weight_decay, dropout=parsed.dropout, freeze_language_at=parsed.freeze_language_at,
          eval_every=parsed.eval_every, init_from=parsed.init_from,
          run_dir=parsed.run_dir)
