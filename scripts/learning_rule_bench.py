"""A bench for learning rules other than backpropagation + AdamW, on this project's battle transformer.

    python3 scripts/learning_rule_bench.py --rule backprop --steps 3000                     # the trainer's own update
    python3 scripts/learning_rule_bench.py --rule my_rule.py --steps 3000 --lr 1e-3         # yours
    python3 scripts/learning_rule_bench.py --rule my_rule.py --from data/training_10s/ema.checkpoint --freeze-language

It reads a fixed subset of the label store (--shards), builds a model (fresh, small by default, or --from a trained
checkpoint), and runs --steps updates of the chosen rule on the same batches in the same order for every rule (--seed),
evaluating on held-out battles every --eval-every steps. Results go to stdout and --out (JSON lines), one line per
evaluation: KL (the loss the model could still remove), accuracy, upset accuracy, probability error, wall time.
It never writes to a run directory, so it runs beside the trainer (use --device cpu then).

A rule is a Python file defining make_rule(model, args) that returns an object with

    step(forward, batch) -> float      # update the model's parameters in place; return the batch loss

forward(batch) runs the model with autograd on and returns logits [B, 2] (A win, B win); loss(logits, batch) is the
trainer's loss (soft-label cross-entropy). The rule may ignore autograd entirely (forward under torch.no_grad, local
signals, hooks on the modules in model.strategy.blocks, ...). args carries --lr and anything after --rule-args.
"""

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the project root holds card_engine
from card_engine.model import BattleModel
from card_engine.model.checkpoint import load_checkpoint
from card_engine.model.config import DescriptionConfig, StrategicConfig
from card_engine.training.predict import default_device
from card_engine.training.train import (VALIDATION_EVERY, Inputs, card_table, evaluate, fit_stat_prior, latest_label_dir,
                                        load_split)


def loss(logits, batch):
    """The trainer's loss: cross-entropy against the engine's soft win probabilities."""
    return -(batch["target"] * logits.float().log_softmax(-1)).sum(-1).mean()


class Backprop:
    """The trainer's update: backpropagation, gradient norm clipped at 1, AdamW (weight decay 0.05)."""

    def __init__(self, model, args):
        self.model = model
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                           weight_decay=0.05)

    def step(self, forward, batch):
        value = loss(forward(batch), batch)
        self.optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(value.detach())


def make_rule(name, model, args):
    if name == "backprop":
        return Backprop(model, args)
    spec = importlib.util.spec_from_file_location("rule", name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_rule(model, args)


def subset(store, shards, prefixes, val_shards=None):
    """A temporary folder of links to a fixed subset of the store's shards: `shards` training shards and a tenth as many
    validation ones (seeds divisible by VALIDATION_EVERY), spread evenly over the store, each kind in `prefixes`."""
    paths = sorted(p for p in store.glob("*.npz") if p.name.split("_")[0] in prefixes)
    val = [p for p in paths if int(p.stem.split("_")[1]) % VALIDATION_EVERY == 0]
    train = [p for p in paths if p not in set(val)]
    pick = lambda items, n: items[::max(1, len(items) // n)][:n] if n else []
    folder = tempfile.TemporaryDirectory()
    for path in pick(train, shards) + pick(val, val_shards or max(1, shards // 10)):
        (Path(folder.name) / path.name).symlink_to(path.resolve())
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rule", default="backprop", help="backprop, or a file defining make_rule(model, args)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--shards", type=int, default=100, help="training shards read (about 2,000 battles each)")
    parser.add_argument("--val-shards", type=int, help="validation shards (default: a tenth of --shards)")
    parser.add_argument("--kinds", nargs="+", default=["shard", "fixed"],
                        help="shard kinds by file prefix (shard: random battles, fixed, hard, found, pvp, hidden)")
    parser.add_argument("--from", dest="start", help="start from this checkpoint instead of a fresh model")
    parser.add_argument("--layers", type=int, default=4, help="fresh model: strategic transformer layers")
    parser.add_argument("--width", type=int, default=128, help="fresh model: token width")
    parser.add_argument("--freeze-language", action="store_true",
                        help="keep the description transformer fixed (card vectors computed once): only the strategic "
                             "transformer learns")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--out", help="also append the evaluations here (JSON lines)")
    parser.add_argument("--resume", help="continue a rule's run from its checkpoint (the rule's load(path) gives the step)")
    parser.add_argument("--patience", type=int, default=0,
                        help="N > 0: --steps is a minimum; keep going while --track reached a new best within the last N "
                             "evaluations (up to --max-steps)")
    parser.add_argument("--track", default="upset_accuracy",
                        help="the metric --patience follows, higher is better (a name from the evaluation, e.g. accuracy; "
                             "prefix with - for lower is better, e.g. -kl); a rule with --save also keeps its best by it")
    parser.add_argument("--max-steps", type=int, default=1_000_000, help="--patience: never run past this step")
    parser.add_argument("--rule-args", nargs=argparse.REMAINDER, default=[], help="passed to the rule as args.rule_args")
    args = parser.parse_args()
    device = torch.device(args.device or default_device())
    torch.manual_seed(args.seed)

    folder = subset(latest_label_dir(), args.shards, set(args.kinds), args.val_shards)
    train_rows, val_rows = load_split(Path(folder.name), device)
    if train_rows is None or val_rows is None:
        raise SystemExit("No valid labels in the subset: more --shards, or see python -m card_engine.training.flags status")
    inputs = Inputs(device)
    if args.start:
        model = load_checkpoint(args.start, map_location=device)[0]
    else:
        width = args.width
        model = BattleModel(
            description_config=DescriptionConfig(width=width, layers=2, heads=max(1, width // 64),
                                                 feedforward_width=4 * width, output_width=width,
                                                 projection_hidden_width=width),
            strategic_config=StrategicConfig(layers=args.layers, width=width, heads=max(1, width // 64),
                                             attention_width=max(64, width // 2), feedforward_width=4 * width,
                                             stat_hidden_width=4 * width, stat_pairs=True)).to(device)
        scale, slope, bias, _ = fit_stat_prior(train_rows, inputs)  # the smooth stat prior, as a fresh trainer run
        model.strategy.config = model.strategic_config = dataclasses.replace(
            model.strategy.config, stat_prior=True, stat_prior_scale=scale, stat_prior_slope=slope, stat_prior_bias=bias)
    model.train()
    tokens = inputs.data.description_tokens
    frozen = None
    if args.freeze_language:
        model.description.requires_grad_(False)
        with torch.no_grad():
            frozen = card_table(model, tokens)

    def forward(batch):
        return model(**inputs(batch, frozen if frozen is not None else card_table(model, tokens)))

    rule = make_rule(args.rule, model, args)
    rule.track = args.track
    print(json.dumps({"rule": args.rule, "train_rows": len(train_rows["target"]), "val_rows": len(val_rows["target"]),
                      "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)
    generator = torch.Generator().manual_seed(args.seed)  # the same batches, in the same order, for every rule
    first = 1
    if args.resume:  # the rule restores itself; the batch sequence is fast-forwarded to where it stopped
        first = rule.load(args.resume, forward, {k: v[:args.batch_size] for k, v in train_rows.items()}) + 1
        for _ in range(first - 1):
            torch.randint(len(train_rows["target"]), (args.batch_size,), generator=generator)
    out = open(args.out, "a") if args.out else None
    start, losses = time.time(), []
    sign, metric = (-1.0, args.track[1:]) if args.track.startswith("-") else (1.0, args.track)
    tracked, since = -math.inf, 0  # best tracked value, evaluations since it improved
    step = first - 1
    while True:
        step += 1
        if step > args.steps and (not args.patience or since >= args.patience or step > args.max_steps):
            break
        picks = torch.randint(len(train_rows["target"]), (args.batch_size,), generator=generator).to(device)
        losses.append(rule.step(forward, {k: v[picks] for k, v in train_rows.items()}))
        if step % args.eval_every == 0 or (step == args.steps and not args.patience):
            model.eval()
            with torch.no_grad():
                table = frozen if frozen is not None else card_table(model, tokens)
                metrics = evaluate(model, inputs, table, val_rows)
            model.train()
            row = {"step": step, "train_loss": round(sum(losses) / len(losses), 4), "seconds": round(time.time() - start, 1),
                   **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float) and math.isfinite(v)}}
            losses.clear()
            print(json.dumps(row), flush=True)
            value = row.get(metric)
            if value is not None and sign * value > tracked:
                tracked, since = sign * value, 0
            else:
                since += 1
            if args.patience and step >= args.steps:
                print(json.dumps({"step": step, "track": args.track, "best": round(sign * tracked, 4),
                                  "evals_since_best": since, "patience": args.patience}), flush=True)
            if hasattr(rule, "on_eval"):  # a rule may checkpoint itself (its best by validation KL and by --track)
                rule.on_eval(step, row)
            if out:
                out.write(json.dumps({"rule": args.rule, **row}) + "\n")
                out.flush()
    if hasattr(rule, "on_finish"):  # the final state (latest), to extend the run later
        rule.on_finish(step - 1)
    folder.cleanup()


if __name__ == "__main__":
    main()
