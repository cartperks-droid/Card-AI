"""Hardened-circuit search on a discrete_rule checkpoint: fixed Boolean circuits drawn from its transmission probabilities.

    python3 scripts/harden_search.py data/bench_discrete_mac/ck_or-not_init6_30k/latest.pt --circuits 64

A hardened circuit fixes every connection (and bias) as transmitting or blocked, drawn once from its probability
sigma(w), so every gate is a deterministic Boolean function (a NOR neuron fires iff none of its active inputs' links
nor its bias transmit); attention and mixing are hardened (a > 1/2). The continuous tail is unchanged. Compared:
the stochastic network (sampled bits, as trained), the argmax circuit (a link transmits iff sigma(w) > 1/2), and
--circuits sampled circuits. The validation set is split in two: circuits are chosen by validation loss on the
selection half and reported on the other half, so picking the best of many is not scored on the battles it was
picked on. The run's setup (shards, model size, rule arguments) is read from the checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import discrete_rule as dr  # noqa: E402
from learning_rule_bench import subset  # noqa: E402
from card_engine.model import BattleModel  # noqa: E402
from card_engine.model.config import DescriptionConfig, StrategicConfig  # noqa: E402
from card_engine.training.predict import default_device  # noqa: E402
from card_engine.training.train import Inputs, card_table, evaluate, fit_stat_prior, latest_label_dir, load_split  # noqa: E402


class CircuitFire(dr.Firing):
    """Gates evaluated as one fixed hardened circuit: each gate's link draws are made once (seeded) and reused."""

    def __init__(self, seed, argmax=False):
        super().__init__()
        self.seed, self.argmax = seed, argmax
        self.draws = {}

    hard = property(lambda self: True, lambda self, value: None)  # attention and mixing always hardened (a > 1/2)

    def circuit(self, gate):
        key = id(gate)
        if key not in self.draws:
            sw = torch.sigmoid(gate.w)
            sb = torch.sigmoid(gate.b) if gate.use_bias else torch.zeros_like(gate.b)
            if self.argmax:
                T, tb = sw > 0.5, sb > 0.5
            else:
                g = torch.Generator(device="cpu").manual_seed(self.seed * 1_000_003 + len(self.draws))
                T = torch.rand(sw.shape, generator=g).to(sw.device) < sw
                tb = torch.rand(sb.shape, generator=g).to(sb.device) < sb
            self.draws[key] = ((T & gate.mask).float(), tb)
        return self.draws[key]

    def __call__(self, gate, x):
        x = x.detach()
        T, tb = self.circuit(gate)
        if T.ndim == 3:  # grouped [H, S, D], x [B, T, H, S]
            hits = torch.einsum("bths,hsd->bthd", x, T)
        else:
            hits = x @ T
        out = (~((hits > 0) | tb)).float()  # NOR: nothing transmitted, bias did not transmit
        if gate.ops == ["linear"]:
            out = 1 - out  # OR
        elif gate.ops != ["not"]:
            raise SystemExit("harden_search: or-not / or-linear gates only")
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--circuits", type=int, default=64)
    parser.add_argument("--shards", type=int, default=100, help="the run's --shards (its stat prior and validation)")
    parser.add_argument("--val-shards", type=int)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--out", help="JSON report")
    args = parser.parse_args()
    device = torch.device(args.device or default_device())
    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    folder = subset(latest_label_dir(), args.shards, {"shard", "fixed"}, args.val_shards)
    train_rows, val_rows = load_split(Path(folder.name), device)
    inputs = Inputs(device)
    width = args.width
    model = BattleModel(
        description_config=DescriptionConfig(width=width, layers=2, heads=max(1, width // 64), feedforward_width=4 * width,
                                             output_width=width, projection_hidden_width=width),
        strategic_config=StrategicConfig(layers=args.layers, width=width, heads=max(1, width // 64),
                                         attention_width=max(64, width // 2), feedforward_width=4 * width,
                                         stat_hidden_width=4 * width, stat_pairs=True)).to(device)
    import dataclasses
    scale, slope, bias, _ = fit_stat_prior(train_rows, inputs)
    model.strategy.config = model.strategic_config = dataclasses.replace(
        model.strategy.config, stat_prior=True, stat_prior_scale=scale, stat_prior_slope=slope, stat_prior_bias=bias)
    model.description.requires_grad_(False)
    with torch.no_grad():
        table = card_table(model, inputs.data.description_tokens)

    def forward(batch):
        return model(**inputs(batch, table))

    rule_args = []
    for key, value in state["rule_args"].items():  # the run's own rule arguments
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        rule_args += [flag] if value is True else [flag, str(value)]
    ns = argparse.Namespace(rule_args=rule_args, seed=args.seed, lr=3e-4)
    rule = dr.make_rule(model, ns)
    step = rule.load(args.checkpoint, forward, {k: v[:512] for k, v in train_rows.items()})
    del train_rows

    half = val_rows["target"].shape[0] // 2
    perm = torch.randperm(val_rows["target"].shape[0], generator=torch.Generator().manual_seed(1234)).to(device)
    select = {k: v[perm[:half]] for k, v in val_rows.items()}
    report_rows = {k: v[perm[half:]] for k, v in val_rows.items()}
    stack = rule.stack
    sampled = stack.fire
    keep = ("kl", "loss", "accuracy", "upset_accuracy", "decisive_accuracy", "probabilistic_error")

    def score(fire, rows):
        stack.fire = fire
        model.eval()
        with torch.no_grad():
            metrics = evaluate(model, inputs, table, rows)
        stack.fire = sampled
        return {k: round(float(metrics[k]), 4) for k in keep}

    results = {"checkpoint": args.checkpoint, "step": step, "selection_rows": half,
               "report_rows": report_rows["target"].shape[0]}
    results["stochastic"] = {"selection": score(sampled, select), "report": score(sampled, report_rows)}
    results["argmax"] = {"selection": score(CircuitFire(0, argmax=True), select),
                         "report": score(CircuitFire(0, argmax=True), report_rows)}
    print(json.dumps({"stochastic": results["stochastic"], "argmax": results["argmax"]}), flush=True)
    circuits = []
    for k in range(args.circuits):
        fire = CircuitFire(k + 1)
        sel = score(fire, select)
        circuits.append((sel["loss"], k + 1, sel, fire))
        print(json.dumps({"circuit": k + 1, "selection": sel}), flush=True)
    circuits.sort(key=lambda c: c[0])
    best_loss, best_seed, best_sel, best_fire = circuits[0]
    losses = [c[0] for c in circuits]
    results["sampled_circuits"] = {
        "count": len(circuits), "selection_loss_best": best_loss, "selection_loss_median": sorted(losses)[len(losses) // 2],
        "selection_loss_worst": losses[-1], "best_seed": best_seed,
        "best": {"selection": best_sel, "report": score(best_fire, report_rows)}}
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    folder.cleanup()


if __name__ == "__main__":
    main()
