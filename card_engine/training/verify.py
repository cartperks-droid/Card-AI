"""Check a trained classifier against the simulator on chosen matchups (side A attacks first).

    python -m card_engine.training.verify --checkpoint data/training/best.checkpoint --suite coherent --n 300
    python -m card_engine.training.verify --checkpoint data/training/best.checkpoint --suite random --n 300
    python -m card_engine.training.verify --checkpoint data/training/best.checkpoint --file my_matchups.json

Suites: "random" draws matchups the way training does (fresh seeds, so not training rows); "coherent" builds each
side from one class or pack, the themed teams real decks use and random training rarely produces. A --file holds a
JSON list of matchups, cards by name or ID and borders by name ("GaPl") or ID:

    [{"a": {"cards": ["Vampire Lord", "Set", "Good Boy", "Archer"], "borders": ["GaPl", 1, 1, 1], "red": "3:5", "blue": "11:5"},
      "b": {"cards": ["Immortal Witch", 2, 3, 4]}}]

Missing borders default to no border, mutations to none, supports to none. The report gives winner agreement overall,
on deterministic battles and on upsets, the mean A-win error on random battles, and the largest disagreements.
"""

import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import torch

from ..catalog import load_catalog
from ..model.checkpoint import load_checkpoint
from ..mutations import MUTATION_NAMES
from ..simulator.catalog_rules import ASTRAEUS, SUPPORTED
from .counter import BORDER_NAMES
from .labels import ART_NAMES, FIELDS, label_specs, random_spec
from .train import Inputs, card_table, stat_favourite

BORDER_IDS = {name.lower(): border for border, name in BORDER_NAMES.items()}
FRESH_SEED = 10 ** 9  # far from the label shards' seeds, so suites never replay training rows


def _card(catalog, value):
    if isinstance(value, int):
        return value
    names = {card.name.lower(): card.id for card in catalog.cards}
    if value.lower() not in names:
        raise SystemExit(f"Unknown card: {value!r}")
    return names[value.lower()]


def _border(value):
    if isinstance(value, int):
        return value
    if value.lower() not in BORDER_IDS:
        raise SystemExit(f"Unknown border: {value!r} (use one of {', '.join(BORDER_NAMES.values())})")
    return BORDER_IDS[value.lower()]


def _support(text):
    support, _, tier = str(text or 0).partition(":")
    return (int(support), int(tier or 1)) if int(support) else (0, 0)


def spec_from_entry(catalog, entry):
    """A matchup from the --file format."""
    spec = {name: [] for name in FIELDS}
    for side in ("a", "b"):
        team = entry[side]
        cards = [_card(catalog, c) for c in team["cards"]]
        if len(cards) != 4:
            raise SystemExit(f"Each side needs 4 cards: {team['cards']}")
        mutations = [MUTATION_NAMES.index(m) if isinstance(m, str) else m for m in team.get("mutations", [0] * 4)]
        spec["cards"].append(cards)
        spec["borders"].append([_border(b) for b in team.get("borders", [1] * 4)])
        spec["mutations"].append(mutations)
        spec["arts"].append([1 if c == ASTRAEUS else 0 for c in cards])
        for color in ("red", "blue"):
            support, tier = _support(team.get(color))
            spec[color].append(support)
            spec[color + "_tier"].append(tier)
    return spec


def coherent_spec(rng, catalog, groups):
    """A random matchup (borders, supports as in training) whose sides each come from one class or pack."""
    spec = random_spec(rng, catalog)
    for side in (0, 1):
        members = groups[rng.randrange(len(groups))][1]
        cards = rng.sample(members, 4)
        spec["cards"][side] = cards
        spec["mutations"][side] = [m if catalog.card(c).weather_id == 1 else 0 for c, m in zip(cards, spec["mutations"][side])]
        spec["arts"][side] = [rng.randint(1, len(ART_NAMES)) if c == ASTRAEUS else 0 for c in cards]
    return spec


def theme_groups(catalog):
    """Classes and packs with at least 4 simulated cards: (label, card IDs)."""
    groups = {}
    for card_id in sorted(SUPPORTED):
        card = catalog.card(card_id)
        for label in [f"class {c}" for c in card.classes] + [f"pack {p}" for p in card.packs]:
            groups.setdefault(label, []).append(card_id)
    return sorted((label, ids) for label, ids in groups.items() if len(ids) >= 4)


def _simulate(job):
    specs, seed = job
    probs, exact = label_specs(load_catalog(), specs, seed=seed)
    return probs, exact


def simulate(specs, workers, seed):
    chunks = [specs[i:i + 8] for i in range(0, len(specs), 8)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_simulate, [(chunk, seed + i) for i, chunk in enumerate(chunks)])
    return np.concatenate([p for p, _ in parts]), np.concatenate([e for _, e in parts])


def predict(model, inputs, specs, device, batch=1024):
    out = []
    with torch.no_grad():
        table = card_table(model, inputs.data.description_tokens)
        for start in range(0, len(specs), batch):
            part = specs[start:start + batch]
            rows = {key: torch.tensor([s[key] for s in part], device=device) for key in FIELDS}
            out.append(model(**inputs(rows, table)).softmax(-1)[:, 0].float().cpu())
    return torch.cat(out).numpy()


def describe(catalog, spec, side):
    return [f"{catalog.card(c).name} ({BORDER_NAMES[b]})" for c, b in zip(spec["cards"][side], spec["borders"][side])]


def report(catalog, specs, model_a, sim_probs, exact, show):
    finished = sim_probs[:, :2].sum(1)
    keep = finished > 0
    sim_a = np.where(keep, sim_probs[:, 0] / np.maximum(finished, 1e-12), np.nan)
    favourite = stat_favourite({key: torch.tensor([s[key] for s in specs]) for key in FIELDS}).numpy()
    sim_winner, model_winner = (sim_a < 0.5).astype(int), (model_a < 0.5).astype(int)
    deterministic = keep & ((sim_a >= 0.999) | (sim_a <= 0.001))
    upset = keep & (favourite != sim_winner)
    agree = model_winner == sim_winner
    rate = lambda mask: f"{agree[mask].mean():.1%} of {int(mask.sum())}" if mask.any() else "none"
    summary = {"matchups": int(keep.sum()), "unfinished_dropped": int((~keep).sum()),
               "winner_agreement": rate(keep), "deterministic_agreement": rate(deterministic), "upset_agreement": rate(upset),
               "random_battles_mean_A_win_error": round(float(np.abs(model_a - sim_a)[keep & ~deterministic].mean()), 3)
               if (keep & ~deterministic).any() else None,
               "stronger_stats_rule_agreement": f"{(favourite == sim_winner)[keep].mean():.1%}" if keep.any() else "none"}
    order = np.argsort(-np.nan_to_num(np.abs(model_a - sim_a), nan=-1))[:show]
    worst = [{"A": describe(catalog, specs[i], 0), "B": describe(catalog, specs[i], 1),
              "supports": {"A": [specs[i]["red"][0], specs[i]["blue"][0]], "B": [specs[i]["red"][1], specs[i]["blue"][1]]},
              "model_A_win": round(float(model_a[i]), 3), "simulator_A_win": round(float(sim_a[i]), 3),
              "simulator_exact": bool(exact[i]), "upset": bool(upset[i])} for i in order if keep[i]]
    return summary, worst


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", choices=("random", "coherent"), default="coherent")
    parser.add_argument("--file", help="JSON list of matchups (see above); replaces --suite")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--show", type=int, default=10, help="largest disagreements to list")
    parser.add_argument("--out", help="write every matchup's result here (JSON)")
    args = parser.parse_args()

    catalog = load_catalog()
    if args.file:
        specs = [spec_from_entry(catalog, entry) for entry in json.loads(Path(args.file).read_text())]
    else:
        rng = random.Random(FRESH_SEED + args.seed)
        groups = theme_groups(catalog)
        draw = (lambda: coherent_spec(rng, catalog, groups)) if args.suite == "coherent" else (lambda: random_spec(rng, catalog))
        specs = [draw() for _ in range(args.n)]

    model, metadata = load_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()
    model_a = predict(model, Inputs(args.device), specs, args.device)
    sim_probs, exact = simulate(specs, args.workers, FRESH_SEED + args.seed)
    summary, worst = report(catalog, specs, model_a, sim_probs, exact, args.show)
    summary = {"checkpoint_step": metadata.get("step"), "suite": "file" if args.file else args.suite, **summary}
    print(json.dumps({"summary": summary, "largest_disagreements": worst}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps([{"spec": s, "model_A_win": float(m), "simulator": [float(x) for x in p],
                                               "exact": bool(e)} for s, m, p, e in zip(specs, model_a, sim_probs, exact)]))


if __name__ == "__main__":
    main()
