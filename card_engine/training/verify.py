"""Check a trained classifier against the simulator on chosen matchups (side A attacks first).

    python -m card_engine.training.verify --checkpoint data/training/best.checkpoint --suite coherent --n 300
    python -m card_engine.training.verify --checkpoint data/training/best.checkpoint --suite random --n 300
    python -m card_engine.training.verify --checkpoint data/training/best.checkpoint --file my_matchups.json
    python -m card_engine.training.verify --checkpoint data/training/best.checkpoint --suite tower --n 20

Suites: "random" draws matchups the way training does (fresh seeds, so not training rows); "coherent" builds each
side from one class or pack, the themed teams real decks use and random training rarely produces; "tower" plays
borderless teams from the restricted deck (--pool restricted) or DaddyDrago's cheese pool (--pool cheese) against
the fixed Tower teams (data/scenarios/tower_teams.json) with floor-set stats, --n per floor; any win is an upset. A --file holds a
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
from .counter import BORDER_NAMES
from .labels import FIELDS, evaluate, label_specs, random_spec
from .train import Inputs, card_table

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
        spec["arts"].append([0] * 4)  # Astraeus draws its art in battle
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
    return spec


def theme_groups(catalog):
    """Classes and packs with at least 4 simulated cards: (label, card IDs)."""
    groups = {}
    for card_id in sorted(card.id for card in catalog.cards):
        card = catalog.card(card_id)
        for label in [f"class {c}" for c in card.classes] + [f"pack {p}" for p in card.packs]:
            groups.setdefault(label, []).append(card_id)
    return sorted((label, ids) for label, ids in groups.items() if len(ids) >= 4)


TOWER_FILE = Path(__file__).resolve().parents[2] / "data" / "scenarios" / "tower_teams.json"


def tower_stats(floor, difficulty, hp_multipliers, difficulty_ids):
    """Fixed (HP, ATK) per enemy card, as DaddyDrago's engine sets Tower enemies (see TOWER_FILE's note)."""
    power = -(-2 * ((6000 + floor ** 3 * 50) / 2) ** 0.5 * 4 ** (difficulty_ids[difficulty] - 1) // 1)
    keep_hp = difficulty in ("Normal", "Impossible")
    return [(float(-(-power * (m if keep_hp else 1) // 1)), float(-(-power // 2))) for m in hp_multipliers]


def tower_suite(rng, catalog, n, difficulty, aura_tier, pool_name="restricted"):
    """Per floor, n borderless teams (random cards with replacement, order and blue support) against that floor's team.

    pool "restricted": the restricted deck's cards (player-base availability), all borderless (user: tests stay
    borderless, or a max-stat card would win everything); "cheese": DaddyDrago's Tower cheese pool."""
    data = json.loads(TOWER_FILE.read_text())
    if pool_name == "cheese":
        pool = [_card(catalog, name) for name in data["cheese_cards"]]
    else:
        from ..restricted import cards as restricted_cards, load as load_restricted
        pool = sorted(restricted_cards(load_restricted(), catalog))
    single = {_card(catalog, name) for name in data["single_copy"]}
    specs, enemy_stats, floors = [], [], []
    for floor, team in data["teams"].items():
        enemy = [_card(catalog, entry[0]) for entry in team]
        stats = tower_stats(int(floor), difficulty, [entry[1] for entry in team], data["difficulty_ids"])
        for _ in range(n):
            while True:
                cards = [rng.choice(pool) for _ in range(4)]
                if all(cards.count(c) <= 1 for c in single):
                    break
            blue = rng.choice(data["cheese_blue_supports"])
            specs.append({"cards": [cards, enemy], "borders": [[1] * 4, [1] * 4], "mutations": [[0] * 4, [0] * 4],
                          "arts": [[0] * 4, [0] * 4],
                          "red": [0, 0], "red_tier": [0, 0], "blue": [blue, 0], "blue_tier": [aura_tier if blue else 0, 0]})
            enemy_stats.append(stats)
            floors.append(int(floor))
    return specs, enemy_stats, floors


def _simulate(job):
    specs, stats, seed = job
    catalog = load_catalog()
    if stats is None:
        return label_specs(catalog, specs, seed=seed)
    probs = np.zeros((len(specs), 4), dtype=np.float32)
    exact = np.zeros(len(specs), dtype=bool)
    for i, (spec, enemy) in enumerate(zip(specs, stats)):
        outcome, is_exact = evaluate(catalog, spec, seed * 1_000_003 + i, fixed=(1, enemy))
        probs[i], exact[i] = outcome, is_exact
    return probs, exact


def simulate(specs, workers, seed, enemy_stats=None):
    """Simulator outcomes (A, B, tie, unfinished) and exactness; enemy_stats fixes side B's (HP, ATK) per matchup."""
    chunks = [(specs[i:i + 8], None if enemy_stats is None else enemy_stats[i:i + 8]) for i in range(0, len(specs), 8)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_simulate, [(chunk, stats, seed + i) for i, (chunk, stats) in enumerate(chunks)])
    return np.concatenate([p for p, _ in parts]), np.concatenate([e for _, e in parts])


def predict(model, inputs, specs, device, enemy_stats=None, batch=1024):
    """Model A-win probabilities, and the (HP, ATK) it was given per card ([N, 2, 4, 2]); enemy_stats overrides side B's."""
    out, given = [], []
    with torch.no_grad():
        table = card_table(model, inputs.data.description_tokens)
        for start in range(0, len(specs), batch):
            part = specs[start:start + batch]
            rows = {key: torch.tensor([s[key] for s in part], device=device) for key in FIELDS}
            kwargs = inputs(rows, table)
            if enemy_stats is not None:
                kwargs["card_stats"][:, 1] = torch.tensor(enemy_stats[start:start + batch], dtype=kwargs["card_stats"].dtype, device=device)
            out.append(model(**kwargs).softmax(-1)[:, 0].float().cpu())
            given.append(kwargs["card_stats"].double().cpu())
    return torch.cat(out).numpy(), torch.cat(given)


def describe(catalog, spec, side):
    return [f"{catalog.card(c).name} ({BORDER_NAMES[b]})" for c, b in zip(spec["cards"][side], spec["borders"][side])]


def report(catalog, specs, model_a, sim_probs, exact, show, stats):
    finished = sim_probs[:, :2].sum(1)
    keep = finished > 0
    sim_a = np.where(keep, sim_probs[:, 0] / np.maximum(finished, 1e-12), np.nan)
    strength = (stats[..., 0] * stats[..., 1]).sqrt().sum(-1)  # the stat rule on the stats actually played
    favourite = (strength[:, 1] > strength[:, 0]).long().numpy()
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
    parser.add_argument("--suite", choices=("random", "coherent", "tower"), default="coherent")
    parser.add_argument("--difficulty", choices=("Normal", "Hard", "Extreme", "Hell", "Impossible"), default="Impossible",
                        help="tower suite: sets the enemies' stats (his Tower Cheese Maker defaults to Impossible)")
    parser.add_argument("--aura-tier", type=int, default=1, help="tower suite: tier of the cheese team's blue support")
    parser.add_argument("--pool", choices=("restricted", "cheese"), default="restricted",
                        help="tower suite: draw player cards from the restricted deck or DaddyDrago's cheese pool (always borderless)")
    parser.add_argument("--file", help="JSON list of matchups (see above); replaces --suite")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--show", type=int, default=10, help="largest disagreements to list")
    parser.add_argument("--out", help="write every matchup's result here (JSON)")
    args = parser.parse_args()

    catalog = load_catalog()
    enemy_stats = floors = None
    if args.file:
        specs = [spec_from_entry(catalog, entry) for entry in json.loads(Path(args.file).read_text())]
    elif args.suite == "tower":
        specs, enemy_stats, floors = tower_suite(random.Random(FRESH_SEED + args.seed), catalog, args.n, args.difficulty,
                                                 args.aura_tier, args.pool)
    else:
        rng = random.Random(FRESH_SEED + args.seed)
        groups = theme_groups(catalog)
        draw = (lambda: coherent_spec(rng, catalog, groups)) if args.suite == "coherent" else (lambda: random_spec(rng, catalog))
        specs = [draw() for _ in range(args.n)]

    model, metadata = load_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()
    model_a, stats = predict(model, Inputs(args.device), specs, args.device, enemy_stats)
    sim_probs, exact = simulate(specs, args.workers, FRESH_SEED + args.seed, enemy_stats)
    summary, worst = report(catalog, specs, model_a, sim_probs, exact, args.show, stats)
    summary = {"checkpoint_step": metadata.get("step"), "suite": "file" if args.file else args.suite, **summary}
    if floors is not None:
        summary["difficulty"] = args.difficulty
        finished = sim_probs[:, :2].sum(1)
        sim_a = sim_probs[:, 0] / np.maximum(finished, 1e-12)
        summary["by_floor"] = {f: {"simulator_win_rate": round(float(sim_a[[i for i, x in enumerate(floors) if x == f]].mean()), 3),
                                   "model_win_rate": round(float(model_a[[i for i, x in enumerate(floors) if x == f]].mean()), 3),
                                   "winner_agreement": f"{((model_a < 0.5) == (sim_a < 0.5))[[i for i, x in enumerate(floors) if x == f]].mean():.1%}"}
                               for f in sorted(set(floors), reverse=True)}
    print(json.dumps({"summary": summary, "largest_disagreements": worst}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps([{"spec": s, "model_A_win": float(m), "simulator": [float(x) for x in p],
                                               "exact": bool(e)} for s, m, p, e in zip(specs, model_a, sim_probs, exact)]))


if __name__ == "__main__":
    main()
