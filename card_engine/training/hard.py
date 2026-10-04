"""Hard examples: battles where the classifier and the engine disagree most (user, 2026-10-04).

Random sampling almost never produces teams that beat huge fixed stats through abilities (the floor-105 cheese
decks), so the classifier underrates them and its guesses there swing between checkpoints. Proposing teams with the
generator does not find them: it chases the model's own opinion, so its teams are ones the model already gets right
(first hard shards, 2026-10-04: mean |model - engine| 0.004). Disagreement mining does. Each round draws a fixed-stat
enemy and --candidates teams leaning on stat-ignoring cards (labels.STAT_IGNORING), scores all of them with the
current classifier (one batch) and the engine (milliseconds each), and keeps the --keep teams with the largest
|model - engine|, plus --keep-random others so the shards are not only extremes.

Enemies, half each:
  - a tower floor (card_engine.tower): 1-105 at a random difficulty, its fixed team or random cards;
  - random cards with stats HP = 10^U(2, 7.5), ATK = HP / 2 * 10^U(-0.5, 0.5).
The enemy is side B; the candidate attacks first. Candidates: each card a stat-ignoring one with probability 1/2,
else any card; borders, mutations and support tiers drawn per round (borderless, up to Crystal, or all borders;
mutations none or random; supports base or random tier, none 10% of the time).

Shards (hard_<seed>.npz, their own seed sequence) hold the label fields plus model_win, the model's win chance at
mining time; each prints the mean gap over all candidates and over the kept ones. The trainer repeats hard rows
(--hard-repeat) and scores validation's separately (val_hard).

    python -m card_engine.training.hard --shards 100 --rounds 20 --workers 7
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np

from .. import tower
from ..catalog import load_catalog
from ..mutations import MUTATION_NAMES
from ..teams import ASTRAEUS, ASTRAEUS_ARTS, SINGLE_COPY, side, spec
from .labels import AURA_TIERS, FIELDS, STORE, evaluate, stat_ignoring_cards
from .predict import Classifier
from .train import RUN_DIR

POD_RUN = Path(__file__).resolve().parents[2] / "data" / "training_pod"  # the pod's run, as scripts/pod_sync.sh brings it down
BORDER_SETS = ((1,), (1, 2, 3), tuple(range(1, 17)))  # borderless, up to Crystal, every border


def draw_enemy(rng, catalog):
    """(enemy side, fixed stats (HP, ATK, HP multiplier applies))."""
    if rng.random() < 0.5:
        floor, level = rng.randint(1, tower.FLOORS), rng.choice(list(tower.DIFFICULTIES))
        team = tower.fixed_team(catalog, floor)
        stats = tower.stats(floor, level)
    else:
        team, hp = None, 10 ** rng.uniform(2, 7.5)
        stats = (hp, hp / 2 * 10 ** rng.uniform(-0.5, 0.5), False)
    if team is None:
        cards = [rng.choice(catalog.cards).id for _ in range(4)]
        team = side([(card, 1, 0, rng.randint(1, len(ASTRAEUS_ARTS)) if card == ASTRAEUS else 0) for card in cards])
    return team, stats


def draw_candidate(rng, catalog, borders, mutate, tiered):
    """One attacking team, each card a stat-ignoring one with probability 1/2."""
    ignoring, every = stat_ignoring_cards(catalog), [c.id for c in catalog.cards]
    cards = []
    for _ in range(4):
        card = rng.choice(ignoring if rng.random() < 0.5 else every)
        while card in SINGLE_COPY and card in [c[0] for c in cards]:
            card = rng.choice(every)
        mutation = rng.randrange(len(MUTATION_NAMES)) if mutate and catalog.card(card).weather_id == 1 else 0
        cards.append((card, rng.choice(borders), mutation, rng.randint(1, len(ASTRAEUS_ARTS)) if card == ASTRAEUS else 0))
    supports = []
    for table in (catalog.red_supports, catalog.blue_supports):
        support = 0 if rng.random() < 0.1 else rng.choice(sorted(s.id for s in table))
        supports.append((support, (rng.choice(AURA_TIERS) if tiered else 1) if support else 0))
    return side(cards, *supports)


def _label(job):
    battle, seed, per_card = job
    return evaluate(_CATALOG, battle, seed, fixed=(1, per_card))


_CATALOG = None


def _init():
    global _CATALOG
    _CATALOG = load_catalog()


def hard_shard(classifier, catalog, seed, rounds, pool, candidates=1024, keep=64, keep_random=16):
    """One shard's rows: per round, the `keep` candidates the model gets most wrong plus `keep_random` others."""
    rng = random.Random(f"hard-{seed}")
    rows, gaps_all, gaps_kept = [], [], []
    for r in range(rounds):
        enemy, fixed = draw_enemy(rng, catalog)
        borders, mutate, tiered = rng.choice(BORDER_SETS), rng.random() < 0.5, rng.random() < 0.5
        teams = [draw_candidate(rng, catalog, borders, mutate, tiered) for _ in range(candidates)]
        model = classifier.ally_win([(team, enemy) for team in teams], fixed)[:, 0]
        per_card = tower.engine_stats(catalog, enemy["cards"], fixed)
        battles = [spec(team, enemy) for team in teams]
        results = pool.map(_label, [(b, seed * 1_000_003 + r * candidates + i, per_card) for i, b in enumerate(battles)],
                           chunksize=16)
        probs = np.array([p for p, _ in results], dtype=np.float32)
        finished = probs[:, :2].sum(1)
        engine = np.where(finished > 0, probs[:, 0] / np.maximum(finished, 1e-12), np.nan)
        gap = np.abs(engine - model)
        ranked = [i for i in np.argsort(-np.nan_to_num(gap, nan=-1.0)) if finished[i] > 0]
        chosen = ranked[:keep]
        rest = ranked[keep:]
        chosen += rng.sample(rest, min(keep_random, len(rest)))
        gaps_all.append(float(np.nanmean(gap)))
        gaps_kept.append(float(np.mean(gap[chosen[:keep]])) if chosen else float("nan"))
        rows += [(battles[i], results[i], fixed, float(model[i])) for i in chosen]
    arrays = {name: np.array([b[name] for b, *_ in rows], dtype=np.int16) for name in FIELDS}
    probs = np.array([res[0] for _, res, *_ in rows], dtype=np.float32)
    exact = np.array([res[1] for _, res, *_ in rows], dtype=bool)
    extra = {"fixed_side": np.ones(len(rows), dtype=np.int8),
             "fixed_stats": np.array([f[:2] for *_, f, _ in rows], dtype=np.float32),
             "fixed_hp_mult": np.array([f[2] for *_, f, _ in rows], dtype=np.int8),
             "model_win": np.array([m for *_, m in rows], dtype=np.float32)}
    return arrays, probs, exact, extra, float(np.mean(gaps_all)), float(np.nanmean(gaps_kept))


def run(shards, *, rounds=20, workers=None, checkpoint=None, out_dir=STORE, first_seed=None, device=None,
        candidates=1024, keep=64, keep_random=16):
    from .flags import snapshot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {int(p.stem.split("_")[1]) for p in out_dir.glob("hard_*.npz")}
    seed = first_seed if first_seed is not None else (max(existing) + 1 if existing else 1)
    catalog = load_catalog()
    with mp.get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        for done in range(1, shards + 1):
            while seed in existing:
                seed += 1
            path = checkpoint or next(p for p in (POD_RUN / "model.checkpoint", RUN_DIR / "model.checkpoint") if p.exists())
            classifier = Classifier(path, device)  # reloaded every shard: the mining follows training
            started = time.time()
            arrays, probs, exact, extra, gap_all, gap_kept = hard_shard(classifier, catalog, seed, rounds, pool,
                                                                        candidates, keep, keep_random)
            target = out_dir / f"hard_{seed:08d}.npz"
            tmp = target.with_name(f"partial_{target.name}")
            np.savez_compressed(tmp, probs=probs, exact=exact, snapshot=np.array(snapshot(catalog)), **arrays, **extra)
            os.replace(tmp, target)
            print(json.dumps({"shard": target.name, "done": done, "of": shards, "rows": len(probs),
                              "model_step": classifier.metadata.get("step"), "seconds": round(time.time() - started),
                              "mean_gap_all": round(gap_all, 3), "mean_gap_kept": round(gap_kept, 3)}), flush=True)
            seed += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=20, help="enemies per shard")
    parser.add_argument("--candidates", type=int, default=1024, help="teams scored per enemy")
    parser.add_argument("--keep", type=int, default=64, help="largest-disagreement teams kept per enemy")
    parser.add_argument("--keep-random", type=int, default=16, help="other teams kept per enemy")
    parser.add_argument("--workers", type=int, help="engine labelling processes")
    parser.add_argument("--checkpoint", help="model (default: data/training_pod's, else data/training's), reloaded per shard")
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    run(args.shards, rounds=args.rounds, workers=args.workers, checkpoint=args.checkpoint, first_seed=args.first_seed,
        device=args.device, candidates=args.candidates, keep=args.keep, keep_random=args.keep_random)


if __name__ == "__main__":
    main()
