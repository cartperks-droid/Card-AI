"""Hard examples: battles the generator proposes, labelled by the engine (user, 2026-10-04).

Random sampling almost never produces teams that beat huge fixed stats through abilities (the floor-105 cheese
decks), so the classifier's guesses there swing between checkpoints. This loop puts labels where the model is
weakest: each round draws a fixed-stat enemy, lets the generator (training.generate) ascend the current classifier
toward 32 counters, and labels every one with the engine. Each shard (hard_<seed>.npz, its own seed sequence) has
the label fields plus the model's win chance at proposal time (model_win, for tracking the gap; training ignores it).
The trainer repeats hard rows in training (--hard-repeat) and scores validation's separately (val_hard).

Enemies, half each:
  - a tower floor (card_engine.tower): 1-105 at a random difficulty, its fixed team or random cards;
  - random cards with stats HP = 10^U(2, 7.5), ATK = HP / 2 * 10^U(-0.5, 0.5).
The enemy is side B; the generated team attacks first. Pool: every card, with masks drawn per round (borderless,
up to Crystal, or all borders; mutations none or all; support tiers base or all), so cheap and strong answers both
appear.

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
from ..teams import ASTRAEUS, ASTRAEUS_ARTS, side, spec
from .generate import Settings, SlotSpace, counters, make_pool, masks
from .labels import FIELDS, STORE, evaluate
from .predict import Classifier
from .train import RUN_DIR

POD_RUN = Path(__file__).resolve().parents[2] / "data" / "training_pod"  # the pod's run, as scripts/pod_sync.sh brings it down
MASKS = [(b, m, t) for b in (("none",), ("none", "Pl", "Cr"), ("all",)) for m in (("None",), ("all",)) for t in (("base",), ("all",))]


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


def _label(job):
    battle, seed, per_card = job
    return evaluate(_CATALOG, battle, seed, fixed=(1, per_card))


_CATALOG = None


def _init():
    global _CATALOG
    _CATALOG = load_catalog()


def hard_shard(classifier, catalog, seed, rounds, pool, spaces, settings):
    """One shard's rows: `rounds` enemies, 32 generated counters each, engine-labelled."""
    rng = random.Random(f"hard-{seed}")
    battles, stats, model_win, jobs = [], [], [], []
    for r in range(rounds):
        enemy, fixed = draw_enemy(rng, catalog)
        mask = rng.choice(MASKS)
        if mask not in spaces:
            borders, mutations, tiers = masks(*map(list, mask))
            spaces[mask] = SlotSpace(classifier, make_pool(catalog, "all", borders=borders, mutations=mutations, tiers=tiers))
        found = counters(spaces[mask], enemy, count=32, restarts=64, settings=settings, seed=seed * 1000 + r,
                         enemy_stats=fixed)
        per_card = tower.engine_stats(catalog, enemy["cards"], fixed)
        for team, win, _ in found:
            jobs.append((spec(team, enemy), seed * 1_000_003 + len(jobs), per_card))
            battles.append(jobs[-1][0])
            stats.append(fixed)
            model_win.append(win)
    results = pool.map(_label, jobs, chunksize=4)
    probs = np.array([p for p, _ in results], dtype=np.float32)
    exact = np.array([e for _, e in results], dtype=bool)
    arrays = {name: np.array([b[name] for b in battles], dtype=np.int16) for name in FIELDS}
    extra = {"fixed_side": np.ones(len(battles), dtype=np.int8),
             "fixed_stats": np.array([s[:2] for s in stats], dtype=np.float32),
             "fixed_hp_mult": np.array([s[2] for s in stats], dtype=np.int8),
             "model_win": np.array(model_win, dtype=np.float32)}
    return arrays, probs, exact, extra


def run(shards, *, rounds=20, workers=None, checkpoint=None, out_dir=STORE, first_seed=None, device=None):
    from .flags import snapshot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {int(p.stem.split("_")[1]) for p in out_dir.glob("hard_*.npz")}
    seed = first_seed if first_seed is not None else (max(existing) + 1 if existing else 1)
    catalog = load_catalog()
    settings = Settings(role="attack")
    with mp.get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        for done in range(1, shards + 1):
            while seed in existing:
                seed += 1
            path = checkpoint or next(p for p in (POD_RUN / "model.checkpoint", RUN_DIR / "model.checkpoint") if p.exists())
            classifier = Classifier(path, device)  # reloaded every shard: the proposals follow training
            started = time.time()
            arrays, probs, exact, extra = hard_shard(classifier, catalog, seed, rounds, pool, {}, settings)
            target = out_dir / f"hard_{seed:08d}.npz"
            tmp = target.with_name(f"partial_{target.name}")
            np.savez_compressed(tmp, probs=probs, exact=exact, snapshot=np.array(snapshot(catalog)), **arrays, **extra)
            os.replace(tmp, target)
            finished = probs[:, :2].sum(1)
            engine = np.where(finished > 0, probs[:, 0] / np.maximum(finished, 1e-12), np.nan)
            print(json.dumps({"shard": target.name, "done": done, "of": shards, "rows": len(probs),
                              "model_step": classifier.metadata.get("step"), "seconds": round(time.time() - started),
                              "mean_abs_gap": round(float(np.nanmean(np.abs(engine - extra["model_win"]))), 3),
                              "engine_wins_over_half": int(np.nansum(engine > 0.5))}), flush=True)
            seed += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=20, help="enemies per shard (32 battles each)")
    parser.add_argument("--workers", type=int, help="engine labelling processes")
    parser.add_argument("--checkpoint", help="model (default: data/training_pod's, else data/training's), reloaded per shard")
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    run(args.shards, rounds=args.rounds, workers=args.workers, checkpoint=args.checkpoint, first_seed=args.first_seed,
        device=args.device)


if __name__ == "__main__":
    main()
