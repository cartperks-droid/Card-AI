"""Tune the generator's settings by the engine: each trial generates counters to tower floors with one setting and
the engine plays the best of them. Settings that propose teams the engine lets win score highest.

    python3 scripts/tune_generator.py --checkpoint data/training_10s/step298k_ema.checkpoint
    python3 scripts/tune_generator.py --checkpoint ... --floors "105 Impossible" "100 Impossible" --trials 40 --pool custom

Trial 0 is the current defaults (generate.Settings), the baseline; the others draw each setting from SPACE. The
model-search trials ("M" plus its size, --model-search) replace the ascent with generate.model_search, an evolution
scored by the model, and keep its distinct best teams (2026-10-07: every ascent setting scored 0 at floors 95-105). Each
trial verifies its `--top` best teams per floor attacking first (the player always starts in the tower) and scores
the floor's best and mean engine win. One JSON line per trial goes to --output as it finishes, so a long run can be
stopped and read; the end prints the trials best first. Pool and masks are generate's (--pool, --borders ...).
"""

import argparse
import dataclasses
import json
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the project root holds card_engine
from card_engine import tower
from card_engine.catalog import load_catalog
from card_engine.teams import describe
from card_engine.training.counter import _init
from card_engine.training.generate import Settings, SlotSpace, _role_win, counters, make_pool, masks, model_search
from card_engine.training.hard import distinct
from card_engine.training.predict import Classifier

SPACE = {  # each trial draws one value per setting
    "steps": (150, 300, 600),
    "lr": (0.02, 0.05, 0.1),
    "temperature": (0.05, 0.1, 0.2),
    "commitment": (0.5, 1.0, 2.0),
    "nearest": (1, 3, 5),
    "noise_levels": (1, 6, 12, 24),
    "sigma_max": (2.0, 4.0, 8.0, 16.0),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--floors", nargs="+", default=["105 Impossible", "100 Impossible", "95 Impossible"],
                        help="tower floors with fixed teams (every fifth), as 'FLOOR DIFFICULTY'")
    parser.add_argument("--trials", type=int, default=24, help="ascent trials, the defaults first (0: model search only)")
    parser.add_argument("--top", type=int, default=8, help="teams per floor the engine plays")
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--pool", choices=("own", "custom", "restricted", "all"), default="restricted")
    parser.add_argument("--borders", nargs="+", default=["none"])
    parser.add_argument("--mutations", nargs="+", default=["None"])
    parser.add_argument("--support-tiers", nargs="+", default=["base"])
    parser.add_argument("--model-search", type=int, nargs="*", default=[10000, 50000],
                        help="model-search trial sizes (teams the model scores); none: ascent trials only")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--output", default="data/tune_generator.jsonl")
    args = parser.parse_args()

    catalog = load_catalog()
    floors = []
    for text in args.floors:
        floor, level = text.split()
        enemy = tower.fixed_team(catalog, int(floor))
        if enemy is None:
            parser.error(f"floor {floor} has no fixed team (fixed floors are every fifth)")
        stats = tower.stats(int(floor), tower.difficulty(level))
        enemy.update(borders=[1] * 4, mutations=[0] * 4, red=0, red_tier=0, blue=0, blue_tier=0)
        floors.append((text, enemy, stats, tower.engine_stats(catalog, enemy["cards"], stats)))
    borders, mutations, tiers = masks(args.borders, args.mutations, args.support_tiers)
    space = SlotSpace(Classifier(args.checkpoint, args.device),
                      make_pool(catalog, args.pool, borders=borders, mutations=mutations, tiers=tiers))
    rng = random.Random(args.seed)
    trials = [(f"M{n}", n) for n in args.model_search] + [(0, Settings())] * (args.trials > 0)
    trials += [(i, Settings(**{k: rng.choice(v) for k, v in SPACE.items()})) for i in range(1, args.trials)]

    def teams(setting, enemy, stats, seed):
        """[(team, model win)]: the ascent's counters, or the model search's distinct best."""
        if isinstance(setting, Settings):
            return [(team, win) for team, win, *_ in counters(space, enemy, count=args.top, restarts=args.restarts,
                                                              settings=setting, seed=seed, enemy_stats=stats)]
        evolved = model_search(space, enemy, [], evaluations=setting, enemy_stats=stats, seed=seed, catalog=catalog)
        return [evolved[i] for i in distinct([team for team, _ in evolved], range(len(evolved)), args.top)]

    results = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init) as engine, \
            open(args.output, "a") as log:
        for index, settings in trials:
            started, per_floor = time.time(), {}
            for f, (name, enemy, stats, per_card) in enumerate(floors):
                found = teams(settings, enemy, stats, args.seed + f)
                jobs = [(team, enemy, args.seed + 2 * i, per_card, 0) for i, (team, _) in enumerate(found)]
                wins = list(engine.map(_role_win, jobs)) or [0.0]
                model = [win for _, win in found] or [0.0]
                per_floor[name] = {"best": round(max(wins), 3), "mean": round(sum(wins) / len(wins), 3),
                                   "model_mean": round(sum(model) / len(model), 3),
                                   "teams": [{**describe(catalog, team), "model": round(m, 3), "engine": round(w, 3)}
                                             for (team, m), w in zip(found, wins)]}
            score = sum(v["best"] for v in per_floor.values()) / len(per_floor)
            mean = sum(v["mean"] for v in per_floor.values()) / len(per_floor)
            record = {"trial": index, "score": round(score, 3), "mean": round(mean, 3), "floors": per_floor,
                      "settings": ({k: getattr(settings, k) for k in SPACE} if isinstance(settings, Settings)
                                   else {"model_search": settings}),
                      "seconds": round(time.time() - started)}
            results.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    print("\ntrial  best   mean   settings  (best: the floors' mean of their best engine win; trial 0: defaults)")
    for r in sorted(results, key=lambda r: (-r["score"], -r["mean"])):
        print(f"{str(r['trial']):<6} {r['score']:.3f}  {r['mean']:.3f}  {r['settings']}")


if __name__ == "__main__":
    main()
