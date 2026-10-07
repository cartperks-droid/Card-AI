"""Depths, from DaddyDrago's engine (engine/depths.ts and simulation.ts at sim_js/ENGINE_COMMIT), played by our engine.

A run climbs floors until its first loss; every floor is a fresh battle of the player's team, attacking first, against
four enemies the game draws. Each enemy is drawn, duplicates allowed, from the floor's pool by weight (his weather
weights: Storm and Snow 0.5, Aurora 0.3, ..., 1 for base cards). A card is in the pool once its ATK x HP (base,
borderless) is below the floor's budget, 3000 + floor^2.75 x 40, unless the game excludes it (his eligibility: no
unobtainable, expiring, boss or holiday cards, nor Vampire Lord, Parallax or Samurai) or the player banned it (up to
14). Every enemy has power ceil(sqrt(2 x budget)), ATK power / 2 and HP power times its HP multiplier; enemies are
borderless, unmutated and without supports. Hard depths (user, 2026-10-07: the mode of most interest) keeps the
floor's pool but takes the stats of floor x 10.

A floor's enemies are independent of the others, so a team reaches floor F with probability prod_{f <= F} p(f),
p(f) its mean win chance over the floor's draws. The curve is sampled at a geometric grid of floors (p held between
grid floors); expected floors cleared is the sum of the survival curve.

    python -m card_engine.depths run --ally Parallax "Judgment Day" "Judgment Day" "Robin Hood" --ally-red Fate --checkpoint data/training_10s/ema.checkpoint
    python -m card_engine.depths run --ally ... --checkpoint ... --simulate      # the engine's curve too
    python -m card_engine.depths search --checkpoint data/training_10s/ema.checkpoint --pool own
"""

import argparse
import json
import math
import random
import sys

import numpy as np

from .teams import ASTRAEUS, ASTRAEUS_ARTS, side

HARD_STATS = 10  # hard depths: the stats of floor x 10 (his simulation.ts hardMode)
MAX_BANS = 14
FLOOR_RANGE = (1, 10_000)  # training draws: floors log-uniform over this range (hard depths ~5,000 is about tower
# 105 Impossible's stats; the pool is complete from about floor 150)
HARD_SHARE = 0.8  # training draws that are hard depths (user: of most interest); the rest are normal depths


def budget(floor):
    return 3000 + floor ** 2.75 * 40


def power(floor):
    return math.ceil(math.sqrt(budget(floor) * 2))


def stats(floor, hard=True):
    """The floor's fixed enemy stats: (HP before the card's HP multiplier, ATK, HP multiplier applies)."""
    p = power(max(1, int(floor)) * (HARD_STATS if hard else 1))
    return float(p), p / 2, True


_POOL = {}


def pool(catalog):
    """[(card id, weight, ATK x HP)] for every card Depths can field (his pool, via the worker), and the cards of his
    pool our catalog lacks."""
    if id(catalog) not in _POOL:
        from .simulator import drago
        reply = drago.worker().request({"op": "depths"})
        ours = {his: card for card, his in drago.names(catalog)[0].items()}
        entries = [(ours[name], weight, threshold) for name, weight, threshold in reply["pool"] if name in ours]
        missing = [name for name, *_ in reply["pool"] if name not in ours]
        _POOL[id(catalog)] = entries, missing
    return _POOL[id(catalog)]


def floor_pool(catalog, floor, bans=()):
    """(card ids, weights) unlocked at `floor` (the real floor, also in hard depths), without the banned cards."""
    entries = [(card, weight) for card, weight, threshold in pool(catalog)[0]
               if threshold < budget(floor) and card not in set(bans)]
    return [card for card, _ in entries], [weight for _, weight in entries]


def draw_team(rng, catalog, floor, bans=()):
    """Four enemies drawn by weight from the floor's pool (duplicates allowed), as a side; Astraeus with a random art
    (his engine draws its Constellar ability)."""
    cards, weights = floor_pool(catalog, floor, bans)
    if not cards:
        raise SystemExit(f"Depths floor {floor} has no enemies left to draw (bans: {len(bans)})")
    picks = rng.choices(cards, weights=weights, k=4)
    return side([(card, 1, 0, rng.randint(1, len(ASTRAEUS_ARTS)) if card == ASTRAEUS else 0) for card in picks])


def draw_floor(rng):
    """A training draw: (floor, hard), floors log-uniform over FLOOR_RANGE, hard depths HARD_SHARE of the time."""
    low, high = FLOOR_RANGE
    return int(round(10 ** rng.uniform(math.log10(low), math.log10(high)))), rng.random() < HARD_SHARE


def grid(cap, points):
    """Geometric floors from 1 to cap, distinct."""
    return sorted({max(1, round(cap ** (i / (points - 1)))) for i in range(points)} | {1, cap})


def survival(floors, wins, cap):
    """(expected floors cleared, survival through each grid floor's span, median death floor: the first floor at
    which the chance to have survived falls below a half). p(f) is the win chance at the nearest grid floor at or
    below f, up to cap."""
    expected, alive, curve, median = 0.0, 1.0, [], None
    for start, end, p in zip(floors, [*floors[1:], cap + 1], wins):
        length, p = end - start, min(max(float(p), 0.0), 1.0)
        if median is None and alive * p ** length < 0.5:
            k = 1 if p <= 0 or alive * p < 0.5 else int(math.log(0.5 / alive) / math.log(p)) + 1
            median = start + min(k, length) - 1
        expected += alive * (length if p >= 1.0 else p * (1 - p ** length) / (1 - p))
        alive *= p ** length
        curve.append(alive)
    return expected, curve, median


def enemies(catalog, floors, samples, bans=(), seed=1):
    """{floor: [enemy sides]}: a fixed draw per floor, so teams are compared on the same enemies."""
    rng = random.Random(f"depths-{seed}")
    return {floor: [draw_team(rng, catalog, floor, bans) for _ in range(samples)] for floor in floors}


def model_wins(classifier, teams, drawn, hard=True):
    """[teams, floors]: the model's mean win chance per grid floor, attacking first against the floor's draws."""
    from .teams import spec
    out = np.zeros((len(teams), len(drawn)))
    for j, (floor, foes) in enumerate(drawn.items()):
        wins = classifier.win_a([spec(team, foe) for team in teams for foe in foes], fixed=(1, stats(floor, hard)))
        out[:, j] = wins.reshape(len(teams), len(foes)).mean(1)
    return out


def engine_wins(catalog, team, drawn, hard=True, workers=None, seed=1):
    """[floors]: the engine's mean win chance per grid floor against the same draws."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from . import tower
    from .training.counter import _init
    from .training.generate import _role_win
    jobs, owners = [], []
    for j, (floor, foes) in enumerate(drawn.items()):
        for foe in foes:
            jobs.append((team, foe, seed + 2 * len(jobs), tower.engine_stats(catalog, foe["cards"], stats(floor, hard)), 0))
            owners.append(j)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init) as pool:
        wins = list(pool.map(_role_win, jobs, chunksize=4))
    out = np.zeros(len(drawn))
    for j, win in zip(owners, wins):
        out[j] += win
    return out / np.bincount(owners, minlength=len(drawn))


def add_arguments(parser):
    parser.add_argument("--normal", action="store_true", help="normal depths (default: hard depths, stats of floor x 10)")
    parser.add_argument("--bans", nargs="+", default=[], metavar="CARD", help=f"the player's Depth bans (up to {MAX_BANS})")
    parser.add_argument("--cap", type=int, default=FLOOR_RANGE[1], help="last floor of the curve")
    parser.add_argument("--points", type=int, default=24, help="grid floors sampled")
    parser.add_argument("--samples", type=int, default=32, help="enemy draws per grid floor")
    parser.add_argument("--seed", type=int, default=1)


def parse_bans(catalog, names):
    from .deck import _match
    if len(names) > MAX_BANS:
        raise SystemExit(f"At most {MAX_BANS} Depth bans")
    return [_match(name, [(c.id, c.name) for c in catalog.cards], "Card")[0] for name in names]


def main():
    from .catalog import load_catalog
    from .teams import describe, parse_side
    from .training.predict import Classifier
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="a team's depth curve")
    run.add_argument("--ally", nargs=4, required=True, metavar="CARD", help="four cards, Name[@Border][/Mutation]")
    run.add_argument("--ally-red")
    run.add_argument("--ally-blue")
    run.add_argument("--simulate", action="store_true", help="also the engine's curve, on the same draws")
    run.add_argument("--workers", type=int)
    search = commands.add_parser("search", help="the model search for the team that goes deepest, engine-verified")
    search.add_argument("--pool", choices=("own", "custom", "restricted", "all"), default="own")
    search.add_argument("--borders", nargs="+", default=["none"])
    search.add_argument("--mutations", nargs="+", default=["None"])
    search.add_argument("--support-tiers", nargs="+", default=["base"])
    search.add_argument("--evaluations", type=int, default=5_000, help="teams the model scores")
    search.add_argument("--top", type=int, default=4, help="best distinct teams the engine plays")
    search.add_argument("--workers", type=int)
    for command in (run, search):
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device")
        add_arguments(command)
    args = parser.parse_args()
    catalog = load_catalog()
    hard = not args.normal
    bans = parse_bans(catalog, args.bans)
    floors = grid(args.cap, args.points)
    drawn = enemies(catalog, floors, args.samples, bans, args.seed)
    classifier = Classifier(args.checkpoint, args.device)
    missing = pool(catalog)[1]
    if missing:
        print(f"his Depths pool has cards our catalog lacks (never drawn): {', '.join(missing)}", file=sys.stderr)

    def report(team, model, engine=None):
        result = {"team": describe(catalog, team), "mode": "hard" if hard else "normal"}
        for name, wins in (("model", model), ("engine", engine)):
            if wins is None:
                continue
            expected, curve, median = survival(floors, wins, args.cap)
            result[name] = {"expected_floors": round(expected, 1), "median_death_floor": median,
                            "floors": {f: [round(float(w), 3), round(c, 4)] for f, w, c in zip(floors, wins, curve)}}
        print(json.dumps(result), flush=True)

    if args.command == "run":
        team = parse_side(catalog, args.ally, args.ally_red, args.ally_blue)
        model = model_wins(classifier, [team], drawn, hard)[0]
        report(team, model, engine_wins(catalog, team, drawn, hard, args.workers, args.seed) if args.simulate else None)
        print("floors: grid floor -> [mean win chance there, chance to have survived through the floors it covers]")
        return
    from types import SimpleNamespace
    from .training.generate import _team, evolve, make_pool, masks
    from .training.hard import distinct
    borders, mutations, tiers = masks(args.borders, args.mutations, args.support_tiers)
    space = SimpleNamespace(classifier=classifier, pool=make_pool(catalog, args.pool, borders=borders, mutations=mutations,
                                                                  tiers=tiers))

    def score(members, done):
        teams = [_team(space, *m) for m in members]
        return [survival(floors, wins, args.cap)[0] for wins in model_wins(classifier, teams, drawn, hard)]

    seen = evolve(space, [], score, evaluations=args.evaluations, seed=args.seed, prior=0.0, catalog=catalog,
                  population=128, parents=32, children=4)
    ranked = sorted(seen, key=lambda m: -seen[m])
    teams = [_team(space, *m) for m in ranked]
    for i in distinct(teams, range(len(teams)), args.top):
        team = teams[i]
        report(team, model_wins(classifier, [team], drawn, hard)[0], engine_wins(catalog, team, drawn, hard, args.workers, args.seed))


if __name__ == "__main__":
    main()
