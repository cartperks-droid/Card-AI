"""Hard examples: battles where the classifier and the engine disagree most (user, 2026-10-04).

Random sampling almost never produces teams that beat huge fixed stats through abilities (the floor-105 cheese
decks), so the classifier underrates them and its guesses there swing between checkpoints. Proposing teams with the
generator does not find them: it chases the model's own opinion, so its teams are ones the model already gets right
(first hard shards, 2026-10-04: mean |model - engine| 0.004). Random candidates do not either: against huge stats
almost none win (0 of 200 at 10.9M HP), since cheese decks are specific four-card lineups. So each round searches
with the engine: --candidates evaluations (milliseconds each) of an evolution from 64 random teams leaning on
stat-ignoring cards (labels.STAT_IGNORING), whose 16 best each get 4 variants per generation (a card, a support, or
the lineup order changed). Every team evaluated is then scored by the current classifier (one batch), and the
--keep teams with the largest |model - engine| are kept, each at least two places apart from the others (one
evolution's teams are mostly one-change variants: 64 per enemy were memorised, 2026-10-05), plus --keep-random
others so the shards are not only extremes. Many enemies with few teams each, rather than the reverse.
Every --generator-every-th enemy (4) gets the annealed generator's teams instead (training.generate, sigma 4): the
engine search finds the winners the model underrates, the generator the losers it overrates (2026-10-05: 0.92 by the
model, 0.0 by the engine, borderless at floor 105).

Enemies, half each:
  - a tower floor (card_engine.tower), weighted toward the top floors (a quarter on 95-105) and hard difficulties
    (Normal 1 : Hard 1 : Extreme 2 : Hell 3 : Impossible 3), its fixed team or random cards;
  - random cards with stats HP = 10^U(2, 7.5), ATK = HP / 2 * 10^U(-0.5, 0.5).
The enemy is side B; the candidate attacks first. Candidates: each card a stat-ignoring one with probability 1/2,
else any card; borders, mutations and support tiers drawn per round (borderless, up to Crystal, or all borders;
mutations none or random; supports base or random tier, none 10% of the time).

Shards (hard_<seed>.npz, their own seed sequence) hold the label fields plus model_win, the model's win chance at
mining time; each prints the mean gap over every team evaluated and over the kept ones, and the mean best engine win
chance the search reached. The trainer repeats hard rows
(a share of each batch: train.py --mix) and scores validation's separately (val_hard).

    python -m card_engine.training.hard --shards 100 --workers 7
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import tower
from ..catalog import load_catalog
from ..mutations import MUTATION_NAMES
from ..teams import ASTRAEUS, ASTRAEUS_ARTS, SINGLE_COPY, side, spec
from .labels import AURA_TIERS, FIELDS, STORE, evaluate, next_seed, stat_ignoring_cards
from .predict import Classifier
from .train import RUN_DIR

POD_RUN = Path(__file__).resolve().parents[2] / "data" / "training_pod"  # the pod's run, as scripts/pod_sync.sh brings it down
BORDER_SETS = ((1,), (1, 2, 3), tuple(range(1, 17)))  # borderless, up to Crystal, every border


def draw_enemy(rng, catalog):
    """(enemy side, fixed stats (HP, ATK, HP multiplier applies))."""
    if rng.random() < 0.5:
        floor, level = tower.draw_floor(rng)  # the deep model rated Drago's decks 0.02 under uniform floors
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


def mutate(rng, catalog, team, borders, mutate_cards, tiered):
    """One change to a team: a card (stat-ignoring with probability 1/2), a support, or the lineup order."""
    cards = [list(entry) for entry in zip(team["cards"], team["borders"], team["mutations"], team["arts"])]
    supports = [(team["red"], team["red_tier"]), (team["blue"], team["blue_tier"])]
    move = rng.random()
    if move < 0.6:
        slot = rng.randrange(4)
        fresh = draw_candidate(rng, catalog, borders, mutate_cards, tiered)
        cards[slot] = [fresh["cards"][0], fresh["borders"][0], fresh["mutations"][0], fresh["arts"][0]]
        if any(c in SINGLE_COPY and [x[0] for x in cards].count(c) > 1 for c in SINGLE_COPY):
            return team
    elif move < 0.8:
        fresh = draw_candidate(rng, catalog, borders, mutate_cards, tiered)
        color = rng.randrange(2)
        supports[color] = ((fresh["red"], fresh["red_tier"]), (fresh["blue"], fresh["blue_tier"]))[color]
    else:
        a, b = rng.sample(range(4), 2)
        cards[a], cards[b] = cards[b], cards[a]
    return side([tuple(c) for c in cards], *supports)


def _key(team):
    return tuple(tuple(v) if isinstance(v, list) else v for v in team.values())


def search(rng, catalog, pool, enemy, per_card, seed, candidates, population=64, parents=16, children=4):
    """Engine-guided evolution toward teams that beat `enemy`; returns every team evaluated as (team, probs, engine
    win chance, exact)."""
    borders, mutate_cards, tiered = rng.choice(BORDER_SETS), rng.random() < 0.5, rng.random() < 0.5
    seen = {}

    def score(teams):
        fresh = [t for t in dict((_key(t), t) for t in teams).values() if _key(t) not in seen]
        jobs = [(spec(t, enemy), seed + len(seen) + i, per_card) for i, t in enumerate(fresh)]
        for team, (probs, exact) in zip(fresh, pool.map(_label, jobs, chunksize=8)):
            finished = probs[0] + probs[1]
            seen[_key(team)] = (team, probs, probs[0] / finished if finished > 0 else float("nan"), exact)

    alive = [draw_candidate(rng, catalog, borders, mutate_cards, tiered) for _ in range(population)]
    score(alive)
    while len(seen) < candidates:
        ranked = sorted((seen[_key(t)] for t in alive), key=lambda e: -np.nan_to_num(e[2], nan=-1.0))
        best = [team for team, *_ in ranked[:parents]]
        offspring = [mutate(rng, catalog, team, borders, mutate_cards, tiered) for team in best for _ in range(children)]
        before = len(seen)
        score(offspring)
        alive = [e[0] for e in sorted({_key(t): seen[_key(t)] for t in best + offspring}.values(),
                                      key=lambda e: -np.nan_to_num(e[2], nan=-1.0))[:population]]
        if len(seen) == before:  # nothing new to try
            break
    return list(seen.values())


def distinct(teams, ranked, keep, apart=2):
    """The first `keep` of `ranked` that differ from every one already taken in at least `apart` places (the four
    lineup slots' cards and the two supports). One evolution's teams are mostly one-change variants of each other:
    keeping all of them taught the model those few lineups by heart (probe, 2026-10-05: hard training KL 0.04,
    held-out 0.46)."""
    def places(team):
        return (*team["cards"], team["red"], team["blue"])
    taken = []
    for i in ranked:
        if len(taken) == keep:
            break
        if all(sum(a != b for a, b in zip(places(teams[i]), places(teams[j]))) >= apart for j in taken):
            taken.append(i)
    return taken


def proposals(rng, classifier, catalog, pool, enemy, fixed, per_card, seed, spaces):
    """Teams the classifier rates highest against `enemy`, from the annealed generator (training.generate), each
    played by the engine: [(team, probs, engine win chance, exact)]. The engine search finds teams the model
    underrates; these are the ones it overrates (2026-10-05: at floor 105, borderless, the deep model rated
    Kira / Time Lord Stryx / Legends / Hades 0.92; the engine, 0.0)."""
    from .generate import Settings, SlotSpace, counters, make_pool
    borders, mutations, tiers = rng.choice(BORDER_SETS), rng.choice(((0,), None)), rng.choice(((1,), None))
    key = (borders, mutations, tiers)
    if key not in spaces:  # one slot space per mask and shard (the classifier is reloaded per shard)
        spaces[key] = SlotSpace(classifier, make_pool(catalog, "all", borders=list(borders), tiers=tiers,
                                                      mutations=None if mutations is None else list(mutations)))
    settings = Settings(steps=300, noise_levels=12, sigma_max=4.0)
    teams = [team for team, *_ in counters(spaces[key], enemy, count=32, restarts=32, settings=settings,
                                           seed=seed, enemy_stats=fixed)]
    jobs = [(spec(team, enemy), seed + i, per_card) for i, team in enumerate(teams)]
    out = []
    for team, (probs, exact) in zip(teams, pool.map(_label, jobs, chunksize=4)):
        finished = probs[0] + probs[1]
        out.append((team, probs, probs[0] / finished if finished > 0 else float("nan"), exact))
    return out


def hard_shard(classifier, catalog, seed, rounds, pool, candidates=1024, keep=16, keep_random=4, generator_every=4):
    """One shard's rows: per round, the `keep` distinct searched teams the model gets most wrong plus `keep_random`
    others. Every `generator_every`-th round proposes its teams with the annealed generator instead of the engine
    search (0: never)."""
    rng = random.Random(f"hard-{seed}")
    rows, gaps_all, gaps_kept, best_found, spaces = [], [], [], [], {}
    for r in range(rounds):
        enemy, fixed = draw_enemy(rng, catalog)
        per_card = tower.engine_stats(catalog, enemy["cards"], fixed)
        if generator_every and r % generator_every == generator_every - 1:
            evaluated = proposals(rng, classifier, catalog, pool, enemy, fixed, per_card,
                                  seed * 1_000_003 + r * 4 * candidates, spaces)
        else:
            evaluated = search(rng, catalog, pool, enemy, per_card, seed * 1_000_003 + r * 4 * candidates, candidates)
        teams = [team for team, *_ in evaluated]
        engine = np.array([e for _, _, e, _ in evaluated])
        model = classifier.ally_win([(team, enemy) for team in teams], fixed)[:, 0]
        gap = np.abs(engine - model)
        ranked = [i for i in np.argsort(-np.nan_to_num(gap, nan=-1.0)) if not np.isnan(engine[i])]
        chosen = distinct(teams, ranked, keep)
        rest = [i for i in ranked if i not in set(chosen)]
        chosen += rng.sample(rest, min(keep_random, len(rest)))
        gaps_all.append(float(np.nanmean(gap)))
        gaps_kept.append(float(np.mean(gap[chosen[:keep]])) if chosen else float("nan"))
        best_found.append(float(np.nanmax(engine)))
        rows += [(spec(teams[i], enemy), (evaluated[i][1], evaluated[i][3]), fixed, float(model[i])) for i in chosen]
    arrays = {name: np.array([b[name] for b, *_ in rows], dtype=np.int16) for name in FIELDS}
    probs = np.array([res[0] for _, res, *_ in rows], dtype=np.float32)
    exact = np.array([res[1] for _, res, *_ in rows], dtype=bool)
    extra = {"fixed_side": np.ones(len(rows), dtype=np.int8),
             "fixed_stats": np.array([f[:2] for *_, f, _ in rows], dtype=np.float32),
             "fixed_hp_mult": np.array([f[2] for *_, f, _ in rows], dtype=np.int8),
             "model_win": np.array([m for *_, m in rows], dtype=np.float32)}
    return arrays, probs, exact, extra, float(np.mean(gaps_all)), float(np.nanmean(gaps_kept)), float(np.mean(best_found))


def run(shards, *, rounds=80, workers=None, checkpoint=None, out_dir=STORE, first_seed=None, device=None,
        candidates=1024, keep=16, keep_random=4, generator_every=4):
    from .flags import snapshot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {int(p.stem.split("_")[1]) for p in out_dir.glob("hard_*.npz")}
    seed = next_seed(existing, first_seed)
    catalog = load_catalog()
    # An executor, not multiprocessing.Pool: a worker the OS kills (memory pressure) raises BrokenProcessPool and
    # ends the run, so a shell loop restarts it; Pool waited forever for the lost result (2026-10-05, after shard 120).
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 1), mp_context=mp.get_context("spawn"),
                             initializer=_init) as pool:
        for done in range(1, shards + 1):
            while seed in existing:
                seed += 1
            path = checkpoint or next(p for p in (POD_RUN / "model.checkpoint", RUN_DIR / "model.checkpoint") if p.exists())
            classifier = Classifier(path, device)  # reloaded every shard: the mining follows training
            started = time.time()
            arrays, probs, exact, extra, gap_all, gap_kept, best = hard_shard(classifier, catalog, seed, rounds, pool,
                                                                              candidates, keep, keep_random,
                                                                              generator_every)
            target = out_dir / f"hard_{seed:08d}.npz"
            tmp = target.with_name(f"partial_{target.name}")
            np.savez_compressed(tmp, probs=probs, exact=exact, snapshot=np.array(snapshot(catalog)), **arrays, **extra)
            os.replace(tmp, target)
            print(json.dumps({"shard": target.name, "done": done, "of": shards, "rows": len(probs),
                              "model_step": classifier.metadata.get("step"), "seconds": round(time.time() - started),
                              "mean_gap_all": round(gap_all, 3), "mean_gap_kept": round(gap_kept, 3),
                              "mean_best_engine_win": round(best, 3)}), flush=True)
            seed += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=80, help="enemies per shard")
    parser.add_argument("--candidates", type=int, default=1024, help="teams scored per enemy")
    parser.add_argument("--keep", type=int, default=16, help="largest-disagreement teams kept per enemy, each at least "
                        "two places (card slots, supports) apart from the others")
    parser.add_argument("--generator-every", type=int, default=4,
                        help="every Nth enemy gets the annealed generator's teams (the model's overconfident ones) "
                        "instead of the engine search's (0: never)")
    parser.add_argument("--keep-random", type=int, default=4, help="other teams kept per enemy")
    parser.add_argument("--workers", type=int, help="engine labelling processes")
    parser.add_argument("--checkpoint", help="model (default: data/training_pod's, else data/training's), reloaded per shard")
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    run(args.shards, rounds=args.rounds, workers=args.workers, checkpoint=args.checkpoint, first_seed=args.first_seed,
        device=args.device, candidates=args.candidates, keep=args.keep, keep_random=args.keep_random,
        generator_every=args.generator_every)


if __name__ == "__main__":
    main()
