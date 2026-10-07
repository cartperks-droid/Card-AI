"""Hard examples: battles where the classifier and the engine disagree most (user, 2026-10-04).

Random sampling almost never produces teams that beat huge fixed stats through abilities (the floor-105 cheese
decks), so the classifier underrates them and its guesses there swing between checkpoints. Proposing teams with the
generator does not find them: it chases the model's own opinion, so its teams are ones the model already gets right
(first hard shards, 2026-10-04: mean |model - engine| 0.004). Random candidates do not either: against huge stats
almost none win (0 of 200 at 10.9M HP), since cheese decks are specific four-card lineups. So each round searches
with the engine: --candidates battles (milliseconds each) of an evolution from 64 random teams, whose 16 best each get
4 variants per generation (a card, a support, or the lineup order changed), with the stat gap annealed (user,
2026-10-06: cheese should be discovered, not named): the enemy starts at the fraction of its stats where random teams
sometimes win and doubles whenever the best 16 win half their battles, so earlier gaps leave a signal to follow. From
uniformly random teams (--prior 0) it climbed floor 105 Impossible's 14 levels in about 3,400 battles and found an exact
win there (Parallax / Legends / Sarimanok / Robin Hood, Dinosaur King, Fate). --prior (0.5) still draws that share of
cards from labels.STAT_IGNORING, a prior the user means to drop. Every battle played is a label at its own stats. Every team evaluated is then scored by the current classifier (one batch), and the
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


def draw_enemy(rng, catalog, floor=None):
    """(enemy side, fixed stats (HP, ATK, HP multiplier applies)). floor: (floor, difficulty) for every enemy, its fixed
    team or random enemies on a floor without one (2026-10-07: floor 105 Impossible fields part of the cheese deck
    itself, Judgment Day, so its wins come from cross-side interactions only its own battles teach)."""
    if floor is not None:
        level = tower.difficulty(floor[1])
        team, stats = tower.fixed_team(catalog, floor[0]), tower.stats(floor[0], level)
    elif rng.random() < 0.5:
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


PRIOR = 0.5  # chance a drawn card comes from labels.STAT_IGNORING (a prior the user means to drop: --prior 0)


def draw_candidate(rng, catalog, borders, mutate, tiered, prior=PRIOR):
    """One attacking team: each card from labels.STAT_IGNORING with probability `prior`, else any card."""
    ignoring, every = stat_ignoring_cards(catalog), [c.id for c in catalog.cards]
    cards = []
    for _ in range(4):
        card = rng.choice(ignoring if rng.random() < prior else every)
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
    battle, seed, per_card, *fixed_side = job
    return evaluate(_CATALOG, battle, seed, fixed=(fixed_side[0] if fixed_side else 1, per_card))


_CATALOG = None


def _init():
    global _CATALOG
    _CATALOG = load_catalog()


def mutate(rng, catalog, team, borders, mutate_cards, tiered, prior=PRIOR):
    """One change to a team: a card (stat-ignoring with probability 1/2), a support, or the lineup order."""
    cards = [list(entry) for entry in zip(team["cards"], team["borders"], team["mutations"], team["arts"])]
    supports = [(team["red"], team["red_tier"]), (team["blue"], team["blue_tier"])]
    move = rng.random()
    if move < 0.6:
        slot = rng.randrange(4)
        fresh = draw_candidate(rng, catalog, borders, mutate_cards, tiered, prior)
        cards[slot] = [fresh["cards"][0], fresh["borders"][0], fresh["mutations"][0], fresh["arts"][0]]
        if any(c in SINGLE_COPY and [x[0] for x in cards].count(c) > 1 for c in SINGLE_COPY):
            return team
    elif move < 0.8:
        fresh = draw_candidate(rng, catalog, borders, mutate_cards, tiered, prior)
        color = rng.randrange(2)
        supports[color] = ((fresh["red"], fresh["red_tier"]), (fresh["blue"], fresh["blue_tier"]))[color]
    else:
        a, b = rng.sample(range(4), 2)
        cards[a], cards[b] = cards[b], cards[a]
    return side([tuple(c) for c in cards], *supports)


def _key(team):
    return tuple(tuple(v) if isinstance(v, list) else v for v in team.values())


def start_gap(catalog, teams, stats):
    """The fraction of the enemy's stats a search starts at: about the teams' own median HP, where random teams
    sometimes win, so the evolution has a signal (2026-10-06: at the full gap random teams almost never win)."""
    from .labels import _base_stats
    base = _base_stats(catalog)
    hp = np.median([np.exp(np.mean([np.log(base[c, b, m][0]) for c, b, m in zip(t["cards"], t["borders"], t["mutations"])]))
                    for t in teams])
    return float(min(1.0, 2.0 ** np.floor(np.log2(max(hp, 1.0) / stats[0])))) if stats[0] > hp else 1.0


def search(rng, catalog, pool, enemy, stats, seed, candidates, population=64, parents=16, children=4, prior=PRIOR):
    """Engine-guided evolution toward teams that beat `enemy` at its fixed stats (HP, ATK, HP multiplier applies),
    with the gap annealed (user, 2026-10-06: cheese should be discovered, not named): the enemy starts at the
    fraction of its stats where random teams sometimes win (start_gap), and doubles toward the full stats whenever
    the best `parents` teams win half their battles on average, so the cards that keep winning as the gap grows are
    the ones the engine selected. Returns every battle played as (team, probs, engine win chance, exact, stats)."""
    borders, mutate_cards, tiered = rng.choice(BORDER_SETS), rng.random() < 0.5, rng.random() < 0.5
    seen = {}
    alive = [draw_candidate(rng, catalog, borders, mutate_cards, tiered, prior) for _ in range(population)]
    gap = start_gap(catalog, alive, stats)

    def at(fraction):
        return (stats[0] * fraction, stats[1] * fraction, stats[2])

    def score(teams):
        now = at(gap)
        per_card = tower.engine_stats(catalog, enemy["cards"], now)
        fresh = [t for t in dict((_key(t), t) for t in teams).values() if (_key(t), gap) not in seen]
        jobs = [(spec(t, enemy), seed + len(seen) + i, per_card) for i, t in enumerate(fresh)]
        for team, (probs, exact) in zip(fresh, pool.map(_label, jobs, chunksize=8)):
            finished = probs[0] + probs[1]
            seen[(_key(team), gap)] = (team, probs, probs[0] / finished if finished > 0 else float("nan"), exact, now)

    def ranked(teams):
        return sorted((seen[(_key(t), gap)] for t in teams), key=lambda e: -np.nan_to_num(e[2], nan=-1.0))

    score(alive)
    while len(seen) < candidates:
        top = ranked(alive)[:parents]
        if gap < 1 and np.mean([np.nan_to_num(e[2]) for e in top]) >= 0.5:  # this gap is solved: double it
            gap = min(1.0, gap * 2)
            alive = [e[0] for e in top]
            score(alive)
            continue
        best = [team for team, *_ in top]
        offspring = [mutate(rng, catalog, team, borders, mutate_cards, tiered, prior) for team in best for _ in range(children)]
        before = len(seen)
        score(offspring)
        alive = [e[0] for e in ranked(list({_key(t): t for t in best + offspring}.values()))[:population]]
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
        out.append((team, probs, probs[0] / finished if finished > 0 else float("nan"), exact, fixed))
    return out


def hard_shard(classifier, catalog, seed, rounds, pool, candidates=1024, keep=16, keep_random=4, generator_every=4,
               prior=PRIOR, floor=None):
    """One shard's rows: per round, the `keep` distinct searched teams the model gets most wrong plus `keep_random`
    others. Every `generator_every`-th round proposes its teams with the annealed generator instead of the engine
    search (0: never). With no classifier (--select engine) the engine's own choice is kept instead: the `keep`
    distinct teams that won most, those at the full stats first, plus `keep_random` battles from any gap, so no
    model has to score the search (2026-10-06: 12,000 teams per enemy through 18 layers on the Mac's CPU)."""
    rng = random.Random(f"hard-{seed}")
    rows, gaps_all, gaps_kept, best_found, spaces = [], [], [], [], {}
    for r in range(rounds):
        enemy, fixed = draw_enemy(rng, catalog, floor)
        per_card = tower.engine_stats(catalog, enemy["cards"], fixed)
        if classifier is not None and generator_every and r % generator_every == generator_every - 1:
            evaluated = proposals(rng, classifier, catalog, pool, enemy, fixed, per_card,
                                  seed * 1_000_003 + r * 4 * candidates, spaces)
        else:
            evaluated = search(rng, catalog, pool, enemy, fixed, seed * 1_000_003 + r * 4 * candidates, candidates,
                               prior=prior)
        teams = [team for team, *_ in evaluated]
        engine = np.array([e for _, _, e, _, _ in evaluated])
        model = np.full(len(teams), np.nan)
        if classifier is not None:
            for stats in {e[4] for e in evaluated}:  # each gap the search played, scored at its own stats
                where = [i for i, e in enumerate(evaluated) if e[4] == stats]
                model[where] = classifier.ally_win([(teams[i], enemy) for i in where], stats)[:, 0]
        gap = np.abs(engine - model)
        if classifier is not None:
            ranked = [i for i in np.argsort(-np.nan_to_num(gap, nan=-1.0)) if not np.isnan(engine[i])]
        else:  # the engine's selection: the full-stats winners first, then by win chance at any gap
            ranked = sorted((i for i in range(len(teams)) if not np.isnan(engine[i])),
                            key=lambda i: (evaluated[i][4] != fixed, -engine[i]))
        chosen = distinct(teams, ranked, keep)
        rest = [i for i in ranked if i not in set(chosen)]
        chosen += rng.sample(rest, min(keep_random, len(rest)))
        gaps_all.append(float(np.nanmean(gap)) if classifier is not None else float("nan"))
        gaps_kept.append(float(np.mean(gap[chosen[:keep]])) if chosen and classifier is not None else float("nan"))
        best_found.append(float(np.nanmax([e for e, x in zip(engine, evaluated) if x[4] == fixed] or [np.nan])))
        rows += [(spec(teams[i], enemy), (evaluated[i][1], evaluated[i][3]), evaluated[i][4], float(model[i]), 1)
                 for i in chosen]
        # The same battles with the roles reversed, the enemy attacking first (user, 2026-10-07: the player always
        # starts in the tower, but the reversed battles teach a near-symmetry; at step 48,000 the model gave
        # Parallax/JD/JD/Robin Hood 0.54 attacking and 0.01 defending at floor 105, the engine 0.316 both). Each is
        # the engine's own label for that order, never 1 - p: who attacks first and who loses a draw both change.
        swapped = [(spec(enemy, teams[i]), seed * 1_000_003 + r * 4 * candidates + candidates + j,
                    tower.engine_stats(catalog, enemy["cards"], evaluated[i][4]), 0) for j, i in enumerate(chosen)]
        rows += [(battle, result, evaluated[i][4], float("nan"), 0)
                 for (battle, *_), result, i in zip(swapped, pool.map(_label, swapped, chunksize=8), chosen)]
    arrays = {name: np.array([b[name] for b, *_ in rows], dtype=np.int16) for name in FIELDS}
    probs = np.array([res[0] for _, res, *_ in rows], dtype=np.float32)
    exact = np.array([res[1] for _, res, *_ in rows], dtype=bool)
    extra = {"fixed_side": np.array([side for *_, side in rows], dtype=np.int8),
             "fixed_stats": np.array([f[:2] for _, _, f, _, _ in rows], dtype=np.float32),
             "fixed_hp_mult": np.array([f[2] for _, _, f, _, _ in rows], dtype=np.int8),
             "model_win": np.array([m for _, _, _, m, _ in rows], dtype=np.float32)}
    reached = float(np.mean([not np.isnan(b) for b in best_found])) if best_found else float("nan")
    best = float(np.nanmean(best_found)) if reached else float("nan")
    return arrays, probs, exact, extra, float(np.mean(gaps_all)), float(np.nanmean(gaps_kept)), (best, reached)


def run(shards, *, rounds=40, workers=None, checkpoint=None, out_dir=STORE, first_seed=None, device=None,
        candidates=12000, keep=16, keep_random=4, generator_every=4, prior=PRIOR, select="model", floor=None):
    from .flags import snapshot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = "hard" if select == "model" else "found"  # the engine's own picks are their own kind (train.py --mix)
    existing = {int(p.stem.split("_")[1]) for p in out_dir.glob(f"{kind}_*.npz")}
    seed = next_seed(existing, first_seed)
    catalog = load_catalog()
    # An executor, not multiprocessing.Pool: a worker the OS kills (memory pressure) raises BrokenProcessPool and
    # ends the run, so a shell loop restarts it; Pool waited forever for the lost result (2026-10-05, after shard 120).
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 1), mp_context=mp.get_context("spawn"),
                             initializer=_init) as pool:
        for done in range(1, shards + 1):
            while seed in existing:
                seed += 1
            if select == "model":
                path = checkpoint or next(p for p in (POD_RUN / "model.checkpoint", RUN_DIR / "model.checkpoint") if p.exists())
                classifier = Classifier(path, device)  # reloaded every shard: the mining follows training
            else:
                classifier = None
            started = time.time()
            arrays, probs, exact, extra, gap_all, gap_kept, best = hard_shard(classifier, catalog, seed, rounds, pool,
                                                                              candidates, keep, keep_random,
                                                                              generator_every, prior, floor)
            target = out_dir / f"{kind}_{seed:08d}.npz"
            tmp = target.with_name(f"partial_{target.name}")
            np.savez_compressed(tmp, probs=probs, exact=exact, snapshot=np.array(snapshot(catalog)), **arrays, **extra)
            os.replace(tmp, target)
            print(json.dumps({"shard": target.name, "done": done, "of": shards, "rows": len(probs),
                              "model_step": classifier.metadata.get("step") if classifier else None, "seconds": round(time.time() - started),
                              "mean_gap_all": round(gap_all, 3), "mean_gap_kept": round(gap_kept, 3),
                              "mean_best_engine_win": round(best[0], 3), "reached_full_stats": round(best[1], 3)}),
                  flush=True)
            seed += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=40, help="enemies per shard")
    parser.add_argument("--candidates", type=int, default=12000, help="engine battles per enemy (the annealed search climbs "
                        "floor 105 Impossible from random teams in about 3,400, 2026-10-06)")
    parser.add_argument("--keep", type=int, help="largest-disagreement teams kept per enemy, each at least "
                        "two places (card slots, supports) apart from the others")
    parser.add_argument("--generator-every", type=int, default=4,
                        help="every Nth enemy gets the annealed generator's teams (the model's overconfident ones) "
                        "instead of the engine search's (0: never)")
    parser.add_argument("--keep-random", type=int, help="other teams kept per enemy (4; 200 with --select engine: "
                        "every battle played is a label, and the gap ladder's random ones cost nothing more)")
    parser.add_argument("--workers", type=int, help="engine labelling processes")
    parser.add_argument("--checkpoint", help="model (default: data/training_pod's, else data/training's), reloaded per shard")
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--select", choices=("model", "engine"), default="model",
                        help="keep the battles the model gets most wrong (model), or the engine's own winners and "
                        "random battles from every gap, with no model to score (engine; no --checkpoint needed)")
    parser.add_argument("--prior", type=float, default=PRIOR,
                        help="chance a drawn card comes from the stat-ignoring list (0: discovery from the whole pool)")
    parser.add_argument("--tower", nargs=2, metavar=("FLOOR", "DIFFICULTY"),
                        help="every enemy on this tower floor and difficulty (e.g. 105 Impossible), not drawn")
    args = parser.parse_args()
    floor = (int(args.tower[0]), args.tower[1]) if args.tower else None
    if floor is not None:
        tower.difficulty(floor[1])  # an unknown difficulty fails here, not after the first shard's search
    engine = args.select == "engine"  # the engine's picks: keep many (the model mode keeps few: near-duplicates were memorised)
    args.keep = args.keep if args.keep is not None else 32 if engine else 16
    args.keep_random = args.keep_random if args.keep_random is not None else 200 if engine else 4
    run(args.shards, rounds=args.rounds, workers=args.workers, checkpoint=args.checkpoint, first_seed=args.first_seed,
        device=args.device, candidates=args.candidates, keep=args.keep, keep_random=args.keep_random,
        generator_every=args.generator_every, prior=args.prior, select=args.select, floor=floor)


if __name__ == "__main__":
    main()
