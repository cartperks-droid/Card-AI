"""Incomplete mode and PvP duels: a defender built blind, attacked by players who see it (user, 2026-10-06/07).

In PvP a defender does not know who will attack it, while the attacker sees the team it attacks (user, 2026-10-07:
"the defender has hidden slots ... the attacker has complete information"). Each duel (user, 2026-10-07):
  - The defender is a player at a progression: rolls log-uniform from 100k to 1B (no player has reached 400M; 1B is
    the cap) and luck log-uniform from 1x to 100x. Its pool (Progression.pool) is a random draw: each rollable
    card x border x mutation is obtained Poisson(rolls x luck / expected rolls) times, so two players at one level
    own slightly different collections. Supports, every tier, are always owned (user). The defender's one team is
    the model search's best in incomplete mode (generate.model_search, Classifier.defend_win), against the attackers
    it cannot see.
  - 32 attackers, each a player near the defender's progression, its own draw: log10 rolls the defender's plus
    N(0, ROLL_SPREAD), log10 luck plus N(0, LUCK_SPREAD), luck varying more than rolls (user); both kept in the
    defender's ranges. Each builds the model search's best counter to the defender inside its own pool, in
    complete mode.
  - The engine plays the 32 battles.
Mutations (user, 2026-10-07): a roll gets some mutation with a chance correlated with luck, 1/2,000 at luck 1 to
1/150 at luck 100 (log-linear), split evenly over the mutations (each about as likely whatever the weather, user);
a mutation's share multiplies the entry's chance.

Rows, per shard of --duels duels:
  - pvp_<seed>.npz, complete mode: every attacker against its defender, the engine's result. The attackers are the
    model's own counters, so the ones it overrates meet the engine (user: "this will also expose its weaknesses").
    They are hard examples (train's mix "hard", val_hard) and stay valid whatever model made them.
  - hidden_<generation * GENERATION_SEEDS + n>.npz, incomplete mode: the defender against the unseen attacker (side
    A, hidden_side 0), labelled with its mean win over the 32 battles. Its meaning is "strong attackers near my level
    who see me", so it moves with the model that builds the attackers: the generation is the model's step //
    PVP_GENERATION_STEPS, and the trainer reads only the newest generations (train.py --pvp-generations).

    python -m card_engine.training.incomplete --shards 10 --checkpoint data/training_10s/ema.checkpoint --workers 3
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from ..availability import roll_rarity
from ..catalog import load_catalog
from ..mutations import MUTATION_NAMES
from ..restricted import entries as restricted_entries, load as load_restricted
from ..teams import ASTRAEUS, ASTRAEUS_ARTS, spec
from .labels import FIELDS, GENERATION_SEEDS, HIDDEN_TEAM, STORE, evaluate, next_seed

ROLL_RANGE = (1e5, 1e9)  # a defender's rolls, log-uniform (user: no player has reached 400M, so 1B is the cap)
LUCK_RANGE = (1.0, 100.0)  # a defender's luck, log-uniform (user)
ROLL_SPREAD, LUCK_SPREAD = 0.25, 0.5  # attackers: sd of log10 around the defender's; luck varies more (user)
MUTATION_CHANCE = (1 / 2000, 1 / 150)  # a roll's chance of any mutation at the lowest and highest luck, log-linear (user)
MAX_COPIES = 4  # a team holds four cards
ATTACKERS = 32
PVP_GENERATION_STEPS = 50_000  # hidden rows' generation: the model's step // this


def mutation_chance(luck):
    """A roll's chance of any mutation at `luck`: 1/2,000 at 1x to 1/150 at 100x, log-linear, held outside that range."""
    t = min(1.0, max(0.0, math.log10(luck) / math.log10(LUCK_RANGE[1] / LUCK_RANGE[0])))
    low, high = MUTATION_CHANCE
    return low * (high / low) ** t


class Progression:
    """Every rollable entry (the player base's cards and borders, restricted.py; mutations on base-weather cards;
    Astraeus once per art) with its expected rolls at luck 1, unmutated, and what a player at some rolls and luck
    owns."""

    def __init__(self, catalog):
        rows, cost, mutation = [], [], []
        for card, border in restricted_entries(load_restricted(), catalog):
            arts = range(1, len(ASTRAEUS_ARTS) + 1) if card == ASTRAEUS else (0,)  # a roll gets one art
            rolls = roll_rarity(catalog, card, border) * len(arts)
            for m in range(len(MUTATION_NAMES)) if catalog.card(card).weather_id == 1 else (0,):
                for art in arts:
                    rows.append((card, border, m, art))
                    cost.append(rolls)
                    mutation.append(m > 0)
        self.entries = np.array(rows, dtype=np.int64)
        self.cost, self.mutated = np.array(cost), np.array(mutation)
        every = lambda table: [(s.id, tier) for s in table for tier in range(1, 6)]
        self.reds, self.blues = every(catalog.red_supports), every(catalog.blue_supports)

    def pool(self, rng, rolls, luck):
        """A player's collection (generate.Pool): each entry obtained Poisson(rolls x luck / expected rolls) times,
        a mutated one times its mutation's share of the mutation chance at that luck; at most MAX_COPIES kept."""
        from .generate import Pool
        each = mutation_chance(luck) / (len(MUTATION_NAMES) - 1)  # split evenly over the mutations (user)
        expected = rolls * luck / self.cost * np.where(self.mutated, each, 1.0)
        copies = np.minimum(rng.poisson(expected), MAX_COPIES)
        owned = copies > 0
        return Pool(self.entries[owned], copies[owned], self.reds, self.blues)


def draw_defender(rng):
    """(rolls, luck), each log-uniform over its range."""
    return tuple(10 ** rng.uniform(math.log10(low), math.log10(high)) for low, high in (ROLL_RANGE, LUCK_RANGE))


def draw_attacker(rng, rolls, luck):
    """(rolls, luck) near the defender's, luck the more varied, both kept in the defender's ranges."""
    near = lambda value, spread, bounds: float(np.clip(10 ** (math.log10(value) + rng.normal(0, spread)), *bounds))
    return near(rolls, ROLL_SPREAD, ROLL_RANGE), near(luck, LUCK_SPREAD, LUCK_RANGE)


def _init():
    global _CATALOG
    _CATALOG = load_catalog()


def _match(job):
    """The engine's (probs, exact) for the attacker (side A) against the defender."""
    attacker, defender, seed = job
    return evaluate(_CATALOG, spec(attacker, defender), seed)


def a_win(probs):
    """Side A's win chance among finished battles; nan if it never finishes."""
    finished = probs[0] + probs[1]
    return probs[0] / finished if finished > 0 else float("nan")


def duel(classifier, progression, pool, seed, *, attackers=ATTACKERS, defender_search=20_000, attacker_search=5_000):
    """One defender and its attackers: {"defender": team, "rolls", "luck", "battles": [(attacker, rolls, luck,
    probs, exact, model win)], "win": the defender's mean engine win, "model": the model's incomplete-mode
    estimate}, or None when a pool cannot fill a team."""
    from .generate import model_search
    rng = np.random.default_rng(seed)
    space = lambda collection: SimpleNamespace(classifier=classifier, pool=collection)
    catalog = load_catalog()

    def best(collection, enemy, search_seed, **kind):
        if collection.copies.sum() < 4:
            return None
        found = model_search(space(collection), enemy, [], seed=search_seed, catalog=catalog, **kind)
        return found[0][0] if found else None

    rolls, luck = draw_defender(rng)
    defender = best(progression.pool(rng, rolls, luck), HIDDEN_TEAM, seed, evaluations=defender_search,
                    role="defend", incomplete=True)
    if defender is None:
        return None
    teams = []
    for i in range(attackers):
        a_rolls, a_luck = draw_attacker(rng, rolls, luck)
        team = best(progression.pool(rng, a_rolls, a_luck), defender, seed * 1_000 + i, evaluations=attacker_search,
                    role="attack")
        if team is not None:
            teams.append((team, a_rolls, a_luck))
    if not teams:
        return None
    results = list(pool.map(_match, [(team, defender, seed * 1_000 + i) for i, (team, *_) in enumerate(teams)]))
    model = classifier.win_a([spec(team, defender) for team, *_ in teams])
    battles = [(team, r, l, np.asarray(probs, dtype=np.float32), bool(exact), float(m))
               for (team, r, l), (probs, exact), m in zip(teams, results, model)]
    wins = [1 - a_win(b[3]) for b in battles if not np.isnan(a_win(b[3]))]
    return {"defender": defender, "rolls": rolls, "luck": luck, "battles": battles,
            "win": float(np.mean(wins)) if wins else float("nan"), "model": float(classifier.defend_win([defender])[0])}


def pvp_shard(classifier, progression, pool, seed, duels, **search):
    """A shard's rows from `duels` duels: (complete rows, incomplete rows, summary)."""
    found = [d for d in (duel(classifier, progression, pool, seed * 1_000 + i, **search) for i in range(duels))
             if d is not None and not np.isnan(d["win"])]
    battles = [(spec(team, d["defender"]), probs, exact, m) for d in found for team, _, _, probs, exact, m in d["battles"]]
    complete = ({name: np.array([b[name] for b, *_ in battles], dtype=np.int16) for name in FIELDS},
                np.array([p for _, p, _, _ in battles], dtype=np.float32), np.array([e for _, _, e, _ in battles], dtype=bool),
                {"model_win": np.array([m for *_, m in battles], dtype=np.float32)})
    hidden = ({name: np.array([spec(HIDDEN_TEAM, d["defender"])[name] for d in found], dtype=np.int16) for name in FIELDS},
              np.array([[1 - d["win"], d["win"], 0.0, 0.0] for d in found], dtype=np.float32),  # side A: the hidden attacker
              np.zeros(len(found), dtype=bool),
              {"hidden_side": np.zeros(len(found), dtype=np.int8),
               "model_win": np.array([d["model"] for d in found], dtype=np.float32),
               "rolls": np.array([d["rolls"] for d in found]), "luck": np.array([d["luck"] for d in found])})
    gap = lambda pairs: round(float(np.mean([abs(a - b) for a, b in pairs])), 3) if pairs else None
    summary = {"duels": len(found), "pvp_rows": len(battles),
               "defend_gap": gap([(d["win"], d["model"]) for d in found]),
               "attack_gap": gap([(a_win(p), m) for _, p, _, m in battles if not np.isnan(a_win(p))]),
               "attack_model": round(float(np.mean([m for *_, m in battles])), 3) if battles else None,
               "attack_engine": round(float(np.nanmean([a_win(p) for _, p, _, _ in battles])), 3) if battles else None}
    return complete, hidden, summary


def main():
    from .flags import snapshot
    from .predict import Classifier
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="the model that builds the defenders and the attackers")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, help="engine processes")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--duels", type=int, default=8, help="duels per shard")
    parser.add_argument("--attackers", type=int, default=ATTACKERS, help="attackers per defender")
    parser.add_argument("--defender-search", type=int, default=20_000, help="teams the model scores for the defender")
    parser.add_argument("--attacker-search", type=int, default=5_000, help="teams the model scores per attacker")
    args = parser.parse_args()
    classifier = Classifier(args.checkpoint, args.device)
    progression = Progression(load_catalog())
    generation = int(classifier.metadata.get("step") or 0) // PVP_GENERATION_STEPS
    STORE.mkdir(parents=True, exist_ok=True)
    first = generation * GENERATION_SEEDS
    existing = {int(p.stem.split("_")[1]) for kind in ("hidden", "pvp") for p in STORE.glob(f"{kind}_*.npz")}
    seed = next_seed({s for s in existing if s < first + GENERATION_SEEDS}, first)
    with ProcessPoolExecutor(args.workers or max(1, (os.cpu_count() or 2) - 1), mp_context=mp.get_context("spawn"),
                             initializer=_init) as pool:
        for done in range(1, args.shards + 1):
            started = time.time()
            complete, hidden, summary = pvp_shard(classifier, progression, pool, seed, args.duels, attackers=args.attackers,
                                                   defender_search=args.defender_search, attacker_search=args.attacker_search)
            for kind, (arrays, probs, exact, extra) in (("pvp", complete), ("hidden", hidden)):
                if not len(probs):
                    continue
                target = STORE / f"{kind}_{seed:08d}.npz"
                tmp = target.with_name(f"partial_{target.name}")
                np.savez_compressed(tmp, probs=probs, exact=exact, snapshot=np.array(snapshot()), **arrays, **extra)
                os.replace(tmp, target)
            print(json.dumps({"seed": seed, "done": done, "of": args.shards, "generation": generation,
                              "model_step": classifier.metadata.get("step"), **summary,
                              "seconds": round(time.time() - started)}), flush=True)
            seed += 1


if __name__ == "__main__":
    main()
