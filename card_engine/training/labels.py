"""Battle labels for the general win predictor: random matchups scored by DaddyDrago's battle rules.

Every battle is searched best-first over the engine's chance points (largest probabilities first, under a node
budget); small or leftover branches are resolved by pooled playouts, as many as reducing the variance needs
(sim_c: the C port of his engine and of sim_js/search.ts, with identical answers; his TypeScript builds the start). Battles with no estimated probability are exact and kept in the tablebase; estimated ones are
recomputed whenever they are labelled again. Side A always initiates; a draw counts as A's loss (user).
Each shard records the rules snapshot (training.flags), so rows go stale when the rules behind them change.

Fixed-stat battles (--fixed; user, 2026-10-04): battle modes where every enemy card has the same stats, borders
ignored; the general problem, not one mode (user). One side, A or B at random, is borderless and starts all four
cards at one (HP, ATK): the opponent's geometric-mean stats times a level, with the HP/ATK balance moved by
10^U(-0.5, 0.5). A third are tower floors instead (2026-10-06: the cheese decks' battles, a floor's fixed team,
stats and HP multiplier, had appeared only in hard examples): side B is the floor (tower.draw_floor, weighted toward
the top and the hardest difficulties) and each of side A's cards leans on stat-ignoring abilities as below. Of the
rest, half draw the level from 10^U(-2, 4); the other half are big gaps, 10^U(1, 4.5), where
each of the weaker side's cards is, with probability 1/2, one whose ability ignores raw stats (STAT_IGNORING:
damage scaled to the enemy's HP, kills, revives, shared damage), so that wins against huge stats appear at all (user,
2026-10-04: floor 105 Impossible is about 2,700x a borderless deck). The first fixed shards spread the level only
10^U(-1.5, 1.5); they stay valid. Rows store fixed_side, fixed_stats (HP, ATK) and
fixed_hp_mult (set on tower floors whose HP takes each card's multiplier). They go to fixed_<seed>.npz, never to the
tablebase (its key has no stats); trainers that predate them only read shard_*.npz.
"""

import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..catalog import load_catalog
from ..mutations import MUTATION_NAMES
from ..simulator import drago, kernel
from .. import tower
from ..teams import ASTRAEUS, ASTRAEUS_ARTS, SINGLE_COPY
from .tablebase import Tablebase

ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "data" / "labels"
FIELDS = ("cards", "borders", "mutations", "arts", "red", "red_tier", "blue", "blue_tier")
# fixed_side: -1, or the side whose cards start at fixed_stats (HP, ATK); fixed_hp_mult: HP times each card's multiplier.
# hidden_side: -1, or the side the model cannot see (incomplete mode, training.incomplete): its fields hold
# HIDDEN_TEAM, and the target is the visible team's mean result against a field of strong teams.
BATTLE_FIELDS = ("fixed_side", "fixed_stats", "fixed_hp_mult", "hidden_side")


AURA_TIERS = tuple(drago.AURA_BORDERS)  # support cards: Base, Platinum, Crystal, Ruby, Galaxy


def random_spec(rng, catalog, cards=None, none_support=0.1, mutation_rate=0.5, matched=0.7):
    """One random 4v4 matchup: any card (Astraeus with a random art), border, eligible mutation, and support tier.

    With probability `matched`, every border sits within two rarity ranks of a shared level, so the
    outcome depends on the cards rather than on a border gap; the rest draw borders uniformly.
    """
    spec = {name: [] for name in FIELDS}
    ranked = sorted(range(1, 17), key=lambda b: (catalog.border(b).rarity, b))
    level = rng.randrange(16) if rng.random() < matched else None
    border = lambda: rng.randint(1, 16) if level is None else ranked[min(15, max(0, level + rng.randint(-2, 2)))]
    cards = cards or tuple(card.id for card in catalog.cards)
    for _side in (0, 1):
        row = {name: [] for name in ("cards", "borders", "mutations", "arts")}
        for _slot in range(4):
            card = rng.choice(cards)
            while card in SINGLE_COPY and card in row["cards"]:  # a team holds one of these at most
                card = rng.choice(cards)
            mutation = 0
            if catalog.card(card).weather_id == 1 and rng.random() < mutation_rate:
                mutation = rng.randrange(1, len(MUTATION_NAMES))
            row["cards"].append(card)
            row["borders"].append(border())
            row["mutations"].append(mutation)
            row["arts"].append(rng.randint(1, len(ASTRAEUS_ARTS)) if card == ASTRAEUS else 0)
        for name, values in row.items():
            spec[name].append(values)
        for color, table in (("red", catalog.red_supports), ("blue", catalog.blue_supports)):
            if rng.random() < none_support:
                spec[color].append(0)
                spec[color + "_tier"].append(0)
            else:
                support = rng.choice(sorted(s.id for s in table))
                spec[color].append(support)
                spec[color + "_tier"].append(rng.choice(AURA_TIERS))
    return spec


# Abilities whose effect does not scale with the holder's stats, by his ability texts (a sampling bias, not a rule).
STAT_IGNORING = re.compile(
    r"current HP|of (?:its|their|the enemy's|enemy's|target's|the target's) (?:HP|health|stats)|loses? \d+% of (?:its|their)"
    r" (?:Max )?HP|loses? \d+% (?:Max )?HP|dies instead|die after|both active cards die|inflicts? death|chance to kill|"
    r"will kill|steals? \d+% of|revive|Max HP as damage|Max HP damage|infinite damage|HP in half|shared with the other|"
    r"abilities apply to you instead", re.I)
_STAT_IGNORING_CARDS = None


def stat_ignoring_cards(catalog):
    """Card IDs whose ability matches STAT_IGNORING."""
    global _STAT_IGNORING_CARDS
    if _STAT_IGNORING_CARDS is None:
        abilities = drago.his_data("abilities")
        his = {c["name"]: c["ability"] for c in drago.his_data("cards")}
        _STAT_IGNORING_CARDS = sorted(card for card, name in drago.names(catalog)[0].items()
                                      if STAT_IGNORING.search(abilities.get(his[name], "") or ""))
    return _STAT_IGNORING_CARDS


TOWER_SHARE = 1 / 3  # fixed-stat battles that are tower floors (module docstring)


PRIOR = 0.5  # chance a leaning side's card comes from STAT_IGNORING, a prior the user means to drop (--prior 0)


def _lean_on_stat_ignoring(rng, catalog, spec, side):
    """Each of a side's cards becomes, with probability PRIOR, one whose ability ignores raw stats."""
    pool = stat_ignoring_cards(catalog)
    for slot in range(4):
        if rng.random() < PRIOR:
            card = rng.choice(pool)
            while card in SINGLE_COPY and card in spec["cards"][side]:
                card = rng.choice(pool)
            spec["cards"][side][slot] = card
            spec["mutations"][side][slot] = 0
            spec["arts"][side][slot] = rng.randint(1, len(ASTRAEUS_ARTS)) if card == ASTRAEUS else 0


def fixed_battle(rng, catalog, spec):
    """Turns a random spec into a fixed-stat battle: (side, (HP, ATK, HP multiplier applies)); that side's borders
    become none. A third are tower floors, the rest half big gaps (module docstring); in both of those the other side
    leans on stat-ignoring abilities."""
    if rng.random() < TOWER_SHARE:
        floor, level = tower.draw_floor(rng)
        team = tower.fixed_team(catalog, floor)  # None: the game draws the floor's team, so the random cards stay
        for key in FIELDS:
            spec[key][1] = team[key] if team is not None else spec[key][1]
        spec["mutations"][1] = [0] * 4
        for key in ("red", "red_tier", "blue", "blue_tier"):  # tower enemies have no supports (as in training.hard)
            spec[key][1] = 0
        _lean_on_stat_ignoring(rng, catalog, spec, 0)
        spec["borders"][1] = [tower.enemy_border(catalog, level)] * 4
        return 1, tower.stats(floor, level)
    base = _base_stats(catalog)
    side = rng.randrange(2)
    other = 1 - side
    big = rng.random() < 0.5
    if big:
        _lean_on_stat_ignoring(rng, catalog, spec, other)
    logs = np.log([base[c, b, m] for c, b, m in zip(spec["cards"][other], spec["borders"][other], spec["mutations"][other])])
    level = 10 ** (rng.uniform(1, 4.5) if big else rng.uniform(-2, 4))
    balance = 10 ** rng.uniform(-0.5, 0.5)
    hp, attack = np.exp(logs.mean(0)) * level * np.array([balance, 1 / balance])
    spec["borders"][side] = [1] * 4
    return side, (float(hp), float(attack), False)


_BASE = None


def _base_stats(catalog):
    global _BASE
    if _BASE is None:
        _BASE = np.asarray(drago.stat_tables(catalog)[0])
    return _BASE


ENGINE_ERRORS = SHARD_DIR / "engine_errors.jsonl"  # battles the engine could not finish, for investigation


def evaluate(catalog, spec, seed, **overrides):
    """((A, B, tie, unfinished), exact) for one label spec; ties never happen (a draw is A's loss).

    A battle the engine fails on (his "Maximum call stack size exceeded", or the C engine running out of card slots
    or recursion depth) is left unfinished, so training skips
    the row, and is appended to ENGINE_ERRORS instead of stopping the whole label run. If the engine process itself
    died (a broken pipe or an empty reply), the next battle starts a fresh one."""
    if os.environ.get("CARD_ENGINE_TRACE"):  # each battle before the engine runs it: the last one shown is a crash's
        print(json.dumps({"seed": int(seed), "spec": spec, "overrides": repr(overrides)}), file=sys.stderr, flush=True)
    try:
        return kernel.evaluate(catalog, spec, seed, **overrides)
    except (RuntimeError, OSError, ValueError) as error:  # engine error; dead process (pipe); empty reply (JSON)
        if not isinstance(error, RuntimeError):
            drago._WORKER = None
        ENGINE_ERRORS.parent.mkdir(parents=True, exist_ok=True)
        with open(ENGINE_ERRORS, "a") as log:
            log.write(json.dumps({"error": str(error), "seed": int(seed), "spec": spec}) + "\n")
        return (0.0, 0.0, 0.0, 1.0), False


def label_specs(catalog, specs, *, seed, tablebase=None, **overrides):
    """Per spec: outcome probabilities (A, B, tie, unfinished) and whether they are exact.

    Exact outcomes are stored in (and read from) the tablebase; estimated ones are never stored, so a battle
    queued again later is re-estimated with fresh playouts (user)."""
    probs = np.zeros((len(specs), 4), dtype=np.float32)
    exact = np.zeros(len(specs), dtype=bool)
    found = []
    for index, spec in enumerate(specs):
        known = tablebase.get(spec) if tablebase is not None else None
        if known is not None:
            probs[index], exact[index] = known, True
            continue
        outcome, is_exact = evaluate(catalog, spec, seed * 1_000_003 + index, **overrides)
        probs[index], exact[index] = outcome, is_exact
        if is_exact:
            found.append((spec, outcome))
    if tablebase is not None and found:
        tablebase.put_many(found)
    return probs, exact


STORE = SHARD_DIR / "store"  # every rules version's shards; validity per row comes from flags.valid_rows


def _worker(args):
    shard_seed, rows, out_dir, fingerprint, fixed, *tablebase_root = args  # fingerprint: the rules snapshot id
    catalog = load_catalog()
    rng = random.Random(f"fixed-{shard_seed}" if fixed else shard_seed)  # fixed_N's battles differ from shard_N's
    specs = [random_spec(rng, catalog) for _ in range(rows)]
    extra = {}
    if fixed:
        battles = [fixed_battle(rng, catalog, spec) for spec in specs]
        probs = np.zeros((rows, 4), dtype=np.float32)
        exact = np.zeros(rows, dtype=bool)
        for index, (spec, (side, stats)) in enumerate(zip(specs, battles)):
            per_card = tower.engine_stats(catalog, spec["cards"][side], stats)
            probs[index], exact[index] = evaluate(catalog, spec, shard_seed * 1_000_003 + index, fixed=(side, per_card))
        extra = {"fixed_side": np.array([side for side, _ in battles], dtype=np.int8),
                 "fixed_stats": np.array([stats[:2] for _, stats in battles], dtype=np.float32),
                 "fixed_hp_mult": np.array([stats[2] for _, stats in battles], dtype=np.int8)}
    else:
        probs, exact = label_specs(catalog, specs, seed=shard_seed, tablebase=Tablebase(fingerprint, *tablebase_root))
    arrays = {name: np.array([spec[name] for spec in specs], dtype=np.int16) for name in FIELDS}
    path = Path(out_dir) / f"{'fixed' if fixed else 'shard'}_{shard_seed:08d}.npz"
    tmp = path.with_name(f"partial_{path.name}")  # not matched by shard_*.npz until complete
    np.savez_compressed(tmp, probs=probs, exact=exact, snapshot=np.array(fingerprint), **arrays, **extra)
    os.replace(tmp, path)
    return path.name, rows, int(exact.sum())


POD_SEEDS = 5_000_000  # the pod numbers from here up (docs/runpod.md); the Mac stays below
# What an incomplete-mode row stores for its unseen side; the model never reads it (card_visible), only valid ids.
HIDDEN_TEAM = {"cards": [1, 2, 3, 4], "borders": [1] * 4, "mutations": [0] * 4, "arts": [0] * 4,
               "red": 0, "red_tier": 0, "blue": 0, "blue_tier": 0}
GENERATION_SEEDS = 1_000_000  # incomplete-mode shards: hidden_<generation * this + n>, so the name gives the model generation


def next_seed(existing, first_seed=None):
    """The first free seed of this machine's range: past its highest shard, from first_seed (the pod's ranges) or
    below POD_SEEDS (the Mac, which also holds the pod's shards once pod_sync pulls them)."""
    own = [seed for seed in existing if (seed >= first_seed if first_seed is not None else seed < POD_SEEDS)]
    return max(own) + 1 if own else (first_seed if first_seed is not None else 1)


def _set_prior(prior):
    global PRIOR
    PRIOR = prior


def generate(shards, *, rows=2000, workers=None, first_seed=None, out_dir=STORE, fixed=False, prior=None):
    """Write `shards` new shards (seeds continue after those on disk) with a process pool, stamped with the
    current rules snapshot (training.flags). fixed: fixed-stat battles (fixed_<seed>.npz, their own seeds). prior:
    the leaning sides' chance per card of a STAT_IGNORING card (default PRIOR)."""
    from .flags import snapshot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = snapshot()
    existing = {int(p.stem.split("_")[1]) for p in out_dir.glob("fixed_*.npz" if fixed else "shard_*.npz")}
    seed = next_seed(existing, first_seed)
    jobs = []
    while len(jobs) < shards:
        if seed not in existing:
            jobs.append((seed, rows, str(out_dir), fingerprint, fixed))
        seed += 1
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    started = time.time()
    # An executor, not multiprocessing.Pool: a worker the OS kills raises BrokenProcessPool and ends the run (a shell
    # loop restarts it; unfinished shards are redone), where Pool waited forever for the lost shard.
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_set_prior,
                             initargs=(PRIOR if prior is None else prior,)) as pool:
        exact_total = 0
        for done, future in enumerate(as_completed([pool.submit(_worker, job) for job in jobs]), 1):
            name, count, exact = future.result()
            exact_total += exact
            rate = done * rows / (time.time() - started)
            print(json.dumps({"shard": name, "done": done, "of": shards, "battles_per_s": round(rate, 1),
                              "exact_fraction": round(exact_total / (done * rows), 3)}), flush=True)
    return out_dir


def load_shards(directory=STORE):
    """Concatenate the rows still valid under the current rules (training.flags) from every shard, fixed-stat ones
    included (fixed_side -1 elsewhere)."""
    from .flags import entity_hashes, valid_rows
    current = entity_hashes()
    parts = []
    for path in sorted(shard_paths(directory)):
        with np.load(path) as shard:
            arrays = {name: shard[name] for name in (*FIELDS, "probs", "exact")}
            arrays.update(battle_arrays(shard, len(arrays["probs"])))
            mask = valid_rows(arrays, str(shard["snapshot"]), current) & possible_rows(arrays["cards"])
        parts.append({k: v[mask] for k, v in arrays.items()})
    if not parts:
        raise FileNotFoundError(f"No label shards in {directory}")
    return {name: np.concatenate([part[name] for part in parts]) for name in (*FIELDS, *BATTLE_FIELDS, "probs", "exact")}


def possible_rows(cards):
    """Rows whose sides hold at most one of each single-copy card (teams.SINGLE_COPY). cards: [N, 2, 4]. Older shards
    have such battles, scored by his engine, where the copies work twice; the game doesn't allow that (user)."""
    cards = np.asarray(cards)
    return ~np.any([(cards == card).sum(-1).max(-1) > 1 for card in SINGLE_COPY], axis=0)


def shard_paths(directory):
    """Every label shard: random battles (shard_*), fixed-stat battles (fixed_*), hard examples (hard_*, training.hard)
    incomplete-mode battles (hidden_*, training.incomplete), PvP battles in complete mode (pvp_*, training.incomplete)
    and the annealed search's own winners (found_*, training.hard --select engine); partial files excluded."""
    return [path for kind in ("shard", "fixed", "hard", "hidden", "found", "pvp") for path in Path(directory).glob(f"{kind}_*.npz")]


def battle_arrays(shard, rows):
    """A shard's fixed-stat and hidden-side fields; a shard without them has none (side -1), an older fixed shard no
    HP multiplier."""
    hidden = {"hidden_side": shard["hidden_side"] if "hidden_side" in shard else np.full(rows, -1, dtype=np.int8)}
    if "fixed_side" not in shard:
        return {"fixed_side": np.full(rows, -1, dtype=np.int8), "fixed_stats": np.zeros((rows, 2), dtype=np.float32),
                "fixed_hp_mult": np.zeros(rows, dtype=np.int8), **hidden}
    return {"fixed_side": shard["fixed_side"], "fixed_stats": shard["fixed_stats"],
            "fixed_hp_mult": shard["fixed_hp_mult"] if "fixed_hp_mult" in shard else np.zeros(rows, dtype=np.int8),
            **hidden}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--first-seed", type=int, help="number shards on from here (the pod's range); default: below POD_SEEDS")
    parser.add_argument("--prior", type=float, default=PRIOR,
                        help="fixed-stat battles: chance a leaning side's card comes from the stat-ignoring list (0: none)")
    parser.add_argument("--fixed", action="store_true", help="fixed-stat battles (every card on one side at the same "
                        "stats, up to 10,000x the other side's), written as fixed_<seed>.npz")
    parsed = parser.parse_args()
    print(generate(parsed.shards, rows=parsed.rows, workers=parsed.workers, first_seed=parsed.first_seed,
                   fixed=parsed.fixed, prior=parsed.prior))
