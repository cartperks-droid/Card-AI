"""Battle labels for the general win predictor: random matchups scored by DaddyDrago's battle engine.

Every battle is searched best-first over the engine's chance points (largest probabilities first, under a node
budget); small or leftover branches are resolved by pooled playouts, as many as reducing the variance needs
(sim_js/search.ts). Battles with no estimated probability are exact and kept in the tablebase; estimated ones are
recomputed whenever they are labelled again. Side A always initiates; a draw counts as A's loss (user).
Each shard records the rules snapshot (training.flags), so rows go stale when the rules behind them change.
"""

import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np

from ..catalog import load_catalog
from ..mutations import MUTATION_NAMES
from ..simulator import drago
from .tablebase import Tablebase

ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "data" / "labels"
FIELDS = ("cards", "borders", "mutations", "arts", "red", "red_tier", "blue", "blue_tier")


AURA_TIERS = tuple(drago.AURA_BORDERS)  # support cards: Base, Platinum, Crystal, Ruby, Galaxy


def random_spec(rng, catalog, cards=None, none_support=0.1, mutation_rate=0.5, matched=0.7):
    """One random 4v4 matchup: any card, border, eligible mutation, and support tier. Astraeus draws its art in
    battle (his engine), so arts stay 0.

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
            mutation = 0
            if catalog.card(card).weather_id == 1 and rng.random() < mutation_rate:
                mutation = rng.randrange(1, len(MUTATION_NAMES))
            row["cards"].append(card)
            row["borders"].append(border())
            row["mutations"].append(mutation)
            row["arts"].append(0)
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


ENGINE_ERRORS = SHARD_DIR / "engine_errors.jsonl"  # battles the engine could not finish, for investigation


def evaluate(catalog, spec, seed, **overrides):
    """((A, B, tie, unfinished), exact) for one label spec; ties never happen (a draw is A's loss).

    A battle the engine fails on (e.g. "Maximum call stack size exceeded") is left unfinished, so training skips
    the row, and is appended to ENGINE_ERRORS instead of stopping the whole label run. If the engine process itself
    died (a broken pipe or an empty reply), the next battle starts a fresh one."""
    try:
        return drago.evaluate(catalog, spec, seed, **overrides)
    except (RuntimeError, OSError, ValueError) as error:  # engine error; dead process (pipe); empty reply (JSON)
        if not isinstance(error, RuntimeError):
            drago._WORKER = None
        ENGINE_ERRORS.parent.mkdir(parents=True, exist_ok=True)
        with open(ENGINE_ERRORS, "a") as log:
            log.write(json.dumps({"error": str(error), "seed": int(seed), "spec": spec}) + "\n")
        return (0.0, 0.0, 0.0, 1.0), False


# Label budget (user, 2026-10-03: as fast as the old C kernel). His engine cannot pause mid-battle, so every branch
# the search expands is replayed from the start; 100 nodes and a 7% playout error label at ~26/s per core (the old
# kernel: ~30) and move labels by 0.5 points on average from the 2,000-node / 3% budget; exact battles are unchanged.
LABEL_SEARCH = {"nodeBudget": 100, "rolloutError": 0.07}


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
        outcome, is_exact = evaluate(catalog, spec, seed * 1_000_003 + index, **{**LABEL_SEARCH, **overrides})
        probs[index], exact[index] = outcome, is_exact
        if is_exact:
            found.append((spec, outcome))
    if tablebase is not None and found:
        tablebase.put_many(found)
    return probs, exact


STORE = SHARD_DIR / "store"  # every rules version's shards; validity per row comes from flags.valid_rows


def _worker(args):
    shard_seed, rows, out_dir, fingerprint, *tablebase_root = args  # fingerprint: the rules snapshot id
    catalog = load_catalog()
    rng = random.Random(shard_seed)
    specs = [random_spec(rng, catalog) for _ in range(rows)]
    probs, exact = label_specs(catalog, specs, seed=shard_seed, tablebase=Tablebase(fingerprint, *tablebase_root))
    arrays = {name: np.array([spec[name] for spec in specs], dtype=np.int16) for name in FIELDS}
    path = Path(out_dir) / f"shard_{shard_seed:08d}.npz"
    tmp = path.with_name(f"partial_{path.name}")  # not matched by shard_*.npz until complete
    np.savez_compressed(tmp, probs=probs, exact=exact, snapshot=np.array(fingerprint), **arrays)
    os.replace(tmp, path)
    return path.name, rows, int(exact.sum())


def generate(shards, *, rows=2000, workers=None, first_seed=None, out_dir=STORE):
    """Write `shards` new shards (seeds continue after those on disk) with a process pool, stamped with the
    current rules snapshot (training.flags)."""
    from .flags import snapshot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = snapshot()
    existing = {int(p.stem.split("_")[1]) for p in out_dir.glob("shard_*.npz")}
    seed = first_seed if first_seed is not None else (max(existing) + 1 if existing else 1)
    jobs = []
    while len(jobs) < shards:
        if seed not in existing:
            jobs.append((seed, rows, str(out_dir), fingerprint))
        seed += 1
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    started = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        exact_total = 0
        for done, (name, count, exact) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            exact_total += exact
            rate = done * rows / (time.time() - started)
            print(json.dumps({"shard": name, "done": done, "of": shards, "battles_per_s": round(rate, 1),
                              "exact_fraction": round(exact_total / (done * rows), 3)}), flush=True)
    return out_dir


def load_shards(directory=STORE):
    """Concatenate the rows still valid under the current rules (training.flags) from every shard."""
    from .flags import entity_hashes, valid_rows
    current = entity_hashes()
    parts = []
    for path in sorted(Path(directory).glob("shard_*.npz")):
        with np.load(path) as shard:
            arrays = {name: shard[name] for name in (*FIELDS, "probs", "exact")}
            mask = valid_rows(arrays, str(shard["snapshot"]), current)
        parts.append({k: v[mask] for k, v in arrays.items()})
    if not parts:
        raise FileNotFoundError(f"No label shards in {directory}")
    return {name: np.concatenate([part[name] for part in parts]) for name in (*FIELDS, "probs", "exact")}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--first-seed", type=int, help="number shards from here (a second machine uses its own range)")
    parsed = parser.parse_args()
    print(generate(parsed.shards, rows=parsed.rows, workers=parsed.workers, first_seed=parsed.first_seed))
