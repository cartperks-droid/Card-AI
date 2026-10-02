"""Battle labels for the general win predictor: random matchups scored by the C kernel.

Every battle is searched best-first in branch mode (largest probabilities first, under a node
budget); small or leftover branches are resolved by playouts, as many as reducing the variance
needs. Battles with no estimated probability are exact and kept in the tablebase; estimated ones
are recomputed whenever they are labelled again. Side A always initiates. Unfinished probability
is recorded per row (column 3) and left out of training targets.
Labels depend on the simulator rules: each shard records the kernel version and a hash of
the cleaned data and rule tables, so stale shards are detected after rule fixes.
"""

import hashlib
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np

from ..catalog import load_catalog
from ..mutations import MUTATION_NAMES
from ..simulator import native
from ..simulator.catalog_rules import ASTRAEUS, ASTRAEUS_ARTS, BLUE_SUPPORTS, RED_SUPPORTS, SUPPORTED, compile_battle
from ..simulator.reference import Options
from .tablebase import EXACT_TOLERANCE, Tablebase

ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "data" / "labels"
ART_NAMES = tuple(ASTRAEUS_ARTS)  # art id = index + 1; 0 = not Astraeus
FIELDS = ("cards", "borders", "mutations", "arts", "red", "red_tier", "blue", "blue_tier")


def rules_fingerprint():
    """Kernel version plus a hash of everything that shapes battle outcomes."""
    digest = hashlib.sha256()
    for path in (ROOT / "data/clean/dataset.json", ROOT / "card_engine/simulator/catalog_rules.py",
                 ROOT / "card_engine/simulator/reference.py", ROOT / "card_engine/stats.py",
                 ROOT / "card_engine/data_corrections.py", ROOT / "sim/card_sim.c"):
        digest.update(path.read_bytes())
    return f"{native.load_library().ce_version().decode()}:{digest.hexdigest()[:16]}"


def _tiers(table, support_id):
    values = table[support_id][2] if isinstance(table[support_id], tuple) else table[support_id]
    return [tier for tier, value in enumerate(values, 1) if value is not None]


def random_spec(rng, catalog, cards=tuple(sorted(SUPPORTED)), none_support=0.1, mutation_rate=0.5, matched=0.7):
    """One random 4v4 matchup: any card, border, eligible mutation, and support tier.

    With probability `matched`, every border sits within two rarity ranks of a shared level, so the
    outcome depends on the cards rather than on a border gap; the rest draw borders uniformly.
    """
    spec = {name: [] for name in FIELDS}
    ranked = sorted(range(1, 17), key=lambda b: (catalog.border(b).rarity, b))
    level = rng.randrange(16) if rng.random() < matched else None
    border = lambda: rng.randint(1, 16) if level is None else ranked[min(15, max(0, level + rng.randint(-2, 2)))]
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
            row["arts"].append(rng.randint(1, len(ART_NAMES)) if card == ASTRAEUS else 0)
        for name, values in row.items():
            spec[name].append(values)
        for color, table in (("red", RED_SUPPORTS), ("blue", BLUE_SUPPORTS)):
            if rng.random() < none_support:
                spec[color].append(0)
                spec[color + "_tier"].append(0)
            else:
                support = rng.choice(sorted(table))
                spec[color].append(support)
                spec[color + "_tier"].append(rng.choice(_tiers(table, support)))
    return spec


def compile_spec(catalog, spec):
    support = lambda color, side: (spec[color][side], spec[color + "_tier"][side]) if spec[color][side] else 0
    arts = {(side, slot): ART_NAMES[art - 1] for side in (0, 1) for slot, art in enumerate(spec["arts"][side]) if art}
    return compile_battle(catalog, tuple(tuple(team) for team in spec["cards"]), borders=spec["borders"],
                          mutations=[[MUTATION_NAMES[m] for m in team] for team in spec["mutations"]],
                          red_supports=(support("red", 0), support("red", 1)),
                          blue_supports=(support("blue", 0), support("blue", 1)), arts=arts)


# User: branch the largest probabilities first; simulate the small branches once a depth is exhausted or the
# search budget is spent, with as many playouts as reducing the variance needs (not a fixed number).
SEARCH = dict(mode="branch", max_frontier=512, prune_probability=0.0, node_budget=20000, rollouts=1024,
              rollout_error=0.03, sample_below=1e-3, max_steps=200000)


def evaluate(battle, seed, **overrides):
    """(A, B, tie, unfinished) probabilities and whether they are exact (no estimated probability)."""
    try:
        row = native.simulate(battle, Options(seed=seed % 2**63, **{**SEARCH, **overrides}))
    except native.NativeSimulationError:
        return (0.0, 0.0, 0.0, 1.0), False  # the kernel cannot finish this battle: no training target
    outcome = (row["p_a"], row["p_b"], row["tie"], row["unresolved"])
    return outcome, row["estimated"] == 0 and row["unresolved"] <= EXACT_TOLERANCE


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
        outcome, is_exact = evaluate(compile_spec(catalog, spec), seed * 1_000_003 + index, **overrides)
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
    parsed = parser.parse_args()
    print(generate(parsed.shards, rows=parsed.rows, workers=parsed.workers))
