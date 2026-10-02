"""Which labels are still valid after a rule change, decided per entity rather than per rules version.

Every rules version has one small snapshot of entity fingerprints (data/labels/snapshots/<id>.json, shared by
every shard made under it; a shard stores only the 12-character snapshot id). Rows already hold their cards,
borders, mutations and supports, so what a row involves is derived when loading, never stored.

Entities:
  card:<id>        the card's compiled rules (ability fields, stats, classes, pack; forms for transforming cards)
  art:<name>       an Astraeus art
  support:<color><id>:<tier>, border:<id>, mutation:<index>
  support_logic    how supports are applied (involved by every row with a support)
  pool             every card's ability (involved by rows with random-ability cards: Pandora, Glamour, Nuwa...)
  core             the engines and battle assembly (reference.py, card_sim.c/h, stats formula, compile_battle)

A row stays valid if every entity it involves is unchanged. A core change cannot be attributed automatically:
labels made before it are held back until it is declared, with the cards/supports it affects (or --all):

    python -m card_engine.training.flags status
    python -m card_engine.training.flags declare --cards Buddha "Mother of Beasts" --note "Buddha revive order"
    python -m card_engine.training.flags declare --none --note "refactor, no outcome change"
    python -m card_engine.training.flags declare --all --note "damage rounding"
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from ..catalog import load_catalog
from ..mutations import MULTIPLIERS, MUTATION_NAMES
from ..simulator import catalog_rules
from ..simulator.catalog_rules import (ASTRAEUS, ASTRAEUS_ARTS, BLUE_SUPPORTS, FORMS, RANDOM_ABILITY_FIELDS, RED_SUPPORTS,
                                       SUPPORTED, compile_fighter)

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "data" / "labels" / "snapshots"
CHANGES_FILE = ROOT / "data" / "labels" / "rule_changes.json"
ART_NAMES = tuple(ASTRAEUS_ARTS)


def _digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode())
    return h.hexdigest()[:12]


def entity_hashes(catalog=None):
    catalog = catalog or load_catalog()
    out, pool = {}, []
    for card in catalog.cards:
        if card.id not in SUPPORTED:
            continue
        fighter = compile_fighter(catalog, card.id)
        forms = FORMS if (fighter.form_first or fighter.transform_form or fighter.kill_summon_form) else None
        out[f"card:{card.id}"] = _digest(fighter, card.classes, card.packs, forms)
        pool.append(out[f"card:{card.id}"])
    for art in ART_NAMES:
        out[f"art:{art}"] = _digest(compile_fighter(catalog, ASTRAEUS, art=art))
    for color, table in (("red", RED_SUPPORTS), ("blue", BLUE_SUPPORTS)):
        for sid, row in table.items():
            values = row[2] if isinstance(row, tuple) else row
            for tier, value in enumerate(values, 1):
                out[f"support:{color}{sid}:{tier}"] = _digest(row[:2] if isinstance(row, tuple) else None, value)
    for border in range(1, 17):
        out[f"border:{border}"] = _digest(catalog.border(border).rarity)
    for index, name in enumerate(MUTATION_NAMES):
        out[f"mutation:{index}"] = _digest(name, MULTIPLIERS[name])
    out["support_logic"] = _digest(*(inspect.getsource(f) for f in (catalog_rules._apply_supports, catalog_rules._red_bonus,
                                                                     catalog_rules._awaken)))
    out["pool"] = _digest(pool, list(out[f"art:{a}"] for a in ART_NAMES))
    core = [(ROOT / p).read_bytes() for p in ("card_engine/simulator/reference.py", "sim/card_sim.c", "sim/card_sim.h",
                                             "card_engine/stats.py", "card_engine/mutations.py")]
    core += [inspect.getsource(f) for f in (catalog_rules.compile_battle, catalog_rules._spares)]
    out["core"] = _digest(*core)
    return out


def snapshot(catalog=None):
    """Write (once) and return the id of the current rules' snapshot."""
    hashes = entity_hashes(catalog)
    ident = _digest(sorted(hashes.items()))
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{ident}.json"
    if not path.exists():
        path.write_text(json.dumps(hashes, separators=(",", ":"), sort_keys=True))
    return ident


def load_snapshot(ident):
    return json.loads((SNAPSHOT_DIR / f"{ident}.json").read_text())


def _pool_cards(catalog):
    cards = set()
    for card in catalog.cards:
        if card.id in SUPPORTED:
            fighter = compile_fighter(catalog, card.id)
            if any(getattr(fighter, name) for name in RANDOM_ABILITY_FIELDS):
                cards.add(card.id)
    return cards


def load_changes():
    return json.loads(CHANGES_FILE.read_text()) if CHANGES_FILE.exists() else []


def changed_entities(old, new, changes):
    """Entities whose fingerprint differs between snapshots; None if a core change is undeclared."""
    changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    if "core" in changed:
        declared = next((c for c in changes if c["from_core"] == old["core"] and c["to_core"] == new["core"]), None)
        if declared is None:
            return None
        changed.discard("core")
        if declared.get("all"):
            return {"*"}
        changed |= set(declared.get("affects", []))
    return changed


def valid_rows(arrays, old_ident, current=None, catalog=None, changes=None, pool_cards=None):
    """Boolean mask over a shard's rows: still valid under the current rules."""
    catalog = catalog or load_catalog()
    current = current or entity_hashes(catalog)
    old = load_snapshot(old_ident)
    changed = changed_entities(old, current, load_changes() if changes is None else changes)
    rows = len(arrays["cards"])
    if changed is None or "*" in changed:
        return np.zeros(rows, dtype=bool)
    if not changed:
        return np.ones(rows, dtype=bool)
    pool_cards = _pool_cards(catalog) if pool_cards is None else pool_cards
    cards, arts = arrays["cards"].reshape(rows, -1), arrays["arts"].reshape(rows, -1)
    bad = np.zeros(rows, dtype=bool)
    for key in changed:
        kind, _, value = key.partition(":")
        if kind == "card":
            bad |= (cards == int(value)).any(1)
        elif kind == "art":
            bad |= (arts == ART_NAMES.index(value) + 1).any(1)
        elif kind == "border":
            bad |= (arrays["borders"].reshape(rows, -1) == int(value)).any(1)
        elif kind == "mutation":
            bad |= (arrays["mutations"].reshape(rows, -1) == int(value)).any(1)
        elif kind == "support":
            color = "red" if value.startswith("red") else "blue"
            sid, tier = value[len(color):].split(":")
            bad |= ((arrays[color] == int(sid)) & (arrays[color + "_tier"] == int(tier))).any(1)
        elif kind == "support_logic":
            bad |= (arrays["red"] != 0).any(1) | (arrays["blue"] != 0).any(1)
        elif kind == "pool":
            bad |= np.isin(cards, list(pool_cards)).any(1)
    return ~bad


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m card_engine.training.flags", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="valid labels per snapshot, and what changed")
    p = commands.add_parser("declare", help="declare the latest core change and which labels it affects")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--cards", nargs="+")
    group.add_argument("--all", action="store_true")
    group.add_argument("--none", action="store_true")
    p.add_argument("--supports", nargs="*", default=[], help="e.g. red27 blue8 (every tier)")
    p.add_argument("--note", required=True)
    p.add_argument("--from-snapshot", help="snapshot id to declare from (default: the newest with a different core)")
    args = parser.parse_args(argv)
    catalog = load_catalog()
    current = entity_hashes(catalog)
    from .labels import SHARD_DIR
    store = SHARD_DIR / "store"
    shards = sorted(store.glob("shard_*.npz"))
    counts = {}
    for path in shards:
        with np.load(path) as part:
            counts[str(part["snapshot"])] = counts.get(str(part["snapshot"]), 0) + len(part["probs"])
    if args.command == "status":
        changes = load_changes()
        for ident, rows in sorted(counts.items()):
            changed = changed_entities(load_snapshot(ident), current, changes)
            state = ("current" if changed == set() else "core change undeclared: held back" if changed is None
                     else "all invalid" if "*" in changed else f"{len(changed)} entities changed: "
                     + ", ".join(sorted(changed)[:8]) + (" ..." if len(changed) > 8 else ""))
            print(f"{ident}: {rows} rows, {state}")
        return
    old_ident = args.from_snapshot or next((i for i in sorted(counts, key=lambda i: -counts[i])
                                            if load_snapshot(i)["core"] != current["core"]), None)
    if old_ident is None:
        raise SystemExit("No stored labels have a different core; nothing to declare")
    names = [(c.id, c.name) for c in catalog.cards]
    from ..deck import _match
    affects = [f"card:{_match(q, names, 'Card')[0]}" for q in (args.cards or [])]
    for s in args.supports:
        color = "red" if s.startswith("red") else "blue"
        sid = int(s[len(color):])
        affects += [f"support:{color}{sid}:{t}" for t in range(1, 6)]
    changes = load_changes()
    changes.append({"from_core": load_snapshot(old_ident)["core"], "to_core": current["core"], "all": bool(args.all),
                    "affects": affects, "note": args.note})
    CHANGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_FILE.write_text(json.dumps(changes, indent=1) + "\n")
    print(f"Declared: {args.note} ({'all labels' if args.all else f'{len(affects)} entities' if affects else 'no outcome change'})")


if __name__ == "__main__":
    main()
