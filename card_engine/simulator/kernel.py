"""The C battle engine (sim_c/): DaddyDrago's battle rules and the label search, compiled for speed.

His TypeScript still builds each battle's starting state (supports, deck passives, Draconian, Astraeus arts, tweaks:
sim_js/worker.ts `state`); this module loads that state into the C engine and runs sim_c's port of sim_js/search.ts,
which gives the same answers as the TypeScript search (tests/test_kernel.py). sim_js/setup.sh builds the library.
"""

import ctypes
import sys

from . import drago

LIBRARY = drago.ROOT / "sim_c" / "build" / ("libcardsim.dylib" if sys.platform == "darwin" else "libcardsim.so")
VERSION = b"card-engine-c-3"
CODES = ["", "Pl", "Cr", "CrPl", "Ru", "RuPl", "RuCr", "RuCrPl", "Ga", "GaPl", "GaCr", "GaCrPl", "GaRu", "GaRuPl", "GaRuCr",
         "GaRuCrPl"]
ABBREVIATION = {"Galaxy": "Ga", "Ruby": "Ru", "Crystal": "Cr", "Platinum": "Pl"}
BOOSTS = ["shielder", "fate", "flameWizard", "phantom", "berserker", "synthHuman", "endTimes", "vampireMatron", "stormSpirit",
          "guardianAngel", "executioner", "mirrorKnight", "finalTestament", "fossils", "composerCount", "composerThreshold",
          "noAbilities"]
DISPLAY_BOOSTS = {"statAuraName", "statAuraValue", "skillAuraName", "skillAuraValue"}  # his debug text only
AO_UNDEFINED, AO_NULL, NOCARD = -2, -1, -1

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        if not LIBRARY.exists():
            raise drago.EngineUnavailable("the C engine is not built: run sim_js/setup.sh")
        lib = ctypes.CDLL(str(LIBRARY))
        lib.ce_version.restype = ctypes.c_char_p
        if lib.ce_version() != VERSION:
            raise drago.EngineUnavailable("the C engine is out of date: run sim_js/setup.sh")
        lib.ce_state_new.restype = ctypes.c_void_p
        lib.ce_state_free.argtypes = [ctypes.c_void_p]
        lib.ce_state_set.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
        for name in ("ce_flag_index", "ce_counter_index", "ce_ability_index", "ce_def_index"):
            getattr(lib, name).argtypes = [ctypes.c_char_p]
        double_p, int_p = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        lib.ce_add_card.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                    ctypes.c_int, double_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, int_p, double_p, ctypes.c_int, ctypes.c_int]
        lib.ce_set_flag.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.ce_set_counter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.ce_set_boosts.argtypes = [ctypes.c_void_p, ctypes.c_int, double_p, ctypes.c_int, ctypes.c_int]
        lib.ce_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                 ctypes.c_uint, double_p]
        _LIB = lib
    return _LIB


def _index(kind, name):
    found = getattr(library(), f"ce_{kind}_index")(name.encode())
    if found < 0:
        raise ValueError(f"the C engine has no {kind} {name!r}")
    return found


def _doubles(values):
    return (ctypes.c_double * len(values))(*values)


class Battle:
    """A battle's starting state inside the C engine (from the worker's `state` reply)."""

    def __init__(self, exported):
        lib = library()
        self.state = ctypes.c_void_p(lib.ce_state_new())
        try:
            self._load(lib, exported)
        except Exception:
            self.close()
            raise

    def _load(self, lib, exported):
        lists = {}
        for which, positions in enumerate(exported["lists"]):
            for position in positions:
                if position in lists:
                    raise ValueError("a card in two lists has no C counterpart")
                lists[position] = which
        if len(lists) != len(exported["cards"]):
            raise ValueError("a card outside the team and fallen lists has no C counterpart")
        # cards in list order, so each list keeps its order
        for which, positions in enumerate(exported["lists"]):
            for position in positions:
                self._add(lib, which, exported["cards"][position])
        for team, boosts in enumerate(exported["boosts"]):
            unknown = set(boosts) - set(BOOSTS) - DISPLAY_BOOSTS
            if unknown:
                raise ValueError(f"boosts {sorted(unknown)} have no C counterpart")
            values = [float(boosts.get(name) or 0) for name in BOOSTS]
            lib.ce_set_boosts(self.state, team, _doubles(values), int(boosts.get("composerThreshold") is not None),
                              int("skillAuraName" in boosts))
        lib.ce_state_set(self.state, float(exported["turn"]), int(exported["moving"]))

    def _add(self, lib, which, card):
        code = "".join(ABBREVIATION[b] for b in card["borders"])
        if "abilityOverride" not in card:
            override = AO_UNDEFINED
        elif card["abilityOverride"] is None:
            override = AO_NULL
        else:
            override = _index("ability", card["abilityOverride"])
        identity = NOCARD if card["identity"] is None else _index("def", card["identity"])
        bonus = [_index("ability", name) for name in card["bonus"]]
        if len(bonus) > 6:
            raise ValueError("more than 6 bonus abilities")
        status = card["status"]
        c = lib.ce_add_card(self.state, which, card["id"].encode(), _index("def", card["def"]), card["team"], float(card["index"]),
                            CODES.index(code) + 1, _doubles([card["power"], card["hp"], card["maxHp"], card["damage"]]),
                            int(card["entered"]), int(card["dead"]), int(card["boss"]), identity, override, len(bonus),
                            (ctypes.c_int * max(1, len(bonus)))(*bonus),
                            _doubles([status.get(k) or 0 for k in ("stunned", "confused", "burn", "shield")]),
                            int(bool(status.get("weakness"))), int(bool(status.get("blind"))))
        if c < 0:
            raise ValueError("too many cards for the C engine")
        for name, value in card["flags"].items():
            if value:
                lib.ce_set_flag(self.state, c, _index("flag", name))
        for name, value in card["counters"].items():
            lib.ce_set_counter(self.state, c, _index("counter", name), float(value))

    def solve(self, nodeBudget, sampleBelow, rollouts, rolloutError, maxTurns, seed):
        """{"a", "b", "draw", "exact", "nodes", "playouts"}, as sim_js/search.ts solve."""
        out = (ctypes.c_double * 7)()
        library().ce_solve(self.state, nodeBudget, sampleBelow, rollouts, rolloutError, maxTurns, seed, out)
        if out[6]:
            raise RuntimeError("C engine: more cards, deeper recursion or more operations in one playthrough than it allows "
                               "(sim_c/engine.h MAXC, engine.c MAXDEPTH, MAXOPS)")
        return {"a": out[0], "b": out[1], "draw": out[2], "exact": bool(out[3]), "nodes": int(out[4]), "playouts": int(out[5])}

    def close(self):
        if self.state:
            library().ce_state_free(self.state)
            self.state = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def start(a, b, tweaks=None):
    """(Battle, unsupported abilities) for two of his loadouts."""
    exported = drago.worker().request({"op": "state", "a": a, "b": b, "options": tweaks or {}})
    return Battle(exported), exported["unsupported"]


def evaluate(catalog, spec, seed, *, fixed=None, scale=None, strip=None, **overrides):
    """drago.evaluate on the C engine: ((P(A wins), P(B wins), 0, unfinished), exact); a draw is A's loss."""
    a, b = drago.loadouts(catalog, spec)
    battle, unsupported = start(a, b, drago._tweaks(fixed, scale, strip))
    with battle:
        if unsupported:
            return (0.0, 0.0, 0.0, 1.0), False
        unknown = set(overrides) - set(drago.SEARCH)
        if unknown:
            raise TypeError(f"unknown search options {sorted(unknown)}")
        options = {**drago.SEARCH, **overrides}
        r = battle.solve(options["nodeBudget"], options["sampleBelow"], options["rollouts"], options["rolloutError"],
                         options["maxTurns"], int(seed) % 2 ** 31 or 1)
    return (r["a"], r["b"] + r["draw"], 0.0, 0.0), r["exact"]
