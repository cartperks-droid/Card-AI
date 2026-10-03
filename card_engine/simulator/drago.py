"""DaddyDrago's battle engine as the label simulator (user, 2026-10-03: his mechanics are more accurate).

sim_js/setup.sh fetches his engine at the pinned commit and patches it so every random draw is a chance point;
sim_js/search.ts runs the best-first chance-tree search around it; sim_js/worker.ts serves one battle per line.
This module translates our label specs (card IDs, border IDs, mutation indices, support IDs and tiers) into his
loadouts and keeps one Node worker per process.
"""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "sim_js"
ENGINE_DIR = SIM_DIR / "vendor" / "CardRngExpansionDepths"
BORDER_PARTS = {"Ga": "Galaxy", "Ru": "Ruby", "Cr": "Crystal", "Pl": "Platinum"}
AURA_BORDERS = {1: None, 2: "Platinum", 3: "Crystal", 5: "Galaxy"}  # support cards have no Ruby border
# our spelling -> his, where they differ beyond case and punctuation
CARD_ALIASES = {"judgementday": "judgmentday", "achyls": "achlys", "sorceror": "sorcerer",
                "demoncultivator": "demoniccultivator", "thejadeemporer": "thejadeemperor", "tricerotops": "triceratops",
                "chupcabra": "chupacabra", "thejackinthebox": "toyjackinthebox", "tyrannodon": "tyranodon"}
# search settings: best-first over chance points, playouts for the rest (as the C kernel's label search)
SEARCH = {"nodeBudget": 2000, "sampleBelow": 1e-3, "rollouts": 1024, "rolloutError": 0.03, "maxTurns": 2000}


class EngineUnavailable(RuntimeError):
    pass


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _his_data(kind):
    files = sorted((ENGINE_DIR / "src" / "data").glob(f"{kind}-*.json"))
    if not files:
        raise EngineUnavailable(f"DaddyDrago's engine is not set up: run sim_js/setup.sh ({ENGINE_DIR} is missing)")
    return [entry for path in files for entry in json.loads(path.read_text())]


_NAMES = {}


def names(catalog):
    """(card id -> his card name, red support id -> his aura name, blue support id -> his aura name)."""
    if id(catalog) in _NAMES:
        return _NAMES[id(catalog)]
    his_cards = {_norm(c["name"]): c["name"] for c in _his_data("cards")}
    cards = {}
    for card in catalog.cards:
        key = _norm(card.name)
        key = CARD_ALIASES.get(key, key)
        if key not in his_cards:
            raise EngineUnavailable(f"Card {card.id} {card.name} has no counterpart in DaddyDrago's data")
        cards[card.id] = his_cards[key]
    auras = {(a["type"], _norm(a["name"])): a["name"] for a in _his_data("auras")}
    red = {s.id: auras[("Stat", _norm(s.name))] for s in catalog.red_supports}
    blue = {s.id: auras[("Skill", _norm(s.name))] for s in catalog.blue_supports}
    _NAMES[id(catalog)] = cards, red, blue
    return cards, red, blue


def border_names(catalog, border_id):
    code = catalog.border(border_id).code
    return [BORDER_PARTS[code[i:i + 2]] for i in range(0, len(code), 2)]


def loadouts(catalog, spec):
    """His (side A, side B) team loadouts for one of our label specs."""
    from ..mutations import MUTATION_NAMES
    cards, red, blue = names(catalog)
    sides = []
    for side in (0, 1):
        team = []
        for card, border, mutation in zip(spec["cards"][side], spec["borders"][side], spec["mutations"][side]):
            entry = {"cardName": cards[card], "borders": border_names(catalog, border)}
            if mutation:
                entry["mutationWeather"] = MUTATION_NAMES[mutation]
            team.append(entry)
        loadout = {"cards": team, "statAura": None, "abilityAura": None}
        for color, table, key in (("red", red, "statAura"), ("blue", blue, "abilityAura")):
            support, tier = spec[color][side], spec[color + "_tier"][side]
            if support:
                if tier not in AURA_BORDERS:
                    raise ValueError(f"{color} support {support} tier {tier}: support cards have no Ruby border")
                loadout[key] = {"auraName": table[support], "border": AURA_BORDERS[tier]}
        sides.append(loadout)
    return sides


class Worker:
    """One Node process running sim_js/worker.ts; requests are answered in order."""

    def __init__(self):
        if not (ENGINE_DIR / "src" / "engine" / "battle-v2.label.ts").exists():
            raise EngineUnavailable("DaddyDrago's engine is not set up: run sim_js/setup.sh")
        self.process = subprocess.Popen(["npx", "--no-install", "tsx", "worker.ts"], cwd=SIM_DIR, text=True, bufsize=1,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def solve(self, a, b, options):
        self.process.stdin.write(json.dumps({"a": a, "b": b, "options": options}) + "\n")
        reply = json.loads(self.process.stdout.readline())
        if "error" in reply:
            raise RuntimeError(f"DaddyDrago engine: {reply['error']}")
        return reply

    def close(self):
        self.process.stdin.close()
        self.process.wait()


_WORKER = None


def evaluate(catalog, spec, seed, b_stats=None, **overrides):
    """((P(A wins), P(B wins), P(draw), unfinished), exact) for a label spec; side A moves first.

    b_stats optionally fixes side B's (HP, ATK) per card (event or Tower teams). A battle with an ability or aura
    his engine marks unsupported gets no answer: ((0, 0, 0, 1), False)."""
    global _WORKER
    if _WORKER is None:
        _WORKER = Worker()
    a, b = loadouts(catalog, spec)
    options = {**SEARCH, **overrides, "seed": int(seed) % 2 ** 31 or 1}
    if b_stats is not None:
        options["bStats"] = [[float(hp), float(attack)] for hp, attack in b_stats]
    reply = _WORKER.solve(a, b, options)
    if reply["unsupported"]:
        return (0.0, 0.0, 0.0, 1.0), False
    return (reply["a"], reply["b"], reply["draw"], 0.0), bool(reply["exact"])
