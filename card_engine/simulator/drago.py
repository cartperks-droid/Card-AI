"""DaddyDrago's battle engine as the label simulator (user, 2026-10-03: his mechanics are more accurate).

sim_js/setup.sh fetches his engine at the pinned commit and patches it so every random draw is a chance point;
sim_js/search.ts runs the best-first chance-tree search around it; sim_js/worker.ts serves one battle per line.
This module translates our label specs (card IDs, border IDs, mutation indices, support IDs and tiers) into his
loadouts and keeps one Node worker per process.
"""

import atexit
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "sim_js"
ENGINE_DIR = SIM_DIR / "vendor" / "CardRngExpansionDepths"
BORDER_PARTS = {"Ga": "Galaxy", "Ru": "Ruby", "Cr": "Crystal", "Pl": "Platinum"}
AURA_BORDERS = {1: None, 2: "Platinum", 3: "Crystal", 4: "Ruby", 5: "Galaxy"}  # Ruby: patched in by codemod.mjs
# our spelling -> his, where they differ beyond case and punctuation
CARD_ALIASES = {"judgementday": "judgmentday", "achyls": "achlys", "sorceror": "sorcerer",
                "demoncultivator": "demoniccultivator", "thejadeemporer": "thejadeemperor", "tricerotops": "triceratops",
                "chupcabra": "chupacabra", "thejackinthebox": "toyjackinthebox", "tyrannodon": "tyranodon"}
# search settings (the agreed design, as the C kernel's label search): best-first over chance points, playouts only for
# branches below 0.1% and whatever is open when the node budget runs out
SEARCH = {"nodeBudget": 20000, "sampleBelow": 1e-3, "rollouts": 1024, "rolloutError": 0.03, "maxTurns": 2000}


class EngineUnavailable(RuntimeError):
    pass


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def his_data(kind):
    """His data files of one kind (cards, auras: lists of entries; abilities: one name -> text mapping)."""
    files = sorted((ENGINE_DIR / "src" / "data").glob(f"{kind}-*.json"))
    if not files:
        raise EngineUnavailable(f"DaddyDrago's engine is not set up: run sim_js/setup.sh ({ENGINE_DIR} is missing)")
    parts = [json.loads(path.read_text()) for path in files]
    return {k: v for part in parts for k, v in part.items()} if isinstance(parts[0], dict) else [e for p in parts for e in p]


_NAMES = {}


def names(catalog):
    """(card id -> his card name, red support id -> his aura name, blue support id -> his aura name)."""
    if id(catalog) in _NAMES:
        return _NAMES[id(catalog)]
    his_cards = {_norm(c["name"]): c["name"] for c in his_data("cards")}
    cards = {}
    for card in catalog.cards:
        key = _norm(card.name)
        key = CARD_ALIASES.get(key, key)
        if key not in his_cards:
            raise EngineUnavailable(f"Card {card.id} {card.name} has no counterpart in DaddyDrago's data")
        cards[card.id] = his_cards[key]
    auras = {(a["type"], _norm(a["name"])): a["name"] for a in his_data("auras")}
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
    from ..teams import ASTRAEUS, ASTRAEUS_ARTS
    cards, red, blue = names(catalog)
    sides = []
    for side in (0, 1):
        team = []
        for card, border, mutation, art in zip(spec["cards"][side], spec["borders"][side], spec["mutations"][side],
                                               spec["arts"][side]):
            entry = {"cardName": cards[card], "borders": border_names(catalog, border)}
            if mutation:
                entry["mutationWeather"] = MUTATION_NAMES[mutation]
            if card == ASTRAEUS:  # each art is its own card with a fixed ability (user); his engine would draw one
                if not art:
                    raise ValueError("Astraeus needs its art (spec arts: 1-7)")
                entry["constellar"] = "Constellar" + ASTRAEUS_ARTS[art - 1].title()
            team.append(entry)
        loadout = {"cards": team, "statAura": None, "abilityAura": None}
        for color, table, key in (("red", red, "statAura"), ("blue", blue, "abilityAura")):
            support, tier = spec[color][side], spec[color + "_tier"][side]
            if support:
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

    def request(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        reply = json.loads(self.process.stdout.readline())
        if "error" in reply:
            raise RuntimeError(f"DaddyDrago engine: {reply['error']}")
        return reply

    def close(self):
        self.process.stdin.close()
        self.process.wait()
        self.process.stdout.close()


_WORKER = None


def worker():
    global _WORKER
    if _WORKER is None:
        _WORKER = Worker()
        atexit.register(_WORKER.close)
    return _WORKER


_SUPPORTED = {}


def supported(catalog):
    """(card ids, blue support ids) his engine fully implements; battles with anything else get no label."""
    if id(catalog) in _SUPPORTED:
        return _SUPPORTED[id(catalog)]
    cards, _, blue = names(catalog)
    team = lambda name, aura=None: {"cards": [{"cardName": name, "borders": []}], "statAura": None, "abilityAura": aura}
    check = lambda a: not worker().request({"op": "check", "a": a, "b": team(cards[1])})["unsupported"]
    ok_cards = frozenset(card for card, name in cards.items() if check(team(name)))
    ok_blue = frozenset(s for s, name in blue.items() if check(team(cards[1], {"auraName": name, "border": None})))
    _SUPPORTED[id(catalog)] = ok_cards, ok_blue
    return ok_cards, ok_blue


def _tweaks(fixed=None, scale=None, strip=None):
    """Search options for the per-battle tweaks (see evaluate)."""
    side = "ab".__getitem__
    options = {}
    if fixed is not None:
        options["fixed"] = [side(fixed[0]), [[float(hp), float(attack)] for hp, attack in fixed[1]]]
    if scale is not None and scale[1] != 1:
        options["scale"] = [side(scale[0]), float(scale[1])]
    if strip is not None:
        options["strip"] = [side(strip[0]), int(strip[1])]
    return options


def initial_cards(catalog, spec, **tweaks):
    """[side][card] = (HP, ATK, ability) as his engine starts the battle (supports, deck passives and tweaks applied)."""
    a, b = loadouts(catalog, spec)
    reply = worker().request({"op": "initial", "a": a, "b": b, "options": _tweaks(**tweaks)})
    return [reply["a"], reply["b"]]


def initial_stats(catalog, spec, **tweaks):
    """[side][card] = (HP, ATK) as his engine starts the battle."""
    return [[card[:2] for card in side] for side in initial_cards(catalog, spec, **tweaks)]


def stat_tables(catalog):
    """(base, red, prehistoric, jurassic) from his engine: base[card id, border id, mutation] = (HP, ATK); red[card id,
    mutation, red support, tier] = (HP, ATK) multiplier of the side's red (stat) support; prehistoric[card id] = Prehistoric
    pack membership; jurassic[tier] = the blue Jurassic World support's fraction per Prehistoric card on the team."""
    import numpy as np
    from ..mutations import MUTATION_NAMES
    cards, red, _ = names(catalog)
    ids = sorted(cards)
    reds = sorted(red)
    reply = worker().request({"op": "tables", "cards": [cards[i] for i in ids],
                              "borders": [border_names(catalog, b) for b in range(1, 17)],
                              "mutations": [None if m == "None" else m for m in MUTATION_NAMES],
                              "reds": [red[r] for r in reds], "tiers": list(AURA_BORDERS.values())})
    base = np.ones((max(ids) + 1, 17, len(MUTATION_NAMES), 2))
    base[np.array(ids)[:, None, None], np.arange(1, 17)[None, :, None], np.arange(len(MUTATION_NAMES))[None, None, :]] = reply["base"]
    table = np.ones((max(ids) + 1, len(MUTATION_NAMES), max(reds) + 1, 6, 2))
    values = np.array(reply["red"])  # [card, mutation, red, tier, 2]
    for slot, tier in enumerate(AURA_BORDERS):
        table[np.array(ids)[:, None, None], np.arange(len(MUTATION_NAMES))[None, :, None], np.array(reds)[None, None, :], tier] = values[:, :, :, slot]
    prehistoric = np.zeros(max(ids) + 1)
    prehistoric[ids] = reply["prehistoric"]
    jurassic = np.zeros(6)
    jurassic[list(AURA_BORDERS)] = np.array(reply["jurassic"]) / 100
    return base, table, prehistoric, jurassic


def evaluate(catalog, spec, seed, *, fixed=None, scale=None, strip=None, **overrides):
    """((P(A wins), P(B wins), 0, unfinished), exact) for a label spec; side A moves first, and a draw counts as A's loss.

    Optional tweaks, each naming a side (0 = A, 1 = B): fixed=(side, [(HP, ATK) per card]) sets that side's stats
    (event or Tower teams); scale=(side, factor) multiplies its HP and ATK; strip=(side, slot) removes one card's
    ability, stats kept. A battle with an ability or aura his engine marks unsupported gets no answer: ((0,0,0,1), False)."""
    a, b = loadouts(catalog, spec)
    options = {**SEARCH, **overrides, **_tweaks(fixed, scale, strip), "seed": int(seed) % 2 ** 31 or 1}
    reply = worker().request({"a": a, "b": b, "options": options})
    if reply["unsupported"]:
        return (0.0, 0.0, 0.0, 1.0), False
    # User (2026-10-03): a draw (both wiped out, or the turn cap) counts as the attacker A's loss.
    return (reply["a"], reply["b"] + reply["draw"], 0.0, 0.0), bool(reply["exact"])
