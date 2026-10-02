"""How likely a random player owns a given team, from the index's per-card copy counts.

Per card: the index counts copies (duplicates, possibly crafted copies) held by everyone who ever played
(1.1M visits), while opponents come from the active players (~289 online, maybe ~350 regulars; user,
2026-10-02). So the divisor is an effective population N: with m_i = copies_i / N and copies spread as a
negative binomial with dispersion d (small d: duplicates pile up on heavy players), a random opponent owns
card i with p_i = 1 - (1 + m_i/d)^(-d). N and d are fitted together to inspected decks (cards with very
different copy counts pin both down); `set-total` only gives the starting value.

Per team (user, 2026-10-02): the joint probability lies between the bounds
    P_min = max(0, sum p_i - (k - 1))   (memberships forced to overlap: e.g. 3 x 50 of 64 balls -> >= 22 ABC)
    P_max = min p_i                     (all memberships nested: progression luck)
and is P_ind = prod p_i under independence. A 2-D independence parameter places it:
    P = lambda * P_ind + (1 - lambda) * ((1 - s) * P_min + s * P_max)
lambda is the weight on independence (the user's table gives, for each lambda, the range s in [0, 1]);
s is the coupling direction (1: fully correlated, 0: maximally exclusive). d, lambda and s are fitted to the
decks of inspected players (the user's own deck counts as one).

Progression (user, 2026-10-02): players show a display card, usually their best; in a lobby these range from
~30T to ~50qd, sometimes 1-10qn. log10 of a player's best card measures their progression (luck x rolls), and
across players it is a mixture of bell curves, fitted to display observations (1-3 Gaussians, chosen by BIC).
A player at progression x owns a card whose rolls-to-obtain (any border, weather availability included) is r
with probability 1 - exp(-kappa 10^x / r); kappa (how a display card maps to rolls; ~1) is fitted so the model
reproduces known owner counts at a border (e.g. 6 players own Infected Maw at GaCr; user, 2026-10-02), over
the active population (`set-total`). An index tile with no count means (nearly) every player owns it (user):
recorded as "all", it asks the model for at least UNIVERSAL ownership.
A team's probability is the average over the mixture of the product: owning cards is independent given
progression, and the shared progression is what couples them, so it lands inside the bounds above.

    python -m card_engine.ownership add-display PlayerName 50qd                     # a player's display card rarity
    python -m card_engine.ownership add-owners "Infected Maw@GaCr" 6                # known owner count at a border
    python -m card_engine.ownership add-owners "Archer@Pl" all                      # no index count: (nearly) everyone
    python -m card_engine.ownership fit-progression
    python -m card_engine.ownership set-total 350
    python -m card_engine.ownership set-copies "Malik The Sovereign" 1234
    python -m card_engine.ownership add-sample PlayerName Malik Kira Hathor ...   # every card that player owns
    python -m card_engine.ownership calibrate
    python -m card_engine.ownership team Malik Kira Hathor "Noveau Riche"
    python -m card_engine.ownership show
"""

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np

from .availability import roll_rarity

from .catalog import load_catalog
from .deck import BORDERS, _border, _match

ROOT = Path(__file__).resolve().parents[1]
OWNERSHIP_FILE = ROOT / "data" / "ownership.json"
LAMBDA_TABLE = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)


def load(path=OWNERSHIP_FILE):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {"total_players": None, "effective_players": None, "copies": {}, "dispersion": 1.0, "independence": None,
            "coupling": None, "samples": {}, "displays": {}, "progression": None, "kappa": 1.0}


def save(data, path=OWNERSHIP_FILE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


# --- per card ---------------------------------------------------------------------------------------------
def owned_probability(copies, players, dispersion):
    """P(a player owns at least one copy): negative binomial with mean copies/players and dispersion d."""
    return 1 - (1 + copies / players / dispersion) ** -dispersion


def ownership_rates(data, dispersion=None):
    """Per-card P(a random active opponent owns it), with the fitted effective population if there is one."""
    players = data.get("effective_players") or data["total_players"]
    if not players:
        raise SystemExit("Set the number of players first: python -m card_engine.ownership set-total N")
    d = data["dispersion"] if dispersion is None else dispersion
    return {int(card): owned_probability(count, players, d) for card, count in data["copies"].items()}


# --- per team ---------------------------------------------------------------------------------------------
def bounds(probabilities):
    """(P_min, P_ind, P_max) for owning every one of the cards."""
    k = len(probabilities)
    return max(0.0, sum(probabilities) - (k - 1)), math.prod(probabilities), min(probabilities)


def team_range(probabilities, independence):
    """The user's table: for a given lambda, the range over the coupling direction s."""
    return team_probability(probabilities, independence, 0.0), team_probability(probabilities, independence, 1.0)


def team_probability(probabilities, independence, coupling):
    low, independent, high = bounds(probabilities)
    return independence * independent + (1 - independence) * ((1 - coupling) * low + coupling * high)


# --- calibration ------------------------------------------------------------------------------------------
def _decks(data, min_samples):
    decks = [set(cards) for cards in data["samples"].values()]
    if len(decks) < min_samples:
        raise SystemExit(f"Calibration needs at least {min_samples} inspected players' decks (have {len(decks)})")
    return decks


def fit_population(data, min_samples=10):
    """(effective players N, dispersion d) whose per-card ownership rates best match the inspected decks
    (least squares; N on a log grid from 1/10 to 1000x the starting value, d from 0.01 to 100)."""
    decks = _decks(data, min_samples)
    observed = {int(card): sum(int(card) in deck for deck in decks) / len(decks) for card in data["copies"]}
    start = data["total_players"]
    best = None
    for n in [start * 10 ** (k / 10) for k in range(-10, 31)]:
        trial = dict(data, total_players=n)
        for d in [10 ** (k / 10) for k in range(-20, 21)]:
            error = sum((rate - observed[card]) ** 2 for card, rate in ownership_rates(trial, d).items())
            if best is None or error < best[0]:
                best = (error, n, d)
    return best[1], best[2]


def fit_independence(data, sizes=(2, 3), min_samples=10):
    """(lambda, s) minimising the squared error to the observed co-ownership of card pairs and triples."""
    decks = _decks(data, min_samples)
    rates = ownership_rates(data)
    known = sorted(card for card in rates if any(card in deck for deck in decks))
    checks = []
    for size in sizes:
        for combo in itertools.combinations(known, size):
            observed = sum(all(card in deck for card in combo) for deck in decks) / len(decks)
            checks.append(([rates[card] for card in combo], observed))
    if not checks:
        raise SystemExit("No card pairs with copy counts appear in the inspected decks")
    grid = [(l / 50, s / 50) for l in range(51) for s in range(51)]
    best = min(grid, key=lambda ls: sum((team_probability(p, *ls) - o) ** 2 for p, o in checks))
    return best[0], best[1], len(checks)


# --- progression (display cards) ----------------------------------------------------------------------------
SUFFIXES = {"": 1, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12, "qd": 1e15, "qn": 1e18, "sx": 1e21, "sp": 1e24}


def parse_rarity(text):
    """'50qd', '1.5T', '7,500,000' -> float."""
    match = re.fullmatch(r"(?:1/)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([a-zA-Z]*)", text.strip())
    if not match or match.group(2).lower() not in SUFFIXES:
        raise SystemExit(f"Cannot read rarity '{text}' (e.g. 30T, 50qd, 1qn)")
    return float(match.group(1).replace(",", "")) * SUFFIXES[match.group(2).lower()]


def fit_mixture(values, max_components=3, iterations=300):
    """Gaussian mixture on `values` (log10 rarities), components chosen by BIC: [(weight, mean, sd)]."""
    x = np.asarray(values, dtype=float)
    if len(x) < 5:
        raise SystemExit(f"Need at least 5 display observations (have {len(x)})")
    best = None
    for k in range(1, min(max_components, len(x) // 5) + 1):
        means = np.quantile(x, (np.arange(k) + 0.5) / k)
        sds = np.full(k, max(x.std(), 0.1))
        weights = np.full(k, 1 / k)
        for _ in range(iterations):
            density = weights * np.exp(-0.5 * ((x[:, None] - means) / sds) ** 2) / (sds * math.sqrt(2 * math.pi))
            resp = density / density.sum(axis=1, keepdims=True)
            totals = resp.sum(axis=0)
            weights = totals / len(x)
            means = (resp * x[:, None]).sum(axis=0) / totals
            sds = np.sqrt(np.maximum((resp * (x[:, None] - means) ** 2).sum(axis=0) / totals, 0.05 ** 2))
        density = weights * np.exp(-0.5 * ((x[:, None] - means) / sds) ** 2) / (sds * math.sqrt(2 * math.pi))
        bic = -2 * np.log(density.sum(axis=1)).sum() + (3 * k - 1) * math.log(len(x))
        if best is None or bic < best[0]:
            best = (bic, [(float(w), float(m), float(s)) for w, m, s in zip(weights, means, sds)])
    return best[1]


UNIVERSAL = 0.95


def fit_kappa(data, catalog):
    """kappa (log grid 1e-4..1e4) minimising the squared log error of the predicted owner counts."""
    counts = data.get("owner_counts", {})
    if not counts:
        return data.get("kappa", 1.0)
    if not data.get("total_players"):
        raise SystemExit("Set the active player count first: python -m card_engine.ownership set-total N")
    items = [((int(key.split("@")[0]), int(key.split("@")[1])), owners) for key, owners in counts.items()]

    def error(kappa):
        trial = dict(data, kappa=kappa)
        total = 0.0
        for item, owners in items:
            share = progression_team(trial, catalog, [item])
            if owners == "all":  # only a shortfall below UNIVERSAL counts (on the not-owned share)
                total += max(0.0, math.log(max(1 - share, 1e-9)) - math.log(1 - UNIVERSAL)) ** 2
            else:
                total += (math.log(max(share * data["total_players"], 1e-3)) - math.log(max(owners, 0.5))) ** 2
        return total
    return min((10 ** (k / 20) for k in range(-80, 81)), key=error)


def card_rolls(catalog, card, border=1):
    """Expected rolls to obtain `card` at `border` or a rarer one (weather availability included)."""
    floor = catalog.border(border).rarity
    return 1 / sum(1 / roll_rarity(catalog, card, b) for b in range(1, 17) if catalog.border(b).rarity >= floor)


def progression_team(data, catalog, cards, points=600):
    """P(a random player owns every card, each at its border or rarer): cards are ids or (id, border) pairs;
    the mixture average of prod_i (1 - exp(-kappa 10^x / r_i))."""
    mixture = data["progression"]
    low = min(m - 5 * s for _, m, s in mixture)
    high = max(m + 5 * s for _, m, s in mixture)
    x = np.linspace(low, high, points)
    density = sum(w * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi)) for w, m, s in mixture)
    density /= density.sum()
    owned = np.ones_like(x)
    for item in cards:
        card, border = item if isinstance(item, tuple) else (item, 1)
        owned *= -np.expm1(-data.get("kappa", 1.0) * 10 ** x / card_rolls(catalog, card, border))
    return float(density @ owned)


def describe_team(data, catalog, cards):
    if data.get("progression") and (not data.get("copies") or any(isinstance(c, tuple) for c in cards)):
        p = [progression_team(data, catalog, [c]) for c in cards]
        team = progression_team(data, catalog, cards)
        low, independent, high = bounds(p)
        label = lambda c: (f"{catalog.card(c[0]).name} at {BORDERS[c[1] - 1]}+" if isinstance(c, tuple) else catalog.card(c).name)
        lines = [f"{label(c)}: owned by {q:.2%} of players (progression model)" for c, q in zip(cards, p)]
        lines.append(f"Owning all {len(cards)}: {team:.2%} (bounds {low:.2%} - {high:.2%}, independent {independent:.2%})")
        return "\n".join(lines)
    rates = ownership_rates(data)
    missing = [catalog.card(c).name for c in cards if c not in rates]
    if missing:
        raise SystemExit(f"No copy count for: {', '.join(missing)} (python -m card_engine.ownership set-copies ...)")
    probabilities = [rates[c] for c in cards]
    low, independent, high = bounds(probabilities)
    lines = [f"{catalog.card(c).name}: {data['copies'][str(c)]} copies -> owned by {p:.2%} of players"
             for c, p in zip(cards, probabilities)]
    lines.append(f"Owning all {len(cards)}: independent {independent:.2%}, possible {low:.2%} - {high:.2%}")
    if data.get("progression"):
        lines.append(f"Progression model: {progression_team(data, catalog, cards):.2%}")
    if data["independence"] is not None:
        lines.append(f"Calibrated (lambda {data['independence']:.0%}, coupling {data['coupling']:.0%}): "
                     f"{team_probability(probabilities, data['independence'], data['coupling']):.2%}")
    lines += ["| Independence | Possible P(team) |", "|---:|---:|"]
    for independence in LAMBDA_TABLE:
        a, b = team_range(probabilities, independence)
        lines.append(f"| {independence:.0%} | {a:.2%} - {b:.2%} |" if b - a > 5e-5 else f"| {independence:.0%} | {a:.2%} exactly |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m card_engine.ownership", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("set-total", help="starting guess of the player population (active ~289-350)").add_argument("count", type=int)
    p = commands.add_parser("set-copies", help="how many copies of a card exist (the index's bottom-right number)")
    p.add_argument("card")
    p.add_argument("count", type=int)
    p = commands.add_parser("add-sample", help="every card one inspected player owns (for calibration)")
    p.add_argument("player")
    p.add_argument("cards", nargs="+")
    commands.add_parser("calibrate", help="fit the copy dispersion and the 2-D independence parameter")
    p = commands.add_parser("add-owners", help="how many players own a card at a border or rarer (Card@Border N)")
    p.add_argument("card")
    p.add_argument("count", help="a number, or 'all' when the index shows no count")
    p = commands.add_parser("add-display", help="a player's display card rarity, e.g. 50qd (one per player)")
    p.add_argument("player")
    p.add_argument("rarity")
    commands.add_parser("fit-progression", help="fit the mixture of bell curves to the display observations")
    p = commands.add_parser("set-independence", help="set lambda and the coupling direction s by hand (0-1 each)")
    p.add_argument("independence", type=float)
    p.add_argument("coupling", type=float)
    commands.add_parser("team", help="likelihood that a random player owns these cards (Card@Border: at that border "
                                     "or rarer)").add_argument("cards", nargs="+")
    commands.add_parser("show", help="print the data")
    args = parser.parse_args(argv)
    catalog = load_catalog()
    data = load()
    card_id = lambda query: _match(query, [(c.id, c.name) for c in catalog.cards], "Card")[0]

    def team_item(query):  # "Malik@RuCrPl": that card at that border or rarer
        name, _, border = query.partition("@")
        return (card_id(name), _border(border)) if border else card_id(name)
    if args.command == "set-total":
        data["total_players"] = args.count
    elif args.command == "set-copies":
        data["copies"][str(card_id(args.card))] = args.count
    elif args.command == "add-display":
        data.setdefault("displays", {})[args.player] = parse_rarity(args.rarity)
    elif args.command == "add-owners":
        item = team_item(args.card)
        card, border = item if isinstance(item, tuple) else (item, 1)
        count = "all" if args.count.lower() == "all" else int(args.count)
        data.setdefault("owner_counts", {})[f"{card}@{border}"] = count
    elif args.command == "fit-progression":
        data["progression"] = fit_mixture([math.log10(v) for v in data.get("displays", {}).values()])
        data["kappa"] = fit_kappa(data, catalog)
        print(f"kappa {data['kappa']:.3g}")
        print("Progression mixture: " + "; ".join(f"{w:.0%} around {10 ** m:.3g} (x/÷{10 ** s:.3g})" for w, m, s in data["progression"]))
    elif args.command == "add-sample":
        data["samples"][args.player] = sorted({card_id(c) for c in args.cards})
    elif args.command == "calibrate":
        data["effective_players"], data["dispersion"] = fit_population(dict(data, effective_players=None))
        data["independence"], data["coupling"], checks = fit_independence(data)
        print(f"Effective players {data['effective_players']:.0f}, dispersion {data['dispersion']:.3g}; independence "
              f"{data['independence']:.0%}, coupling {data['coupling']:.0%} ({checks} pair/triple checks)")
    elif args.command == "set-independence":
        if not (0 <= args.independence <= 1 and 0 <= args.coupling <= 1):
            raise SystemExit("Both values must be between 0 and 1")
        data["independence"], data["coupling"] = args.independence, args.coupling
    elif args.command == "team":
        print(describe_team(data, catalog, [team_item(c) for c in args.cards]))
        return
    else:
        print(json.dumps(data, indent=1))
        return
    save(data)
    if args.command not in ("calibrate", "fit-progression"):
        print("Saved")


if __name__ == "__main__":
    main()
