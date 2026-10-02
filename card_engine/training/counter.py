"""Find teams that beat a given enemy team, trading win chance against how easy the cards are to obtain.

Availability: a card at a border is obtained with probability 1 / (card rarity x border rarity) per roll, so its
cost is that many expected rolls; a weather card can only drop while its weather is active, so its cost is divided
by that chance (scenario "weather_availability": any-weather fraction x the weather's rate; user, 2026-10-02).
Cards the player already owns cost 1. A team's availability is
log10(sum of its costs). Stats grow as 2^log10(rarity), so a cheaper border trades stats for availability at a
known rate; the search uses that both ways:
  * proposals draw cards and borders in proportion to cost^-alpha (several alphas), not uniformly;
  * local moves swap a card, move one card a border up or down, reorder, or change a support;
  * every team that wins is then pushed to its cheapest borders that keep the win chance.
Objectives win - lambda * availability for several lambdas; --max-rolls caps how hard a card may be to obtain.

Every candidate is evaluated with the label search (labels.evaluate: exact when deterministic, else adaptive
playouts) twice: the player attacking first and the enemy attacking first.

    python -m card_engine.training.counter data/scenarios/<name>.json
"""

import json
import math
import multiprocessing as mp
import os
import random
from pathlib import Path

from ..availability import DEFAULT_WEATHER, roll_rarity
from ..catalog import load_catalog
from ..deck import DECK_FILE, load as load_deck, owned_entries, owned_supports
from ..restricted import entries as restricted_entries, load as load_restricted
from ..mutations import MUTATION_NAMES
from ..simulator.catalog_rules import BLUE_SUPPORTS, RED_SUPPORTS
from ..stats import base_stats
from .labels import compile_spec, evaluate

def _side(cards, borders, mutations, red, blue):
    return {"cards": list(cards), "borders": list(borders), "mutations": [MUTATION_NAMES.index(m) for m in mutations],
            "arts": [0] * len(cards), "red": red[0], "red_tier": red[1] if red[0] else 0,
            "blue": blue[0], "blue_tier": blue[1] if blue[0] else 0}


def _spec(first, second):
    return {key: [first[key], second[key]] for key in first}


_CATALOG = None


def _init():
    global _CATALOG
    _CATALOG = load_catalog()


def _evaluate(job):
    """(player win probability attacking first, ... defending first)."""
    player, enemy, seed = job
    first, _ = evaluate(compile_spec(_CATALOG, _spec(player, enemy)), seed)
    second, _ = evaluate(compile_spec(_CATALOG, _spec(enemy, player)), seed + 1)
    return first[0], second[1]


class Search:
    def __init__(self, scenario, catalog, workers=None, seed=1, max_rolls=None):
        enemy = scenario["enemy"]
        self.enemy = _side(enemy["cards"], enemy["borders"], enemy.get("mutations", ["None"] * len(enemy["cards"])),
                           enemy["red"], enemy["blue"])
        self.catalog = catalog
        # Entries: (card, border, mutation, cost, owned). Owned cards first; then the candidate (card, border) pairs
        # the search may add (the restricted deck in semi/restricted mode).
        self.entries = [(e["card"], e["border"], e.get("mutation", "None"), 1.0, True) for e in scenario.get("owned", [])]
        owned = {(c, b) for c, b, *_ in self.entries}
        self.ranked_borders = sorted(range(1, 17), key=lambda b: (catalog.border(b).rarity, b))
        weather = scenario.get("weather_availability", DEFAULT_WEATHER)
        for card, border in scenario.get("candidates", []):
            cost = roll_rarity(catalog, card, border, weather)
            if (card, border) not in owned and (max_rolls is None or cost <= max_rolls):
                self.entries.append((card, border, "None", cost, False))
        self.index = {}
        for i, (card, border, *_rest) in enumerate(self.entries):
            self.index.setdefault((card, border), i)
        # Base stats (HP + ATK) of every entry: stats = 2^log10(card x border rarity) x weather x card modifier, so for a
        # target stat level each card has a cheapest border that reaches it (weather cards need far rarer borders less).
        self.stats = []
        for card, border, mutation, *_ in self.entries:
            s = base_stats(catalog, card, border, mutation=mutation)
            self.stats.append(math.log10(s.hp + s.attack))
        self.by_card = {}
        for i, (card, *_rest) in enumerate(self.entries):
            self.by_card.setdefault(card, []).append(i)
        self.reds = [tuple(r) for r in scenario["red"]]
        self.blues = [tuple(b) for b in scenario["blue"]]
        self.rng = random.Random(seed)
        self.seen = {}
        self.counter = 0
        self.executor = mp.get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 2) - 1), initializer=_init)
        self._weights = {}

    def close(self):
        self.executor.close()
        self.executor.join()

    # --- availability ---------------------------------------------------------------------------------------
    def availability(self, team):
        return math.log10(sum(self.entries[i][3] for i in team[0]))

    def weights(self, alpha, allowed):
        key = (alpha, id(allowed))
        if key not in self._weights:
            self._weights[key] = [self.entries[i][3] ** -alpha for i in allowed]
        return self._weights[key]

    def _allowed_set(self, allowed):
        key = ("set", id(allowed))
        if key not in self._weights:
            self._weights[key] = set(allowed)
        return self._weights[key]

    def draw(self, alpha, allowed, exclude_cards=()):
        while True:
            i = self.rng.choices(allowed, weights=self.weights(alpha, allowed))[0]
            if self.entries[i][0] not in exclude_cards:
                return i

    # --- evaluation -----------------------------------------------------------------------------------------
    def key(self, team):
        return (tuple(team[0]), team[1], team[2])

    def score(self, teams):
        fresh = [t for t in {self.key(t): t for t in teams}.values() if self.key(t) not in self.seen]
        jobs = []
        for lineup, red, blue in fresh:
            chosen = [self.entries[i] for i in lineup]
            player = _side([e[0] for e in chosen], [e[1] for e in chosen], [e[2] for e in chosen], red, blue)
            self.counter += 2
            jobs.append((player, self.enemy, self.counter))
        for team, probs in zip(fresh, self.executor.map(_evaluate, jobs, chunksize=4)):
            self.seen[self.key(team)] = probs
        return [(self.seen[self.key(t)], t) for t in teams]

    @staticmethod
    def win(probs):
        return (probs[0] + probs[1]) / 2

    # --- proposals ------------------------------------------------------------------------------------------
    def at_level(self, card, level, allowed_set):
        """The cheapest allowed entry of `card` whose base stats reach `level` (log10 HP+ATK), if any."""
        options = [i for i in self.by_card.get(card, ()) if i in allowed_set and self.stats[i] >= level]
        return min(options, key=lambda i: self.entries[i][3]) if options else None

    def stat_team(self, allowed, allowed_set):
        """Reverse-engineered proposal: a random stat level, then each of 4 random cards at its cheapest border reaching it."""
        levels = [self.stats[i] for i in allowed]
        level = self.rng.uniform(min(levels), max(levels))
        cards = list({self.entries[i][0] for i in allowed})
        self.rng.shuffle(cards)
        lineup = []
        for card in cards:
            i = self.at_level(card, level, allowed_set)
            if i is not None:
                lineup.append(i)
            if len(lineup) == 4:
                return (tuple(lineup), self.rng.choice(self.reds), self.rng.choice(self.blues))
        return self.random_team(0.3, allowed)

    def random_team(self, alpha, allowed):
        lineup, cards = [], set()
        while len(lineup) < 4:
            i = self.draw(alpha, allowed, cards)
            cards.add(self.entries[i][0])
            lineup.append(i)
        return (tuple(lineup), self.rng.choice(self.reds), self.rng.choice(self.blues))

    def neighbours(self, team, allowed, owned_only):
        lineup, red, blue = team
        used = {self.entries[i][0] for i in lineup}
        out = []
        for slot, i in enumerate(lineup):
            others = used - {self.entries[i][0]}
            for alpha in (0.3, 0.6, 1.0):
                for _ in range(6):
                    out.append((lineup[:slot] + (self.draw(alpha, allowed, others),) + lineup[slot + 1:], red, blue))
            if not owned_only:  # another card at this card's stat level or higher (cheapest border reaching it)
                level = self.stats[i] + self.rng.choice((0.0, 0.3, 1.0))
                allowed_set = self._allowed_set(allowed)
                for card in self.rng.sample(sorted({self.entries[j][0] for j in allowed}), 12):
                    j = self.at_level(card, level, allowed_set)
                    if j is not None and card not in others:
                        out.append((lineup[:slot] + (j,) + lineup[slot + 1:], red, blue))
            if not owned_only:  # one border up or down
                card, border = self.entries[i][0], self.entries[i][1]
                rank = self.ranked_borders.index(border)
                for step in (-1, 1):
                    if 0 <= rank + step < 16:
                        j = self.index.get((card, self.ranked_borders[rank + step]))
                        if j is not None:
                            out.append((lineup[:slot] + (j,) + lineup[slot + 1:], red, blue))
        for a in range(4):
            for b in range(a + 1, 4):
                swapped = list(lineup)
                swapped[a], swapped[b] = swapped[b], swapped[a]
                out.append((tuple(swapped), red, blue))
        out += [(lineup, r, blue) for r in self.reds if r != red] + [(lineup, red, b) for b in self.blues if b != blue]
        return out

    def cheapest_borders(self, team, tolerance=0.02):
        """Lower each card's border while the win chance stays within `tolerance` of the original."""
        (probs, _), = self.score([team])
        target = self.win(probs) - tolerance
        lineup = list(team[0])
        improved = True
        while improved:
            improved = False
            for slot, i in enumerate(lineup):
                card, border, _, _, owned = self.entries[i]
                if owned:
                    continue
                rank = self.ranked_borders.index(border)
                for cheaper in self.ranked_borders[:rank]:
                    j = self.index.get((card, cheaper))
                    if j is None:
                        continue
                    trial = (tuple(lineup[:slot] + [j] + lineup[slot + 1:]), team[1], team[2])
                    (p, _), = self.score([trial])
                    if self.win(p) >= target:
                        lineup[slot] = j
                        improved = True
                        break
        return (tuple(lineup), team[1], team[2])

    # --- search ---------------------------------------------------------------------------------------------
    def run(self, *, lambdas, samples=2000, keep=30, rounds=4, owned_only=False):
        allowed = [i for i, e in enumerate(self.entries) if e[4] or not owned_only]
        alphas = (0.3, 0.6, 1.0) if not owned_only else (0.0,)
        allowed_set = self._allowed_set(allowed)
        proposals = [self.random_team(self.rng.choice(alphas), allowed) for _ in range(samples // 2)]
        proposals += [self.stat_team(allowed, allowed_set) if not owned_only else self.random_team(0.0, allowed)
                      for _ in range(samples - samples // 2)]
        start = self.score(proposals)
        finalists = []
        for lam in lambdas:
            objective = lambda probs, team: self.win(probs) - lam * self.availability(team)
            best = sorted(start, key=lambda r: -objective(*r))[:keep]
            for _ in range(rounds):
                candidates = []
                for _, team in best[:keep // 2]:
                    options = self.neighbours(team, allowed, owned_only)
                    candidates += self.rng.sample(options, min(len(options), 50))
                unique = {self.key(t): (p, t) for p, t in self.score(candidates) + best}
                best = sorted(unique.values(), key=lambda r: -objective(*r))[:keep]
            finalists += [team for _, team in best[:6]]
        if not owned_only:
            for team in finalists:
                if self.win(self.seen[self.key(team)]) > 0:
                    self.cheapest_borders(team)
        return [(p, (k[0], k[1], k[2])) for k, p in self.seen.items()
                if owned_only is False or all(self.entries[i][4] for i in k[0])]

    def describe(self, probs, team):
        entries = [self.entries[i] for i in team[0]]
        return {"lineup": [f"{self.catalog.card(e[0]).name} ({self.border_name(e[1])}{', owned' if e[4] else ''})" for e in entries],
                "red": team[1], "blue": team[2], "win_attacking_first": round(probs[0], 3), "win_defending": round(probs[1], 3),
                "availability_log10_rolls": round(self.availability(team), 2),
                "hardest_card": max(((self.catalog.card(e[0]).name, f"1 in {e[3]:.3g}") for e in entries if not e[4]),
                                    key=lambda x: float(x[1][5:]), default=None)}

    def border_name(self, border):
        return {1: "no border", 2: "Pl", 3: "Cr", 4: "CrPl", 5: "Ru", 6: "RuPl", 7: "RuCr", 8: "RuCrPl", 9: "Ga", 10: "GaPl",
                11: "GaCr", 12: "GaCrPl", 13: "GaRu", 14: "GaRuPl", 15: "GaRuCr", 16: "GaRuCrPl"}[border]


def candidates(rows, search, limit=8, max_shared=2):
    """The `limit` best teams by mean win chance (ties: more available first), each sharing at most `max_shared`
    cards with any team already chosen, so the list offers real alternatives rather than one core."""
    out, chosen = [], []
    for probs, team in sorted(rows, key=lambda r: (-search.win(r[0]), search.availability(r[1]))):
        cards = frozenset(search.entries[i][0] for i in team[0])
        if all(len(cards & other) <= max_shared for other in chosen):
            chosen.append(cards)
            out.append(search.describe(probs, team))
            if len(out) == limit:
                break
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-rolls", type=float, help="leave out cards harder to obtain than this many expected rolls")
    parser.add_argument("--teams", type=int, default=8)
    parser.add_argument("--dump", help="write every evaluated team (JSON lines) here")
    parser.add_argument("--deck", default=str(DECK_FILE), help="the player's deck (python -m card_engine.deck)")
    parser.add_argument("--mode", choices=("own", "semi", "restricted"), default="semi",
                        help="own: only the player's deck; semi: the deck plus cards from the restricted deck; "
                             "restricted: only the restricted deck (player-base availability, python -m card_engine.restricted)")
    parser.add_argument("--no-limited", action="store_true", help="leave the restricted deck's Limited exceptions out")
    args = parser.parse_args()
    scenario = json.loads(Path(args.scenario).read_text())
    catalog = load_catalog()
    if args.mode in ("own", "semi"):
        if not Path(args.deck).exists():
            raise SystemExit(f"No deck at {args.deck}; add cards with python -m card_engine.deck add ...")
        deck = load_deck(args.deck)
        scenario["owned"] = [{"card": c, "border": b, "mutation": m} for c, b, m in owned_entries(deck)]
        red, blue = owned_supports(deck)
        scenario["red"], scenario["blue"] = [list(r) for r in red] or [[0, 0]], [list(b) for b in blue] or [[0, 0]]
    else:
        scenario["owned"] = []
        # The player base: every support at every tier.
        scenario["red"] = [[s, t] for s, (_, _, tiers) in RED_SUPPORTS.items() for t in range(1, 6) if tiers[t - 1] is not None]
        scenario["blue"] = [[s, t] for s, tiers in BLUE_SUPPORTS.items() for t in range(1, 6) if tiers[t - 1] is not None]
    scenario["candidates"] = ([] if args.mode == "own" else
                              restricted_entries(load_restricted(), catalog, limited=not args.no_limited))
    search = Search(scenario, catalog, workers=args.workers, max_rolls=args.max_rolls)
    try:
        # Owned cards alone first (they cost nothing but are a small share of all entries), then everything.
        rows = []
        if scenario["owned"]:
            rows += search.run(lambdas=(0.0,), samples=args.samples, owned_only=True)
        if scenario["candidates"]:
            rows += search.run(lambdas=(0.0, 0.01, 0.03), samples=args.samples)
    finally:
        search.close()
    if args.dump:
        with open(args.dump, "w") as handle:
            for probs, team in rows:
                handle.write(json.dumps(search.describe(probs, team)) + "\n")
    print(json.dumps({"candidates": candidates(rows, search, args.teams), "evaluated_teams": len(search.seen)}, indent=2))


if __name__ == "__main__":
    main()
