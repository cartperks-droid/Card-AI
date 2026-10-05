"""Teams written by name, for the matchup and generator commands.

A card is "Name[+Art][@Border][/Mutation]", e.g. "Vampire Lord@GaPl", "Good Boy@Pl/Storm", "Astraeus+Virgo", "Archer".
Astraeus needs its art: each art is its own card with a fixed ability (user). A support is
"Name[@Tier]", e.g. "Stormcaller@Galaxy", "Guardian Angel@Ruby". Names match like the deck tool: an ID, an exact
name, or a unique case-insensitive part of one. Borders: none, Pl, Cr, CrPl, Ru, ..., GaRuCrPl. Tiers: base,
Platinum, Crystal, Ruby, Galaxy (or 1-5). Mutations: Storm, Snow, ..., Manga; weather cards cannot mutate.
"""

from .deck import BORDERS, TIERS, _match
from .ownership import parse_rarity
from .mutations import MUTATION_NAMES

FIELDS = ("cards", "borders", "mutations", "arts", "red", "red_tier", "blue", "blue_tier")
ASTRAEUS = 56
# At most one copy per team (user, 2026-10-04): Fate Seamstress and Parallax (a second copy doesn't work; DaddyDrago's
# tower tool limits both too), Time Lord Stryx (Better Days doesn't work twice; his engine lets two copies revive
# each other in a loop), Hades (user, 2026-10-05: two copies don't work).
SINGLE_COPY = {240: "Fate Seamstress", 65: "Parallax", 250: "Time Lord Stryx", 49: "Hades"}
# Astraeus's arts (input "arts": index + 1, 0 = not Astraeus). Each is a separate card with one Constellar ability.
ASTRAEUS_ARTS = ("scorpio", "aquarius", "virgo", "gemini", "sagittarius", "taurus", "cancer")


def _choice(query, options, kind):
    """An option by exact name (case-insensitive) or 1-based number."""
    folded = [o.casefold() for o in options]
    if query.casefold() in folded:
        return folded.index(query.casefold())
    if query.isdigit() and 1 <= int(query) <= len(options):
        return int(query) - 1
    raise SystemExit(f"Unknown {kind} {query!r}; use one of {', '.join(options)}")


def parse_card(catalog, text):
    """(card id, border id, mutation index, art) from "Name[+Art][@Border][/Mutation]"."""
    rest, _, mutation = text.partition("/")
    rest, _, border = rest.partition("@")
    name, _, art_name = rest.partition("+")
    card, _ = _match(name.strip(), [(c.id, c.name) for c in catalog.cards], "Card")
    art = 0
    if card == ASTRAEUS:
        if not art_name.strip():
            raise SystemExit(f"Astraeus needs its art, e.g. Astraeus+Virgo ({', '.join(a.title() for a in ASTRAEUS_ARTS)})")
        art = _choice(art_name.strip(), ASTRAEUS_ARTS, "art") + 1
    elif art_name.strip():
        raise SystemExit(f"Only Astraeus has an art: {text!r}")
    border_id = _choice(border.strip(), BORDERS, "border") + 1 if border.strip() else 1
    mutation_index = 0
    if mutation.strip():
        mutation_index = _choice(mutation.strip(), MUTATION_NAMES, "mutation")
        if mutation_index and catalog.card(card).weather_id != 1:
            raise SystemExit(f"{catalog.card(card).name} is a weather card and cannot mutate")
    return card, border_id, mutation_index, art


def parse_support(catalog, color, text):
    """(support id, tier) from "Name[@Tier]"; empty text = no support (0, 0)."""
    if not text:
        return 0, 0
    name, _, tier = text.partition("@")
    supports = catalog.red_supports if color == "red" else catalog.blue_supports
    support, _ = _match(name.strip(), [(s.id, s.name) for s in supports], f"{color.title()} support")
    return support, _choice(tier.strip(), TIERS, "tier") + 1 if tier.strip() else 1


def side(cards, red=(0, 0), blue=(0, 0)):
    """One side's label fields from [(card, border, mutation, art)] and (id, tier) supports."""
    return {"cards": [c[0] for c in cards], "borders": [c[1] for c in cards],
            "mutations": [c[2] for c in cards], "arts": [c[3] for c in cards],
            "red": red[0], "red_tier": red[1] if red[0] else 0, "blue": blue[0], "blue_tier": blue[1] if blue[0] else 0}


def parse_side(catalog, cards, red=None, blue=None):
    if len(cards) != 4:
        raise SystemExit(f"A team needs 4 cards, got {len(cards)}: {cards}")
    return side([parse_card(catalog, c) for c in cards], parse_support(catalog, "red", red), parse_support(catalog, "blue", blue))


def spec(first, second):
    """A label spec with `first` as side A (attacking first) and `second` as side B."""
    return {key: [first[key], second[key]] for key in FIELDS}


def fixed_stats(hp, attack):
    """Fixed enemy stats (HP, ATK, HP multiplier applies) for battle modes that set them; "2.5M" style numbers."""
    return parse_rarity(hp), parse_rarity(attack), False


def describe(catalog, team, stats=None):
    """Readable lineup and supports of one side; stats: its fixed starting stats, if any."""
    cards = []
    for card, border, mutation, art in zip(team["cards"], team["borders"], team["mutations"], team["arts"]):
        text = catalog.card(card).name + (f"+{ASTRAEUS_ARTS[art - 1].title()}" if art else "")
        text += f"@{BORDERS[border - 1]}" if border > 1 else ""
        cards.append(text + (f"/{MUTATION_NAMES[mutation]}" if mutation else ""))
    supports = {}
    for color, table in (("red", catalog.red_supports), ("blue", catalog.blue_supports)):
        if team[color]:
            name = next(s.name for s in table if s.id == team[color])
            supports[color] = f"{name}@{TIERS[team[color + '_tier'] - 1]}"
    if stats:
        supports["stats"] = {"hp": stats[0], "attack": stats[1], **({"hp_times_card_multiplier": True} if stats[2] else {})}
    return {"cards": cards, **supports}
