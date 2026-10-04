"""Tower battles, from DaddyDrago's Tower Cheese Maker (engine/tower.ts and depths-ui.js at sim_js/ENGINE_COMMIT).

Every enemy card on a floor has the same power: ceil(2 sqrt((6000 + floor^3 * 50) / 2) * 4^(difficulty id - 1)), with
ids Normal 1, Hard 2, Extreme 3, Hell 5, Impossible 6. A card starts at HP = power (times its HP multiplier on Normal
and Impossible) and ATK = ceil(power / 2); enemies are borderless, unmutated and have no supports. Every fifth floor
has a fixed team; the others are drawn by the game. The player attacks first.
"""

import math

from .teams import ASTRAEUS, ASTRAEUS_ARTS, side

DIFFICULTIES = {"Normal": 1, "Hard": 2, "Extreme": 3, "Hell": 5, "Impossible": 6}
FLOORS = 105
# His names (depths-ui.js TOWER_FIXED); floor 65's four Astraeus are Virgo, Scorpio, Taurus and Gemini.
FIXED_TEAMS = {
    5: ("Good Boy", "Good Boy", "Good Boy", "Shining Armor"),
    10: ("Sorcerer", "Sorcerer", "Trainee", "Trainee"),
    15: ("Chronus The Hoarder", "Greedy Belly", "Greedy Belly", "Arthur of Excalibur"),
    20: ("Demon Hunter", "Gunslinger", "Stone Scientist", "Darling"),
    25: ("Black Cat", "Black Cat", "Black Cat", "Black Cat"),
    30: ("Crown Prince", "Three-Legged Golden Crow", "Leviathan", "Malik The Sovereign"),
    35: ("Ice Queen", "Kitsune", "A0-ON1", "AK4-ON1"),
    40: ("Zeus", "Arcane Avian", "Zeus", "Arcane Avian"),
    45: ("Frankenstein", "Phoenix", "Phoenix", "Gideon The Insatiable"),
    50: ("Admiral Ice", "Ice Queen", "Hoarfrost Phoenix", "Ice Queen"),
    55: ("Boreas", "Wind Spirit", "Wind Spirit", "Wind Spirit"),
    60: ("Bad Boys", "Poseidon", "Hades", "Lilith The Enchantress"),
    65: ("Astraeus", "Astraeus", "Astraeus", "Astraeus"),
    70: ("Cronus", "Ixion", "Cronus", "Sciron"),
    75: ("Deus Ex", "Bad Boys", "Bad Boys", "Morpheus The Slumberer"),
    80: ("Mastermind", "Domain Master", "Kira", "Priest"),
    85: ("Savior", "Lucifer", "Lucifer", "Lucifer"),
    90: ("Gilgamesh", "Ragon", "Fafnir", "Raze The Destroyer"),
    95: ("Shu", "Sekhmet", "Set", "Ra"),
    100: ("Shuten-dōji", "Susanoo", "Tsukuyomi", "Amaterasu"),
    105: ("Heaven's Armor", "Hell's Army", "Judgment Day", "Sable The Envious"),
}
FIXED_ARTS = {65: ("virgo", "scorpio", "taurus", "gemini")}


def difficulty(name):
    folded = {d.casefold(): d for d in DIFFICULTIES}
    if name.casefold() not in folded:
        raise SystemExit(f"Unknown tower difficulty {name!r}; use one of {', '.join(DIFFICULTIES)}")
    return folded[name.casefold()]


def power(floor, level):
    stage = max(1, int(floor))
    return math.ceil(2 * math.sqrt((6000 + stage ** 3 * 50) / 2) * 4 ** (DIFFICULTIES[level] - 1))


def stats(floor, level):
    """The floor's fixed stats: (HP before the card's multiplier, ATK, whether HP takes the card's HP multiplier)."""
    p = power(floor, level)
    return float(p), float(math.ceil(p / 2)), level in ("Normal", "Impossible")


def fixed_team(catalog, floor):
    """The floor's fixed enemy team as a side, or None when the game draws it."""
    if floor not in FIXED_TEAMS:
        return None
    from .simulator.drago import names
    ours = {his: card for card, his in names(catalog)[0].items()}
    arts = FIXED_ARTS.get(floor, ("",) * 4)
    cards = [ours[name] for name in FIXED_TEAMS[floor]]
    return side([(card, 1, 0, ASTRAEUS_ARTS.index(art) + 1 if card == ASTRAEUS else 0) for card, art in zip(cards, arts)])


def add_arguments(parser):
    """--enemy-stats and --tower for the commands that take an enemy."""
    parser.add_argument("--enemy-stats", nargs=2, metavar=("HP", "ATK"),
                        help="every enemy card's starting HP and ATK (battle modes with fixed enemy stats)")
    parser.add_argument("--tower", nargs=2, metavar=("FLOOR", "DIFFICULTY"),
                        help="a tower floor (1-105) and difficulty (Normal, Hard, Extreme, Hell, Impossible): its stats, "
                             "and its enemy team on fixed floors (every fifth) when --enemy is not given")


def enemy(catalog, args, parser):
    """(enemy side or None, fixed stats or None) from --enemy, --enemy-red, --enemy-blue, --enemy-stats and --tower. A
    fixed-stat enemy is borderless: the mode sets its stats; tower enemies are also unmutated and without supports."""
    from .teams import fixed_stats, parse_side
    if args.tower and args.enemy_stats:
        parser.error("give --tower or --enemy-stats, not both")
    team = parse_side(catalog, args.enemy, args.enemy_red, args.enemy_blue) if args.enemy else None
    fixed = fixed_stats(*args.enemy_stats) if args.enemy_stats else None
    if args.tower:
        floor, level = int(args.tower[0]), difficulty(args.tower[1])
        if not 1 <= floor <= FLOORS:
            parser.error(f"tower floors run 1-{FLOORS}")
        fixed = stats(floor, level)
        if team is None:
            team = fixed_team(catalog, floor)
            if team is None:
                parser.error(f"floor {floor} has no fixed team: give the four enemies with --enemy")
        team.update(mutations=[0] * 4, red=0, red_tier=0, blue=0, blue_tier=0)
    if fixed is not None and team is not None:
        team["borders"] = [1] * 4
    return team, fixed


def engine_stats(catalog, cards, fixed):
    """Per card (HP, ATK) for the engine's fixed tweak, from fixed = (HP, ATK, HP multiplier applies)."""
    from .simulator.drago import hp_multipliers
    hp, attack, multiplied = fixed
    table = hp_multipliers(catalog)
    return [(math.ceil(hp * table[card]) if multiplied else hp, attack) for card in cards]
