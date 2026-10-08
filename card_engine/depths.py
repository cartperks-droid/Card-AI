"""Depths, from DaddyDrago's engine (engine/depths.ts and simulation.ts at sim_js/ENGINE_COMMIT), played by our engine.

A run climbs floors until its first loss; every floor is a fresh battle of the player's team, attacking first, against
four enemies the game draws. Each enemy is drawn, duplicates allowed, from the floor's pool by weight (his weather
weights: Storm and Snow 0.5, Aurora 0.3, ..., 1 for base cards). A card is in the pool once its ATK x HP (base,
borderless) is below the floor's budget, 3000 + floor^2.75 x 40, unless the game excludes it (his eligibility: no
unobtainable, expiring, boss or holiday cards, nor Vampire Lord, Parallax or Samurai) or the player banned it (up to
14). Every enemy has power ceil(sqrt(2 x budget)), ATK power / 2 and HP power times its HP multiplier; enemies are
borderless, unmutated and without supports. Hard depths (user, 2026-10-07: the mode of most interest) keeps the
floor's pool but takes the stats of floor x 10.

A floor's enemies are independent of the others, so a team reaches floor F with probability prod_{f < F} p(f),
p(f) its mean win chance over the floor's draws. The curve is sampled at a geometric grid of floors (p held between
grid floors); expected floors cleared is the sum of the survival curve.

Speed (user, 2026-10-07: speedrun teams such as Triceratops, a dinosaur and two Julius trade depth for floors per
hour, since most floors are far weaker than they): our engine also gives each battle's expected length in turns, and
his timing model (depths-time.ts) turns it into seconds: battle speed 3 (Depths' cap), +1 with the Chrono Shard, +0.25
per battle-speed structure level, the skill tree's bonus, +0.25 per 100 floors up to +4.5; 1.1 s per attack at 1x,
2x/3x/5x/10x faster from the 10th/20th/40th/60th attack; 0.5 s battle start; 0.9 s between floors. A floor's aura packs
are ceil(ceil(sqrt((3000 + floor^2.7 x 40) / 2)) / 500), flat from floor 5,000 (depths-rewards.ts), counted through
the floor a run dies on as his batch summary does. Runs restart from floor 1, so the long-run rate is a run's expected
packs over its expected time.

    python -m card_engine.depths run --ally Parallax "Judgement Day" "Judgement Day" "Robin Hood" --ally-blue Fate --checkpoint data/training_10s/ema.checkpoint
    python -m card_engine.depths run --ally ... --checkpoint ... --simulate      # the engine's curve, speed and packs too
    python -m card_engine.depths search --checkpoint data/training_10s/ema.checkpoint --pool own
    python -m card_engine.depths search --objective speed --optimize-bans 14 --checkpoint ... --pool own   # it builds its own teams
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from .teams import ASTRAEUS, ASTRAEUS_ARTS, side

HARD_STATS = 10  # hard depths: the stats of floor x 10 (his simulation.ts hardMode)
SEARCH_LOG = Path(__file__).resolve().parents[1] / "data" / "depths_search.jsonl"  # every team the engine scored in a
# search and every ban step's per-card gains, the evidence `depths stats` aggregates (user, 2026-10-08)
MAX_BANS = 14
FLOOR_RANGE = (1, 10_000)  # training draws: floors log-uniform over this range (hard depths ~5,000 is about tower
# 105 Impossible's stats; the pool is complete from about floor 150)
HARD_SHARE = 0.8  # training draws that are hard depths (user: of most interest); the rest are normal depths
# Popular ban lists (user, 2026-10-07, from in-game screenshots); --bans takes a name or cards. Odin and Achyls are
# ours; a ban of a card his pool lacks changes nothing.
BAN_PRESETS = {
    "speedrun": ["Achyls", "Amaterasu", "Gilgamesh", "Inari", "Kuchisake-onna", "Limitless Rivals", "Loki", "Mastermind",
                 "Odin", "Pandora", "Piccolo", "Surtr", "The Awakened One", "Zombie Dragon"],
    "pro": ["A0-ON1", "AK4-ON1", "Astraeus", "Immortal Witch", "Julius Leader", "Pangu", "Piccolo", "Priest", "Sciron",
            "Sekhmet", "Shu", "Susanoo", "Yamato no Orochi"],  # a pro's suggestion, likely for the deepest runs (user)
    "drago": ["A0-ON1", "AK4-ON1", "Astraeus", "Immortal Witch", "Julius Leader", "Pangu", "Piccolo", "Priest", "Sciron",
              "Sekhmet", "Shu", "Susanoo", "Yamato no Orochi", "Yeti"],  # DaddyDrago's own (user, 2026-10-08 screenshot)
}
# Timing (his depths-time.ts)
BASE_SPEED, CHRONO_SHARD, STRUCTURE_STEP, SKILL_TREE = 3, 1, 0.25, (0, 0.5, 1, 1.5, 2.5)
FLOOR_SPEED_STEP, FLOOR_SPEED_CAP = 0.25, 4.5
ATTACK_SECONDS, START_SECONDS, BETWEEN_FLOORS = 1.1, 0.5, 0.9
REWARD_CAP_FLOOR = 5000


def budget(floor):
    return 3000 + floor ** 2.75 * 40


def power(floor):
    return math.ceil(math.sqrt(budget(floor) * 2))


def stats(floor, hard=True):
    """The floor's fixed enemy stats: (HP before the card's HP multiplier, ATK, HP multiplier applies)."""
    p = power(max(1, int(floor)) * (HARD_STATS if hard else 1))
    return float(p), p / 2, True


_POOL = {}


def pool(catalog):
    """[(card id, weight, ATK x HP)] for every card Depths can field (his pool, via the worker), and the cards of his
    pool our catalog lacks."""
    if id(catalog) not in _POOL:
        from .simulator import drago
        reply = drago.worker().request({"op": "depths"})
        # by his names as the card mapping folds them (drago.CARD_ALIASES): ours map to his old "Tyranodon",
        # identical to the live "Tyrannodon" his pool fields but for the name and its unobtainable flag
        key = lambda name: drago.CARD_ALIASES.get(drago._norm(name), drago._norm(name))
        ours = {key(his): card for card, his in drago.names(catalog)[0].items()}
        entries = [(ours[key(name)], weight, threshold) for name, weight, threshold in reply["pool"] if key(name) in ours]
        missing = [name for name, *_ in reply["pool"] if key(name) not in ours]
        _POOL[id(catalog)] = entries, missing
    return _POOL[id(catalog)]


def floor_pool(catalog, floor, bans=()):
    """(card ids, weights) unlocked at `floor` (the real floor, also in hard depths), without the banned cards."""
    entries = [(card, weight) for card, weight, threshold in pool(catalog)[0]
               if threshold < budget(floor) and card not in set(bans)]
    return [card for card, _ in entries], [weight for _, weight in entries]


def draw_team(rng, catalog, floor, bans=()):
    """Four enemies drawn by weight from the floor's pool (duplicates allowed), as a side; Astraeus with a random art
    (his engine draws its Constellar ability)."""
    cards, weights = floor_pool(catalog, floor, bans)
    if not cards:
        raise SystemExit(f"Depths floor {floor} has no enemies left to draw (bans: {len(bans)})")
    picks = rng.choices(cards, weights=weights, k=4)
    return side([(card, 1, 0, rng.randint(1, len(ASTRAEUS_ARTS)) if card == ASTRAEUS else 0) for card in picks])


def draw_floor(rng):
    """A training draw: (floor, hard), floors log-uniform over FLOOR_RANGE, hard depths HARD_SHARE of the time."""
    low, high = FLOOR_RANGE
    return int(round(10 ** rng.uniform(math.log10(low), math.log10(high)))), rng.random() < HARD_SHARE


def grid(cap, points):
    """Geometric floors from 1 to cap, distinct."""
    return sorted({max(1, round(cap ** (i / (points - 1)))) for i in range(points)} | {1, cap})


def survival(floors, wins, cap):
    """(expected floors cleared, survival through each grid floor's span, median death floor: the first floor at
    which the chance to have survived falls below a half). p(f) is the win chance at the nearest grid floor at or
    below f, up to cap."""
    expected, alive, curve, median = 0.0, 1.0, [], None
    for start, end, p in zip(floors, [*floors[1:], cap + 1], wins):
        length, p = end - start, min(max(float(p), 0.0), 1.0)
        if median is None and alive * p ** length < 0.5:
            k = 1 if p <= 0 or alive * p < 0.5 else int(math.log(0.5 / alive) / math.log(p)) + 1
            median = start + min(k, length) - 1
        expected += alive * (length if p >= 1.0 else p * (1 - p ** length) / (1 - p))
        alive *= p ** length
        curve.append(alive)
    return expected, curve, median


def enemies(catalog, floors, samples, bans=(), seed=1):
    """{floor: [enemy sides]}: a fixed draw per floor, so teams are compared on the same enemies."""
    rng = random.Random(f"depths-{seed}")
    return {floor: [draw_team(rng, catalog, floor, bans) for _ in range(samples)] for floor in floors}


def model_wins(classifier, teams, drawn, hard=True):
    """[teams, floors]: the model's mean win chance per grid floor, attacking first against the floor's draws."""
    from .teams import spec
    out = np.zeros((len(teams), len(drawn)))
    for j, (floor, foes) in enumerate(drawn.items()):
        wins = classifier.win_a([spec(team, foe) for team in teams for foe in foes], fixed=(1, stats(floor, hard)))
        out[:, j] = wins.reshape(len(teams), len(foes)).mean(1)
    return out


def _init():
    global _CATALOG
    from .catalog import load_catalog
    _CATALOG = load_catalog()


def _battle(job):
    """(win chance, expected turns) of the player, attacking first, against one floor's draw."""
    from .training.labels import evaluate
    team, foe, seed, per_card = job
    probs, _, turns = evaluate(_CATALOG, {key: [team[key], foe[key]] for key in team}, seed, fixed=(1, per_card), turns=True)
    finished = probs[0] + probs[1]
    return (probs[0] / finished if finished > 0 else 0.0), turns


def engine_pool(workers=None):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init)


def engine_battles(catalog, teams, drawn, pool, hard=True, seed=1):
    """([teams, floors, draws] win chance, same shape expected turns, nan where unsupported): every battle of each team
    against the same draws."""
    from . import tower
    per_card = {floor: [tower.engine_stats(catalog, foe["cards"], stats(floor, hard)) for foe in foes]
                for floor, foes in drawn.items()}
    jobs, owners = [], []
    for t, team in enumerate(teams):
        for j, (floor, foes) in enumerate(drawn.items()):
            for k, foe in enumerate(foes):
                jobs.append((team, foe, seed + 2 * (j * len(foes) + k), per_card[floor][k]))
                owners.append((t, j, k))
    shape = (len(teams), len(drawn), len(next(iter(drawn.values()))))
    wins, turns = np.zeros(shape), np.full(shape, np.nan)
    for (t, j, k), (win, length) in zip(owners, pool.map(_battle, jobs, chunksize=4)):
        wins[t, j, k], turns[t, j, k] = win, length
    return wins, turns


def means(wins, turns, keep=None):
    """Per floor (last axis averaged): the win chance and expected turns over the draws `keep` marks (all if None; a
    floor with none kept keeps all its draws)."""
    if keep is not None:
        keep = np.where(keep.any(-1, keepdims=True), keep, True)
        wins, turns = np.where(keep, wins, np.nan), np.where(keep, turns, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(wins, -1), np.nan_to_num(np.nanmean(turns, -1))


def engine_curves(catalog, teams, drawn, pool, hard=True, seed=1):
    """([teams, floors] win chance, [teams, floors] expected turns): the engine's means per grid floor against the
    same draws."""
    return means(*engine_battles(catalog, teams, drawn, pool, hard, seed))


def ban_search(catalog, team_wins, team_turns, drawn, value, count, fixed=()):
    """Greedy bans for one team from its battles against `drawn` (drawn without the `fixed` bans): a ban removes its
    card from the pool, and every remaining draw becomes more likely by the same factor, so the team's results under
    a ban list are its results over the draws holding none of the banned cards, with no new battles. Each step bans
    the card that raises value(wins, turns) most on the even draws, kept only if it also raises it on the odd ones
    (each ban drops the draws that hold it, so on one sample a ban can win by dropping slow or unlucky draws), until
    `count` bans or none helps. Returns (bans, value on all draws, {card: gain when banned alone, on all draws}), the
    last for the search's statistics."""
    cards = [[set(foe["cards"]) for foe in foes] for foes in drawn.values()]
    candidates = sorted({card for floor in cards for foe in floor for card in foe} - set(fixed))
    if not candidates:
        return [], value(*means(team_wins, team_turns)), {}
    holds = {card: np.array([[card in foe for foe in floor] for floor in cards]) for card in candidates}
    base = value(*means(team_wins, team_turns))
    alone = {card: value(*means(team_wins, team_turns, ~holds[card])) - base for card in candidates}
    half = np.zeros_like(holds[candidates[0]])
    half[:, ::2] = True
    score = lambda keep, part: value(*means(team_wins, team_turns, keep & (half if part == 0 else ~half)))
    banned, kept = [], np.ones_like(half)
    while len(banned) < count:
        fit = {card: score(kept & ~holds[card], 0) for card in candidates if card not in banned}
        if not fit:
            break
        card = max(fit, key=fit.get)
        if fit[card] <= score(kept, 0) or score(kept & ~holds[card], 1) <= score(kept, 1):
            break
        banned.append(card)
        kept &= ~holds[card]
    return banned, value(*means(team_wins, team_turns, kept)), alone


def battle_speed(floors, chrono=True, structure=0, skill=0):
    """His effective Depths battle speed at each floor."""
    floors = np.asarray(floors)
    return (BASE_SPEED + (CHRONO_SHARD if chrono else 0) + STRUCTURE_STEP * min(7, structure) + SKILL_TREE[min(4, skill)]
            + np.minimum(FLOOR_SPEED_CAP, (floors // 100) * FLOOR_SPEED_STEP))


_ACCELERATED = np.concatenate([[0.0], np.cumsum([1 / (10 if k >= 60 else 5 if k >= 40 else 3 if k >= 20 else 2 if k >= 10
                                                       else 1) for k in range(1, 4001)])])


def battle_seconds(floors, turns, **speed):
    """His estimated seconds per floor: between-floor wait, battle start and the attacks (accelerated in long fights;
    fractional turns as he interpolates)."""
    v = battle_speed(floors, **speed)
    attacks = np.interp(np.asarray(turns, dtype=float), np.arange(len(_ACCELERATED)), _ACCELERATED)
    return BETWEEN_FLOORS + START_SECONDS / v + ATTACK_SECONDS * attacks / v


def aura_packs(floors):
    """Aura packs a floor gives (his depthsRewardStageAttack / 500, flat from floor 5,000)."""
    f = np.minimum(np.asarray(floors), REWARD_CAP_FLOOR)
    return np.ceil(np.ceil(np.sqrt((3000 + f ** 2.7 * 40) / 2)) / 500)


def run_value(floors, wins, turns, cap, restart=0.0, **speed):
    """A run from floor 1 with each grid floor's win chance and expected turns held to the next grid floor, plus
    `restart` seconds to start the next run: {"expected_floors" cleared, "minutes" per run, "packs" per run,
    "packs_per_hour", "floors_per_hour"}."""
    every = np.arange(1, cap + 1)
    at = np.searchsorted(floors, every, side="right") - 1
    p, t = np.clip(np.asarray(wins)[at], 0, 1), np.asarray(turns)[at]
    alive = np.cumprod(p)  # survived through floor f
    reach = np.concatenate([[1.0], alive[:-1]])  # played floor f
    seconds = float((reach * battle_seconds(every, t, **speed)).sum()) + restart
    packs = float((reach * aura_packs(every)).sum())
    cleared = float(alive.sum())
    return {"expected_floors": round(cleared, 1), "minutes": round(seconds / 60, 1), "packs": round(packs, 1),
            "packs_per_hour": round(packs / seconds * 3600, 1), "floors_per_hour": round(cleared / seconds * 3600, 1)}


def add_arguments(parser):
    parser.add_argument("--normal", action="store_true", help="normal depths (default: hard depths, stats of floor x 10)")
    parser.add_argument("--bans", nargs="+", default=[], metavar="CARD",
                        help=f"the player's Depth bans (up to {MAX_BANS}), or a preset: {', '.join(BAN_PRESETS)}")
    parser.add_argument("--cap", type=int, default=FLOOR_RANGE[1], help="last floor of the curve")
    parser.add_argument("--points", type=int, help="grid floors sampled (24; 12 for the speed search)")
    parser.add_argument("--samples", type=int, help="enemy draws per grid floor (32; 8 for the speed search)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-chrono-shard", action="store_true", help="speed: without the Chrono Shard (+1 speed)")
    parser.add_argument("--structure", type=int, default=0, help="speed: battle-speed structure level (0-7)")
    parser.add_argument("--skill-tree", type=int, default=0, help="speed: battle-speed skill-tree level (0-4)")
    parser.add_argument("--restart-seconds", type=float, default=0.0,
                        help="speed: seconds from a run's death to the next run's first floor (not yet measured: 0)")


def expand_packs(catalog, cards, red=None, blue=None):
    """Teams from four card names, where a "pack:NAME[@Border][/Mutation]" slot stands for every card of that pack
    (user, 2026-10-07: a speedrun team's "1 dino" is any Prehistoric card, in whichever slot the lineup puts it), one
    team per card, the lineup as given; at most one of each single-copy card."""
    from .teams import SINGLE_COPY, parse_side
    from itertools import product
    options = []
    for text in cards:
        if not text.lower().startswith("pack:"):
            options.append([text])
            continue
        name, mark = text[5:], ""
        for sep in ("@", "/"):
            if sep in name:
                name, rest = name.split(sep, 1)
                mark = sep + rest
                break
        members = [c.name for c in catalog.cards if c.packs and any(p.casefold() == name.strip().casefold() for p in c.packs)]
        if not members:
            packs = sorted({p for c in catalog.cards for p in (c.packs or ())})
            raise SystemExit(f"No pack {name!r}; one of {', '.join(packs)}")
        options.append([f"{member}{mark}" for member in members])
    teams = []
    for names in product(*options):
        team = parse_side(catalog, list(names), red, blue)
        if not any(team["cards"].count(card) > 1 for card in SINGLE_COPY):
            teams.append(team)
    return teams


def evidence(path, objective, mode, min_count=5, top=15):
    """Statistics from the searches' own engine evidence (SEARCH_LOG; user, 2026-10-08): per card, card pair and
    support, the lift (mean score of the logged teams holding it minus the mean of all logged teams) with its count;
    per ban, the mean gain banning that card alone gave the teams it was weighed for; the best distinct decks. Teams
    from one search cluster around what it liked, so a lift says "teams with it scored higher", not "it causes that"."""
    from itertools import combinations
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []
    teams = [r for r in records if r["kind"] == "team" and r["objective"] == objective and r["mode"] == mode]
    ban_steps = [r for r in records if r["kind"] == "bans" and r["objective"] == objective and r["mode"] == mode]
    if not teams:
        raise SystemExit(f"No {mode} {objective} evidence in {path} yet: run depths search first")
    scores = np.array([r[objective] for r in teams])
    overall = float(scores.mean())
    print(f"{len(teams)} engine-scored teams, {len(ban_steps)} ban steps; mean {objective} {overall:.1f}")

    def table(title, keys_of):
        groups = {}
        for r, score in zip(teams, scores):
            for key in set(keys_of(r)):
                groups.setdefault(key, []).append(score)
        rows = sorted(((key, len(v), float(np.mean(v)) - overall) for key, v in groups.items() if len(v) >= min_count),
                      key=lambda row: -row[2])
        print(f"\n{title} (at least {min_count} teams; lift = mean with it - mean of all)")
        for key, n, lift in rows[:top]:
            print(f"  {key:<58} {n:>6} {lift:>+9.1f}")
        if len(rows) > top:
            print("  ...")
            for key, n, lift in rows[-3:]:
                print(f"  {key:<58} {n:>6} {lift:>+9.1f}")

    strip = lambda card: card.split("@")[0].split("/")[0]
    table("cards", lambda r: [strip(c) for c in r["team"]["cards"]])
    table("card pairs", lambda r: [" + ".join(pair) for pair in combinations(sorted(strip(c) for c in r["team"]["cards"]), 2)])
    table("supports", lambda r: [f"{color}: {r['team'][color]}" for color in ("red", "blue") if r["team"].get(color)])
    gains = {}
    for r in ban_steps:
        for card, gain in r["gain"].items():
            gains.setdefault(card, []).append(gain)
    rows = sorted(((card, len(v), float(np.mean(v))) for card, v in gains.items()), key=lambda row: -row[2])
    print(f"\nbans (mean gain in {objective} from banning the card alone, over the teams it was weighed for)")
    for card, n, gain in rows[:top]:
        print(f"  {card:<58} {n:>6} {gain:>+9.1f}")
    seen, decks = set(), []
    for r, score in sorted(zip(teams, scores), key=lambda item: -item[1]):
        key = json.dumps(r["team"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            decks.append((r, score))
    print(f"\nbest decks")
    for r, score in decks[:top]:
        supports = ", ".join(f"{r['team'][c]}" for c in ("red", "blue") if r["team"].get(c))
        print(f"  {score:>8.1f}  {' / '.join(r['team']['cards'])}  [{supports}]  bans: {', '.join(r['bans']) or 'none'}")


def parse_bans(catalog, names):
    from .deck import _match
    names = [card for name in names for card in BAN_PRESETS.get(name, [name])]
    if len(names) > MAX_BANS:
        raise SystemExit(f"At most {MAX_BANS} Depth bans")
    return [_match(name, [(c.id, c.name) for c in catalog.cards], "Card")[0] for name in names]


def main():
    from .catalog import load_catalog
    from .teams import describe, parse_side
    from .training.predict import Classifier
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="a team's depth curve (and with --simulate the engine's, speed and packs)")
    run.add_argument("--simulate", action="store_true", help="also the engine's curve, speed and aura packs")
    search = commands.add_parser("search", help="the team that goes deepest (model search) or earns aura packs "
                                 "fastest (engine search), engine-verified")
    search.add_argument("--objective", choices=("depth", "speed"), default="depth",
                        help="depth: expected floors cleared, by the model; speed: aura packs per hour, by the engine "
                        "(the model gives no battle length)")
    search.add_argument("--pool", choices=("own", "custom", "restricted", "all"), default="own")
    search.add_argument("--borders", nargs="+", default=["none"])
    search.add_argument("--mutations", nargs="+", default=["None"])
    search.add_argument("--support-tiers", nargs="+", default=["base"])
    search.add_argument("--no-limited", action="store_true", help="restricted pool without its Limited exceptions")
    search.add_argument("--evaluations", type=int, help="teams the model scores per team step (5,000)")
    search.add_argument("--engine-evaluations", type=int, default=300,
                        help="speed: teams the engine scores per team step, starting from the model's best distinct 32")
    search.add_argument("--top", type=int, default=4, help="best distinct teams reported")
    search.add_argument("--optimize-bans", type=int, default=0, metavar="N",
                        help="also choose N bans (with any --bans kept, at most 14): ban steps and team steps alternate")
    search.add_argument("--rounds", type=int, default=2, help="ban step + team step rounds with --optimize-bans")
    search.add_argument("--ban-samples", type=int, default=64,
                        help="enemy draws per grid floor that a ban step scores ban lists on (no new battles per list)")
    for command in (run, search):
        command.add_argument("--ally", nargs=4, required=command is run, metavar="CARD",
                             help="four cards, Name[@Border][/Mutation] (search: a starting team); in run, a "
                             "pack:NAME[@Border] slot tries every card of that pack and ranks the teams")
        command.add_argument("--ally-red")
        command.add_argument("--ally-blue")
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device")
        command.add_argument("--workers", type=int, help="engine processes")
        add_arguments(command)
    stats = commands.add_parser("stats", help="statistics from the searches' own engine evidence (cards, pairs, "
                                "supports, bans, decks)")
    stats.add_argument("--objective", choices=("packs_per_hour", "expected_floors"), default="packs_per_hour")
    stats.add_argument("--normal", action="store_true", help="normal depths (default: hard depths)")
    stats.add_argument("--min-count", type=int, default=5, help="teams a card, pair or support needs to be listed")
    stats.add_argument("--top", type=int, default=15)
    stats.add_argument("--log", default=str(SEARCH_LOG))
    args = parser.parse_args()
    if args.command == "stats":
        return evidence(args.log, args.objective, "normal" if args.normal else "hard", args.min_count, args.top)
    catalog = load_catalog()
    hard = not args.normal
    fast = args.command == "search" and args.objective == "speed"
    speed = dict(chrono=not args.no_chrono_shard, structure=args.structure, skill=args.skill_tree,
                 restart=args.restart_seconds)
    bans = parse_bans(catalog, args.bans)
    if args.command == "search" and len(bans) + args.optimize_bans > MAX_BANS:
        raise SystemExit(f"--bans and --optimize-bans together exceed the {MAX_BANS} ban slots")
    floors = grid(args.cap, args.points or (12 if fast else 24))
    drawn = enemies(catalog, floors, args.samples or (8 if fast else 32), bans, args.seed)
    classifier = Classifier(args.checkpoint, args.device)
    missing = pool(catalog)[1]
    if missing:
        print(f"his Depths pool has cards our catalog lacks (never drawn): {', '.join(missing)}", file=sys.stderr)
    variants = expand_packs(catalog, args.ally, args.ally_red, args.ally_blue) if args.ally else []
    if args.command == "search" and len(variants) > 1:
        raise SystemExit("search takes one starting team: name its cards (pack: slots are for run)")
    start = variants[0] if variants else None

    def report(team, model, engine=None):
        result = {"team": describe(catalog, team), "mode": "hard" if hard else "normal"}
        for name, wins in (("model", model), ("engine", engine and engine[0])):
            if wins is None:
                continue
            expected, curve, median = survival(floors, wins, args.cap)
            result[name] = {"expected_floors": round(expected, 1), "median_death_floor": median,
                            "floors": {f: [round(float(w), 3), round(c, 4)] for f, w, c in zip(floors, wins, curve)}}
        if engine is not None:
            result["engine"].update(run_value(floors, engine[0], engine[1], args.cap, **speed),
                                    turns={f: round(float(t), 1) for f, t in zip(floors, engine[1])})
        print(json.dumps(result), flush=True)

    with engine_pool(args.workers) as engine:
        curves = lambda teams: engine_curves(catalog, teams, drawn, engine, hard, args.seed)
        if args.command == "run":
            models = model_wins(classifier, variants, drawn, hard)
            engines = curves(variants) if args.simulate else None
            rows = []
            for i, team in enumerate(variants):
                report(team, models[i], (engines[0][i], engines[1][i]) if engines else None)
                row = {"team": " / ".join(describe(catalog, team)["cards"]),
                       "model_floors": survival(floors, models[i], args.cap)[0]}
                if engines:
                    row.update(run_value(floors, engines[0][i], engines[1][i], args.cap, **speed),
                               median=survival(floors, engines[0][i], args.cap)[2])
                rows.append(row)
            print("floors: grid floor -> [mean win chance there, chance to have survived through the floors it covers]; "
                  "turns: expected battle length per grid floor")
            if len(rows) > 1:  # a pack slot: every card of the pack, best first
                key = "packs_per_hour" if engines else "model_floors"
                print(f"\n{'team':<60} {'model floors':>12}" + (f" {'engine floors':>13} {'median death':>12} {'min/run':>8} "
                                                                 f"{'packs/h':>9} {'floors/h':>9}" if engines else ""))
                for row in sorted(rows, key=lambda r: -r[key]):
                    line = f"{row['team']:<60} {row['model_floors']:>12.1f}"
                    if engines:
                        line += (f" {row['expected_floors']:>13.1f} {row['median']:>12} {row['minutes']:>8.1f} "
                                 f"{row['packs_per_hour']:>9.1f} {row['floors_per_hour']:>9.1f}")
                    print(line)
            return
        from types import SimpleNamespace
        from .training.generate import _ids, _team, evolve, make_pool, masks
        from .training.hard import distinct
        borders, mutations, tiers = masks(args.borders, args.mutations, args.support_tiers)
        space = SimpleNamespace(classifier=classifier, pool=make_pool(catalog, args.pool, borders=borders,
                                                                      limited=not args.no_limited, mutations=mutations,
                                                                      tiers=tiers))
        starts = []
        if start is not None:
            for key, supports in (("red", space.pool.reds), ("blue", space.pool.blues)):  # yours, owned or none
                if (start[key], start[f"{key}_tier"]) not in supports:
                    supports.append((start[key], start[f"{key}_tier"]))
            try:
                _ids(space.pool, start)
                starts = [start]
            except (KeyError, ValueError):
                raise SystemExit("The starting team is not in the pool (its cards, borders, mutations or supports)")


        def team_search(drawn, starts, seed):
            """A team step against `drawn`, teams best first. The model builds them (user, 2026-10-08: no reference
            team, it builds its own): its search by expected floors. For speed, which the model cannot score (no
            battle length), its best distinct 32 then seed an engine search by aura packs per hour."""
            def by_model(members, done):
                teams = [_team(space, *m) for m in members]
                return [survival(floors, wins, args.cap)[0] for wins in model_wins(classifier, teams, drawn, hard)]

            def by_engine(members, done):
                teams = [_team(space, *m) for m in members]
                wins, turns = engine_curves(catalog, teams, drawn, engine, hard, seed)
                values = [run_value(floors, w, t, args.cap, **speed) for w, t in zip(wins, turns)]
                log_teams(teams, values, banned_now)
                return [v["packs_per_hour"] for v in values]

            seen = evolve(space, starts, by_model, evaluations=args.evaluations or 5_000, seed=seed, prior=0.0,
                          catalog=catalog, population=128, parents=32, children=4)
            teams = [_team(space, *m) for m in sorted(seen, key=lambda m: -seen[m])]
            if not fast:
                return teams
            seeds = [*starts, *(teams[i] for i in distinct(teams, range(len(teams)), 32))][:32]
            seen = evolve(space, seeds, by_engine, evaluations=args.engine_evaluations, seed=seed, prior=0.0,
                          catalog=catalog, population=32, parents=8, children=4)
            return [_team(space, *m) for m in sorted(seen, key=lambda m: -seen[m])]

        objective = "packs_per_hour" if fast else "expected_floors"
        context = {"mode": "hard" if hard else "normal", "objective": objective, "pool": args.pool, "cap": args.cap,
                   "speed": {k: v for k, v in speed.items()}}
        banned_now = list(bans)

        def log_teams(teams, values, in_force):
            with open(SEARCH_LOG, "a") as log:
                for team, v in zip(teams, values):
                    log.write(json.dumps({"kind": "team", **context, "bans": sorted(name(in_force)),
                                          "team": describe(catalog, team), **v}) + "\n")

        def log_bans(team, alone):
            with open(SEARCH_LOG, "a") as log:
                log.write(json.dumps({"kind": "bans", **context, "fixed": sorted(name(bans)),
                                      "team": describe(catalog, team),
                                      "gain": {catalog.card(c).name: round(g, 2) for c, g in alone.items()}}) + "\n")
        value = lambda wins, turns: run_value(floors, wins, turns, args.cap, **speed)[objective]
        name = lambda cards: [catalog.card(card).name for card in cards]
        chosen, teams = [], [start] if start is not None else []
        for r in range(args.rounds if args.optimize_bans else 1):
            if args.optimize_bans and teams:  # ban step: the best bans for the current team, from its own battles
                big = enemies(catalog, floors, args.ban_samples, bans, args.seed + 100 + r)
                w, t = engine_battles(catalog, teams[:1], big, engine, hard, args.seed)
                chosen, best, alone = ban_search(catalog, w[0], t[0], big, value, args.optimize_bans, bans)
                log_bans(teams[0], alone)
                print(json.dumps({"round": r + 1, "step": "bans", "team": describe(catalog, teams[0])["cards"],
                                  "bans": name(chosen), objective: round(best, 1),
                                  "before": round(value(*means(w[0], t[0])), 1)}), flush=True)
            banned_now[:] = [*bans, *chosen]
            drawn_r = enemies(catalog, floors, args.samples or (8 if fast else 32), banned_now, args.seed + r)
            teams = team_search(drawn_r, teams[:1], args.seed + r)
            print(json.dumps({"round": r + 1, "step": "team", "best": describe(catalog, teams[0])}), flush=True)
        if args.optimize_bans:  # the final team's own bans
            big = enemies(catalog, floors, args.ban_samples, bans, args.seed + 999)
            w, t = engine_battles(catalog, teams[:1], big, engine, hard, args.seed)
            chosen, _, alone = ban_search(catalog, w[0], t[0], big, value, args.optimize_bans, bans)
            log_bans(teams[0], alone)
        best = [teams[i] for i in distinct(teams, range(len(teams)), args.top)]
        if start is not None and start not in best:
            best.append(start)  # the starting team, for comparison
        # verified on fresh draws from the final pool (the bans chosen on one sample are not scored on it)
        final = enemies(catalog, floors, 32, [*bans, *chosen], args.seed + 12345)
        if args.optimize_bans:
            print(json.dumps({"final_bans": name([*bans, *chosen]), "chosen": name(chosen)}), flush=True)
        wins, turns = engine_curves(catalog, best, final, engine, hard, args.seed)
        models = model_wins(classifier, best, final, hard)
        log_teams(best, [run_value(floors, w, t, args.cap, **speed) for w, t in zip(wins, turns)], [*bans, *chosen])
        for team, m, w, t in zip(best, models, wins, turns):
            report(team, m, (w, t))
        if start is not None and args.optimize_bans:  # the baseline: the starting team with only the bans given
            base = enemies(catalog, floors, 32, bans, args.seed + 12345)
            w, t = engine_curves(catalog, [start], base, engine, hard, args.seed)
            print(json.dumps({"baseline": describe(catalog, start)["cards"], "bans": name(bans),
                              **run_value(floors, w[0], t[0], args.cap, **speed)}), flush=True)

if __name__ == "__main__":
    main()
