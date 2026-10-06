"""Generate teams that beat an enemy, by gradient ascent on the frozen classifier (the user's design).

Every candidate starts from noise in the slot space. Each slot has a card vector, a border vector and a mutation
vector, the three factors a card token is built from:
  - card: description + pack + classes + identity;
  - border: the border embedding;
  - mutation: the mutation embedding.
Each side also has a distribution over the pool's red and blue supports.

Adam ascends the classifier's log win probability in one role: --role attack (the ally attacks first and loses a
mutual wipe) or --role defend (the enemy attacks first). The two are separate situations and are never averaged
(user, 2026-10-04). The classifier's weights never change.
  - Annealing (user, 2026-10-05), like a diffusion sampler with the classifier's gradient as the score: the ascent
    runs in --noise-levels levels, each after the first starting with fresh noise that shrinks geometrically from
    --sigma-max, so slots can leave a poor peak early and only polish late.
  - Stats: a slot's stats depend on the discrete choice. So the slot gets the expected log stats over the pool's
    entries under p(entry) = softmax(-distance / temperature). The distance factorises over card, border and
    mutation, each measured in units of its table's typical gap between neighbouring options.
  - Commitment: a penalty, the distance to the slot's nearest entry, ramps up during the ascent and pulls each slot
    onto one entry.
  - Entropy check: while any slot still spreads over several entries (effective count above --max-blur), the
    ascent continues with a doubled penalty.

Decoding is an exact factorised nearest-k match over the pool. Each slot keeps its k nearest (card, border,
mutation) entries, and each colour keeps its two most likely supports. The classifier rescores every
combination exactly and keeps the best per candidate. Distinct teams are ranked by the model; DaddyDrago's engine
verifies the best --counters of them.

Pools:
  - own: data/my_deck.json, with copy counts.
  - custom: data/custom_pool.json, with copy counts: a collection to tailor suggestions to, edited on the fly
    with python -m card_engine.deck --custom (reset, copy-deck, add, remove).
  - restricted: the player base's cards and borders, no mutations.
  - all: every card, border and eligible mutation.
Masks narrow the restricted and all pools, ours and the enemies' (user, 2026-10-04). The pools keep everything; the
generator leaves rare options out unless asked for them. Cards and supports put in a deck (own, custom) override
the masks: a deck pool is used as it is.
  - --borders: none by default; list some (--borders none Pl Cr) or allow all.
  - --mutations: None by default; list some (--mutations None Storm) or allow all.
  - --support-tiers: base by default, for red and blue alike (supports have no index to price them); list some
    (--support-tiers base Platinum) or allow all.
  - --max-rarity: leaves out entries rarer than this, card x border rarity as the game shows it (1 in N: 2.5M,
    30T, 10qd).

Engine search (--engine-search N engine evaluations, off by default): a local refinement, not a replacement for the
ascent, which is the generator (user). The ascent only finds what the model already believes, so where the model
rates almost everything 0, against huge stats, it stacks stats and loses (2026-10-05, an older checkpoint: 32
Titan/Gorilla teams, all 0.0 at floor-105 stats). It cannot cover the team space; it walks from the model's teams. Cheese decks are narrow four-card
combinations, so an evolution scored by the engine itself searches the same pool, masks and copy counts: 64
teams (the model's counters, then random pool teams whose cards are stat-ignoring ones half the time,
labels.STAT_IGNORING), whose 16 best each get 4 variants per generation (a card, a support, or the lineup order
changed), as training.hard mines hard examples. Its --counters best join the model's, and every team is then
verified again with other dice, so a lucky estimate during the search does not rank a team.

Enemies: an explicit team (--enemy), or --enemies N generated broadly. A broad enemy is a team ascended against
a random pool opponent and decoded by sampling at a high temperature.

    python -m card_engine.training.generate --enemy "Immortal Witch" Archer "Good Boy" Set --pool restricted
    python -m card_engine.training.generate --enemies 4 --pool own --borders all --support-tiers all
    python -m card_engine.training.generate --enemy Odin Kira Set Archer --pool custom --borders none Pl Cr --mutations all
"""

import argparse
import dataclasses
import itertools
import json
import sys
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..catalog import load_catalog
from ..deck import BORDERS, CUSTOM_FILE, DECK_FILE, TIERS, _border, _mutation, _tier, load as load_deck, shell_words
from ..mutations import MUTATION_NAMES
from ..ownership import parse_rarity
from ..restricted import entries as restricted_entries, load as load_restricted
from .. import tower
from ..teams import ASTRAEUS, ASTRAEUS_ARTS, SINGLE_COPY, describe, parse_side, side
from .predict import Classifier
from .train import RUN_DIR, card_stats


@dataclass
class Pool:
    entries: np.ndarray  # [E, 4]: card, border, mutation, art (Astraeus once per art, user: each art is its own card)
    copies: np.ndarray | None  # [E]: how many of each the player owns (None: unlimited)
    reds: list  # [(support id, tier)]; (0, 0) = none
    blues: list


def make_pool(catalog, kind, borders=None, limited=True, deck_path=None, mutations=None, tiers=None, max_rarity=None):
    """The pool's entries and supports. borders (IDs), mutations (indices), tiers (1-5) and max_rarity mask the open
    pools (restricted, all); None allows all. A deck pool (own, custom) is used as it is: the deck overrides them."""
    every_support = lambda table: [(s.id, tier) for s in table for tier in range(1, 6)]
    copies = None
    if kind in ("own", "custom"):
        deck_path = deck_path or (DECK_FILE if kind == "own" else CUSTOM_FILE)
        if not Path(deck_path).exists():
            flag = "" if kind == "own" else " --custom"
            raise SystemExit(f"No {kind} pool at {deck_path}; add cards with python -m card_engine.deck{flag} add ...")
        deck = load_deck(deck_path)
        owned = {}
        for e in deck["cards"]:
            key = (e["card"], e["border"], MUTATION_NAMES.index(e["mutation"]))
            owned[key] = owned.get(key, 0) + e["count"]
        rows = sorted(owned)  # the deck has no arts, so an owned Astraeus is offered in every art
        copies = np.array([owned[r] for r in rows])
        reds = sorted({(e["support"], e["tier"]) for e in deck["supports"] if e["color"] == "red"}) or [(0, 0)]
        blues = sorted({(e["support"], e["tier"]) for e in deck["supports"] if e["color"] == "blue"}) or [(0, 0)]
    elif kind == "restricted":
        rows = [(card, border, 0) for card, border in restricted_entries(load_restricted(), catalog, limited=limited)]
        reds, blues = every_support(catalog.red_supports), every_support(catalog.blue_supports)
    else:
        rows = [(c.id, border, mutation) for c in catalog.cards for border in range(1, 17)
                for mutation in (range(len(MUTATION_NAMES)) if c.weather_id == 1 else (0,))]
        reds, blues = every_support(catalog.red_supports), every_support(catalog.blue_supports)
    rows = [(*row, art) for row in rows for art in (range(1, len(ASTRAEUS_ARTS) + 1) if row[0] == ASTRAEUS else (0,))]
    if copies is not None:
        copies = np.array([count for row, count in zip(sorted(owned), copies)
                           for _ in (ASTRAEUS_ARTS if row[0] == ASTRAEUS else (0,))])
    entries = np.array(rows, dtype=np.int64).reshape(-1, 4)
    if kind in ("restricted", "all"):
        keep = np.ones(len(entries), dtype=bool)
        if borders is not None:
            keep &= np.isin(entries[:, 1], borders)
        if mutations is not None:
            keep &= np.isin(entries[:, 2], mutations)
        if max_rarity is not None:
            keep &= np.array([catalog.card(int(card)).rarity * catalog.border(int(border)).rarity <= max_rarity
                              for card, border in entries[:, :2]], dtype=bool)
        entries = entries[keep]
        if tiers is not None:
            reds = [r for r in reds if r[1] in tiers] or [(0, 0)]
            blues = [b for b in blues if b[1] in tiers] or [(0, 0)]
    if not len(entries):
        raise SystemExit(f"The {kind} pool is empty" + (" under these masks; allow more with --borders, --mutations, "
                                                         "--max-rarity" if kind in ("restricted", "all") else ""))
    return Pool(entries, copies, reds, blues)


def masks(borders, mutations, tiers):
    """Command-line masks -> (border IDs, mutation indices, tiers); "all" -> None (no mask). Lists may be separated by
    spaces or commas ("none Pl Cr", "none, Pl, Cr")."""
    def pick(values, parse):
        values = [v.strip() for value in values for v in value.split(",") if v.strip()]
        return None if [v.casefold() for v in values] == ["all"] else sorted({parse(v) for v in values})
    return pick(borders, _border), pick(mutations, lambda m: MUTATION_NAMES.index(_mutation(m))), pick(tiers, _tier)


ROLES = ("attack", "defend")  # the ally's role; also the column of Classifier.ally_win and verify
ART_KEY = 289  # card_table row of Astraeus's art n is ART_KEY + n


def card_keys(cards, arts):
    return torch.where(arts > 0, ART_KEY + arts, cards)


def _sqdist(x, table):
    """Squared distances [..., U], centred on the table's mean so the expansion keeps its precision."""
    centre = table.mean(0)
    x, table = x - centre, table - centre
    return (x.square().sum(-1, keepdim=True) + table.square().sum(-1) - 2 * x @ table.T).clamp_min(0)


class SlotSpace:
    """The classifier's token factors over one pool, and the forward pass for relaxed ally slots."""

    def __init__(self, classifier, pool):
        self.classifier, self.pool = classifier, pool
        self.model = strategy = classifier.model.strategy
        device = self.device = classifier.device
        data = classifier.inputs.data
        width = strategy.config.width
        with torch.no_grad():
            parts = strategy.card_part(classifier.table, data.pack_ids, data.class_weights, data.identity_keys)
            astraeus = ASTRAEUS - 1
            arts = strategy.card_part(classifier.table[astraeus].expand(len(ASTRAEUS_ARTS), -1),
                                      data.pack_ids[astraeus].expand(len(ASTRAEUS_ARTS)),
                                      None if data.class_weights is None else data.class_weights[astraeus].expand(len(ASTRAEUS_ARTS), -1),
                                      data.art_identity_keys[1:])
            # by card key: the card id, or ART_KEY + art for each Astraeus art (its own identity, as in training)
            self.card_table = torch.cat([parts.new_zeros(1, width), parts, arts])
            self.border_table = strategy.border_embedding.weight  # by border id - 1
            self.mutation_table = strategy.mutation_vectors(torch.arange(len(MUTATION_NAMES), device=device))
            entries = torch.as_tensor(pool.entries, device=device)
            self.entries = entries
            keys = card_keys(entries[:, 0], entries[:, 3])
            self.factors = []  # (options [U, W], entry -> option index [E], mean [W], std [W], median nearest gap)
            for column, table in ((0, self.card_table), (1, None), (2, self.mutation_table)):
                values, inverse = torch.unique(keys if column == 0 else entries[:, column], return_inverse=True)
                options = self.border_table[values - 1] if column == 1 else table[values]
                std = options.std(0, unbiased=False)
                gaps = _sqdist(options, options).fill_diagonal_(float("inf")).amin(-1) if len(options) > 1 else std.new_ones(1)
                self.factors.append((options, inverse, options.mean(0), std, float(gaps.median().clamp_min(1e-12))))
            base, red, *_ = classifier.inputs.stat_tables
            card, border, mutation, _ = entries.unbind(1)
            self.log_base = base[card, border, mutation].log()  # [E, 2]
            self.reds = torch.as_tensor(pool.reds, device=device)
            self.blues = torch.as_tensor(pool.blues, device=device)
            self.log_red = red[card[:, None], mutation[:, None], self.reds[None, :, 0], self.reds[None, :, 1]].log()  # [E, R, 2]
            self.red_vectors = strategy.support_vectors(0, self.reds[:, 0], self.reds[:, 1])
            self.blue_vectors = strategy.support_vectors(1, self.blues[:, 0], self.blues[:, 1])

    def distances(self, vectors):
        """[N, 4, E]: each slot's factorised distance to every pool entry."""
        total = 0
        for x, (options, inverse, _, _, scale) in zip(vectors, self.factors):
            if len(options) > 1:  # a factor with one option is fixed at it
                total = total + (_sqdist(x, options) / scale)[..., inverse]
        return total

    def fixed(self, sides, stats=None):
        """Exact tokens of discrete sides: card tokens without stats [N, 4, W], log stats [N, 4, 2], red, blue [N, W].
        stats: (HP, ATK, HP multiplier applies), every side's fixed stats (None: from its cards)."""
        rows = {key: torch.tensor([[s[key], s[key]] for s in sides], device=self.device)
                for key in ("cards", "borders", "mutations", "arts", "red", "red_tier", "blue", "blue_tier")}
        if stats is not None:
            rows["fixed_side"] = torch.zeros(len(sides), dtype=torch.long, device=self.device)
            rows["fixed_stats"] = torch.tensor([stats[:2]] * len(sides), device=self.device)
            rows["fixed_hp_mult"] = torch.full((len(sides),), int(stats[2]), device=self.device)
        tokens = (self.card_table[card_keys(rows["cards"][:, 0], rows["arts"][:, 0])] + self.border_table[rows["borders"][:, 0] - 1]
                  + self.mutation_table[rows["mutations"][:, 0]])
        stats = card_stats(rows, self.classifier.inputs.stat_tables)[:, 0].log()
        red = self.model.support_vectors(0, rows["red"][:, 0], rows["red_tier"][:, 0])
        blue = self.model.support_vectors(1, rows["blue"][:, 0], rows["blue_tier"][:, 0])
        return tokens, stats, red, blue

    def logits(self, a, b, hidden=None):
        """Outcome logits with side A and side B each given as (card tokens, log stats, red, blue). hidden: the side
        the model is not shown (incomplete mode), whose tuple is then ignored: as in training, its cards, stats and
        supports become the hidden stand-ins and MODE is 1."""
        model = self.model
        cards = torch.stack([a[0], b[0]], 1)
        stats = torch.stack([a[1], b[1]], 1)
        visible = torch.ones(stats.shape[:3], dtype=torch.bool, device=self.device)
        if hidden is not None:
            visible[:, hidden] = False
        stats = torch.where(visible[..., None], stats.exp(), 1.0)
        cards, stat_tokens = model.with_stats(cards, model.stat_inputs(stats, visible))
        supports = [torch.stack([a[2], b[2]], 1), torch.stack([a[3], b[3]], 1)]
        mode = torch.zeros(cards.shape[0], dtype=torch.long, device=self.device)
        if hidden is not None:
            cards = torch.where(visible[..., None], cards, model.hidden_card)
            if stat_tokens is not None:
                stat_tokens = torch.where(visible[..., None], stat_tokens, model.hidden_stat)
            seen = visible[:, :, 0, None]
            supports = [torch.where(seen, part, model.hidden_support[index]) for index, part in enumerate(supports)]
            mode = mode + 1
        return model.outcome(model.assemble(cards, supports, mode, stat_tokens))


@dataclass
class Settings:
    steps: int = 300
    lr: float = 0.05
    temperature: float = 0.1  # distance temperature of p(entry)
    commitment: float = 1.0  # final weight of the commitment penalty
    max_blur: float = 1.5  # entropy check: effective entries per slot
    rechecks: int = 3  # extra ascent rounds (doubled penalty) while slots stay blurred
    nearest: int = 3  # decoding: k nearest entries per slot
    role: str = "attack"  # the ascended side attacks first ("attack") or defends ("defend")
    incomplete: bool = False  # against the field the model cannot see (incomplete mode), not a named enemy
    noise_levels: int = 12  # annealing: the ascent runs in this many levels, fresh noise before each after the first
    # noise added before the second level, in units of each factor's spread (z). At floor 105, borderless, restricted
    # pool (2026-10-06): sigma 0.5-1 never left the stat-stacking peak, 2-4 reached high model scores, and 8 found the
    # first team the engine lets win (0.108) where 4's best five all lost.
    sigma_max: float = 8.0
    sigma_min: float = 0.02  # noise before the last level; levels between are geometric


def ascend(space, opponents, settings, generator, enemy_stats=None):
    """Relaxed ally slots ascended against one discrete opponent each; returns (distances [N,4,E], red, blue probs).
    enemy_stats: the opponents' fixed stats (HP, ATK, HP multiplier applies), if the battle mode sets them."""
    n = len(opponents)
    enemy = space.fixed(opponents, enemy_stats)
    width = space.card_table.shape[1]
    z = [torch.randn(n, 4, width, generator=generator, device="cpu").to(space.device).requires_grad_() for _ in range(3)]
    red_logits = torch.zeros(n, len(space.reds), device=space.device, requires_grad=True)
    blue_logits = torch.zeros(n, len(space.blues), device=space.device, requires_grad=True)
    optimizer = torch.optim.Adam([*z, red_logits, blue_logits], lr=settings.lr)

    def state():
        vectors = [mean + std * zi for zi, (_, _, mean, std, _) in zip(z, space.factors)]
        distance = space.distances(vectors)
        p = (-distance / settings.temperature).softmax(-1)
        return vectors, distance, p, red_logits.softmax(-1), blue_logits.softmax(-1)

    def step(weight):
        vectors, distance, p, red, blue = state()
        log_red = torch.einsum("erk,nr->nek", space.log_red, red)
        stats = torch.einsum("nse,ek->nsk", p, space.log_base) + torch.einsum("nse,nek->nsk", p, log_red)
        ally = (vectors[0] + vectors[1] + vectors[2], stats, red @ space.red_vectors, blue @ space.blue_vectors)
        hidden = (1 if settings.role == "attack" else 0) if settings.incomplete else None
        if settings.role == "attack":
            objective = space.logits(ally, enemy, hidden).log_softmax(-1)[:, 0]
        else:
            objective = space.logits(enemy, ally, hidden).log_softmax(-1)[:, 1]
        entropy = lambda q: -(q * q.clamp_min(1e-12).log()).sum(-1)
        # Commitment: the distance to each slot's nearest entry (an expected distance would settle between entries).
        commit = distance.amin(-1).mean(-1) + entropy(red) + entropy(blue)
        loss = (-objective + weight * commit).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Annealing (user, 2026-10-05: "like a diffusion model with a time variable"): the classifier's gradient is the
    # score; each level after the first starts by adding fresh noise of size sigma(t) to the slots and support
    # logits, so a slot stuck on a poor peak can leave it, and sigma falls geometrically to sigma_min, so the last
    # levels only polish. The commitment weight ramps over the whole schedule. One level is a plain ascent.
    levels = max(1, settings.noise_levels)
    sigmas = np.geomspace(settings.sigma_max, settings.sigma_min, levels - 1) if levels > 1 else []
    per_level = max(1, settings.steps // levels)
    total, t = per_level * levels, 0
    for level in range(levels):
        if level:
            sigma = float(sigmas[level - 1])
            with torch.no_grad():
                for zi in z:
                    zi.add_(sigma * torch.randn(zi.shape, generator=generator).to(space.device))
                for logits in (red_logits, blue_logits):
                    logits.add_(sigma * torch.randn(logits.shape, generator=generator).to(space.device))
        for _ in range(per_level):
            t += 1
            step(settings.commitment * (t / total) ** 2)
    weight = settings.commitment
    for _ in range(settings.rechecks):
        with torch.no_grad():
            p = state()[2]
            blur = (-(p * p.clamp_min(1e-12).log()).sum(-1)).exp()  # effective entries per slot
        if blur.max() <= settings.max_blur:
            break
        weight *= 2
        for _ in range(settings.steps // 3):
            step(weight)
    with torch.no_grad():
        _, distance, p, red, blue = state()
        blur = (-(p * p.clamp_min(1e-12).log()).sum(-1)).exp().amax(-1)
    return distance, red, blue, blur


def _team(space, entry_ids, red, blue):
    entries = space.pool.entries
    return side([tuple(int(v) for v in entries[e]) for e in entry_ids], space.pool.reds[red], space.pool.blues[blue])


def _allowed(space, entry_ids):
    """Within the pool's copy counts, and at most one of each single-copy card (teams.SINGLE_COPY)."""
    cards = [int(space.pool.entries[e][0]) for e in entry_ids]
    if any(cards.count(card) > 1 for card in SINGLE_COPY):
        return False
    if space.pool.copies is None:
        return True
    ids, counts = np.unique(entry_ids, return_counts=True)
    return bool((counts <= space.pool.copies[ids]).all())


def decode_nearest(space, distance, red, blue, nearest):
    """Per candidate: every combination of each slot's k nearest entries and the two most likely supports."""
    near = distance.topk(min(nearest, distance.shape[-1]), largest=False).indices.cpu().numpy()  # [N, 4, k]
    reds = red.topk(min(2, red.shape[-1])).indices.cpu().numpy()
    blues = blue.topk(min(2, blue.shape[-1])).indices.cpu().numpy()
    out = []
    for i in range(len(near)):
        teams = [_team(space, combo, r, b) for combo in itertools.product(*near[i])
                 if _allowed(space, combo) for r in reds[i] for b in blues[i]]
        out.append(teams)
    return out


def decode_sample(space, distance, red, blue, temperature, generator):
    """One team per candidate, sampled from p(entry) and the supports at a high temperature (broad enemies)."""
    out = []
    for i in range(distance.shape[0]):
        for _ in range(100):
            probs = (-distance[i] / temperature).softmax(-1).cpu()
            combo = torch.multinomial(probs, 1, generator=generator)[:, 0].numpy()
            if _allowed(space, combo):
                break
        r = int(torch.multinomial(red[i].cpu() ** (1 / temperature), 1, generator=generator))
        b = int(torch.multinomial(blue[i].cpu() ** (1 / temperature), 1, generator=generator))
        out.append(_team(space, combo, r, b))
    return out


def random_team(pool, rng):
    for _ in range(100):
        entry_ids = rng.choice(len(pool.entries), 4, replace=pool.copies is None or len(pool.entries) < 4)
        if all(list(pool.entries[entry_ids, 0]).count(card) <= 1 for card in SINGLE_COPY):
            break
    return side([tuple(int(v) for v in pool.entries[e]) for e in entry_ids],
                pool.reds[rng.integers(len(pool.reds))], pool.blues[rng.integers(len(pool.blues))])


def _key(team):
    return tuple(tuple(v) if isinstance(v, list) else v for v in team.values())


def counters(space, enemy, *, count=32, restarts=64, settings=Settings(), seed=1, enemy_stats=None):
    """The `count` best distinct ally teams against `enemy` in settings.role, by the classifier: [(team, win, blur)].
    With settings.incomplete the teams are ascended against the unseen field instead (enemy: labels.HIDDEN_TEAM)."""
    generator = torch.Generator().manual_seed(seed)
    distance, red, blue, blur = ascend(space, [enemy] * restarts, settings, generator, enemy_stats)
    best = {}
    for teams, slot_blur in zip(decode_nearest(space, distance, red, blue, settings.nearest), blur.cpu().numpy()):
        if not teams:
            continue
        if settings.incomplete:
            score = space.classifier.field_win(teams, ROLES.index(settings.role))
        else:
            score = space.classifier.ally_win([(team, enemy) for team in teams], enemy_stats)[:, ROLES.index(settings.role)]
        top = int(score.argmax())
        key = _key(teams[top])
        if key not in best or score[top] > best[key][1]:
            best[key] = (teams[top], float(score[top]), float(slot_blur.max()))
    return sorted(best.values(), key=lambda r: -r[1])[:count]


def broad_enemies(space, n, *, settings=Settings(), temperature=1.0, seed=1):
    """`n` diverse strong teams: each ascended against a random pool opponent, decoded by sampling. settings.role is
    the ally's, so the enemies are ascended in the other role."""
    rng = np.random.default_rng(seed)
    opponents = [random_team(space.pool, rng) for _ in range(n)]
    generator = torch.Generator().manual_seed(seed)
    enemy_role = ROLES[1 - ROLES.index(settings.role)]
    distance, red, blue, _ = ascend(space, opponents, dataclasses.replace(settings, role=enemy_role), generator)
    return decode_sample(space, distance, red, blue, temperature, generator)


def _role_win(job):
    """The ally's win chance in one role (0 attack, 1 defend) against `enemy`, its stats fixed if per_card is given."""
    from . import counter
    player, enemy, seed, per_card, role = job
    enemy_side = 1 - role
    battle = counter._spec(player, enemy) if role == 0 else counter._spec(enemy, player)
    probs, _ = counter.evaluate(counter._CATALOG, battle, seed,
                                fixed=None if per_card is None else (enemy_side, per_card))
    return float(probs[role])


def _ids(pool, team):
    """A team as pool indices: (entry per slot, red, blue)."""
    index = {tuple(int(v) for v in entry): i for i, entry in enumerate(pool.entries)}
    entries = tuple(index[c] for c in zip(team["cards"], team["borders"], team["mutations"], team["arts"]))
    return entries, pool.reds.index((team["red"], team["red_tier"])), pool.blues.index((team["blue"], team["blue_tier"]))


def engine_search(space, enemy, starts, *, evaluations=4096, role="attack", enemy_stats=None, workers=None, seed=1,
                  catalog=None, population=64, parents=16, children=4):
    """An evolution scored by the engine, inside the pool (see the module notes): [(team, engine win)], best first."""
    from concurrent.futures import ProcessPoolExecutor
    from .counter import _init
    from .labels import stat_ignoring_cards
    catalog = catalog or load_catalog()
    pool, rng = space.pool, np.random.default_rng(seed)
    ignoring = set(stat_ignoring_cards(catalog))
    ignoring_entries = [i for i, entry in enumerate(pool.entries) if int(entry[0]) in ignoring]
    per_card = None if enemy_stats is None else tower.engine_stats(catalog, enemy["cards"], enemy_stats)
    column = ROLES.index(role)

    def entry():
        if ignoring_entries and rng.random() < 0.5:
            return int(rng.choice(ignoring_entries))
        return int(rng.integers(len(pool.entries)))

    def fresh():
        for _ in range(100):
            ids = tuple(entry() for _ in range(4))
            if _allowed(space, ids):
                break
        return ids, int(rng.integers(len(pool.reds))), int(rng.integers(len(pool.blues)))

    def mutate(member):
        ids, red, blue = member
        move = rng.random()
        if move < 0.6:
            ids = list(ids)
            ids[int(rng.integers(4))] = entry()
            ids = tuple(ids)
        elif move < 0.8:
            if rng.random() < 0.5:
                red = int(rng.integers(len(pool.reds)))
            else:
                blue = int(rng.integers(len(pool.blues)))
        else:
            a, b = rng.choice(4, 2, replace=False)
            ids = list(ids)
            ids[a], ids[b] = ids[b], ids[a]
            ids = tuple(ids)
        return (ids, red, blue) if _allowed(space, ids) else member

    seen = {}
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 1), mp_context=mp.get_context("spawn"),
                             initializer=_init) as executor:
        def score(members):
            new = list(dict.fromkeys(m for m in members if m not in seen))
            jobs = [(_team(space, *m), enemy, seed + 2 * (len(seen) + i), per_card, column) for i, m in enumerate(new)]
            for member, win in zip(new, executor.map(_role_win, jobs, chunksize=8)):
                seen[member] = win

        alive = [_ids(pool, team) for team in starts][:population]
        alive += [fresh() for _ in range(population - len(alive))]
        score(alive)
        while len(seen) < evaluations:
            best = sorted(set(alive), key=lambda m: -seen[m])[:parents]
            before = len(seen)
            score([mutate(m) for m in best for _ in range(children)])
            alive = sorted(seen, key=lambda m: -seen[m])[:population]  # the best so far, parents included
            if len(seen) == before:  # nothing new to try
                break
    return [(_team(space, *m), win) for m, win in sorted(seen.items(), key=lambda item: -item[1])]


def verify(teams, enemy, workers, seed=1, enemy_stats=None, catalog=None):
    """The engine's ally win chance attacking first and defending, per team."""
    from .counter import _evaluate, _init
    per_card = None if enemy_stats is None else tower.engine_stats(catalog or load_catalog(), enemy["cards"], enemy_stats)
    jobs = [(team, enemy, seed + 2 * i, per_card, 1, None) for i, team in enumerate(teams)]
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        return pool.map(_evaluate, jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enemy", nargs=4, metavar="CARD", help="the enemy's four cards, Name[@Border][/Mutation]")
    parser.add_argument("--enemy-red")
    parser.add_argument("--enemy-blue")
    tower.add_arguments(parser)
    parser.add_argument("--enemies", type=int, help="generate this many broad enemies instead of --enemy")
    pools = ("own", "custom", "restricted", "all")
    parser.add_argument("--enemy-pool", choices=pools, help="pool for --enemies (default: --pool)")
    parser.add_argument("--pool", choices=pools, default="restricted")
    parser.add_argument("--borders", nargs="+", default=["none"], help=f"allowed borders (default none), from "
                        f"{' '.join(BORDERS)}, or all")
    parser.add_argument("--mutations", nargs="+", default=["None"], help=f"allowed mutations (default None), from "
                        f"{', '.join(MUTATION_NAMES)}, or all")
    parser.add_argument("--support-tiers", nargs="+", default=["base"], help=f"allowed support tiers, red and blue "
                        f"(default base), from {' '.join(TIERS)}, or all")
    parser.add_argument("--max-rarity", help="leave out entries rarer than this card x border rarity (1 in N, e.g. 30T, 10qd)")
    parser.add_argument("--no-limited", action="store_true", help="restricted pool without its Limited exceptions")
    parser.add_argument("--counters", type=int, default=32)
    parser.add_argument("--restarts", type=int, default=64, help="candidates ascended per enemy")
    parser.add_argument("--steps", type=int, default=Settings.steps)
    parser.add_argument("--temperature", type=float, default=Settings.temperature)
    parser.add_argument("--nearest", type=int, default=Settings.nearest)
    parser.add_argument("--noise-levels", type=int, default=Settings.noise_levels,
                        help="annealing levels: fresh noise, shrinking from --sigma-max, before each after the first (1: none)")
    parser.add_argument("--sigma-max", type=float, default=Settings.sigma_max)
    parser.add_argument("--role", choices=ROLES, default="attack",
                        help="attack: your team attacks first and loses a mutual wipe; defend: the enemy attacks first")
    parser.add_argument("--no-verify", action="store_true", help="skip the engine check")
    parser.add_argument("--engine-search", type=int, default=0, metavar="N",
                        help="also run an engine-guided search of N engine evaluations from the model's teams (see above)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--checkpoint", default=str(RUN_DIR / "model.checkpoint"))
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", help="also write the report here (JSON)")
    args = parser.parse_args(shell_words(sys.argv[1:]))
    catalog = load_catalog()
    enemy, enemy_stats = tower.enemy(catalog, args, parser)
    if (enemy is None) == (args.enemies is None):
        parser.error("give either --enemy (or --tower with a fixed floor) or --enemies")
    if enemy_stats is not None and enemy is None:
        parser.error("--enemy-stats and --tower go with one enemy, not --enemies")
    mask = dict(zip(("borders", "mutations", "tiers"), masks(args.borders, args.mutations, args.support_tiers)),
                max_rarity=parse_rarity(args.max_rarity) if args.max_rarity else None)
    classifier = Classifier(args.checkpoint, args.device)
    settings = Settings(steps=args.steps, temperature=args.temperature, nearest=args.nearest, role=args.role,
                        noise_levels=args.noise_levels, sigma_max=args.sigma_max)
    space = SlotSpace(classifier, make_pool(catalog, args.pool, limited=not args.no_limited, **mask))
    if enemy is not None:
        enemies = [enemy]
    else:
        enemy_pool = args.enemy_pool or args.pool
        enemy_space = space if enemy_pool == args.pool else SlotSpace(
            classifier, make_pool(catalog, enemy_pool, limited=not args.no_limited, **mask))
        enemies = broad_enemies(enemy_space, args.enemies, settings=settings, seed=args.seed)
    report = {"checkpoint_step": classifier.metadata.get("step"), "role": args.role, "pool": args.pool,
              "masks": {"borders": args.borders, "mutations": args.mutations, "support_tiers": args.support_tiers,
                        "max_rarity": args.max_rarity},
              "matchups": []}
    for index, enemy in enumerate(enemies):
        found = [(*entry, "model") for entry in counters(space, enemy, count=args.counters, restarts=args.restarts,
                                                         settings=settings, seed=args.seed + index, enemy_stats=enemy_stats)]
        if args.engine_search:
            searched = engine_search(space, enemy, [team for team, *_ in found], evaluations=args.engine_search,
                                     role=args.role, enemy_stats=enemy_stats, workers=args.workers,
                                     seed=args.seed + 7919 * (index + 1), catalog=catalog)
            known = {_key(team) for team, *_ in found}
            extra = [team for team, _ in searched if _key(team) not in known][:args.counters]
            if extra:
                wins = space.classifier.ally_win([(team, enemy) for team in extra], enemy_stats)[:, ROLES.index(args.role)]
                found += [(team, float(win), float("nan"), "engine search") for team, win in zip(extra, wins)]
        checked = None if args.no_verify else verify([team for team, *_ in found], enemy, args.workers, args.seed,
                                                     enemy_stats, catalog)
        rows = []
        column = ROLES.index(args.role)
        for i, (team, win, blur, source) in enumerate(found):
            row = {**describe(catalog, team), "model": round(win, 3), "source": source}
            if blur == blur:  # the ascent's slot blur (none for engine-search teams)
                row["slot_blur"] = round(blur, 2)
            if checked:
                row["simulator"] = round(checked[i][column], 3)
            rows.append(row)
        if checked:
            rows.sort(key=lambda r: (-r["simulator"], -r["model"]))  # engine ties (e.g. all 0.0): the model's order
        report["matchups"].append({"enemy": describe(catalog, enemy, enemy_stats), "counters": rows[:args.counters]})
        print(json.dumps(report["matchups"][-1], ensure_ascii=False), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
