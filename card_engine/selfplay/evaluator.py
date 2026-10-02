"""Batch diagnostic cross-play without promoting provisional rules to labels."""

from dataclasses import asdict, dataclass, replace
import math

from ..simulator.catalog_rules import compile_battle
from ..simulator.reference import Options, simulate
from ..mutations import effective_mutation


@dataclass(frozen=True)
class TeamSpec:
    """Ordered team supplied by the caller; ownership is not inferred here."""

    cards: tuple[int, ...]
    borders: tuple[int, ...]
    red_support: int = 0
    blue_support: int = 0
    mutations: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.mutations is None:
            object.__setattr__(self, "mutations", ("None",) * len(self.cards))
        if (not isinstance(self.cards, tuple) or not isinstance(self.borders, tuple)
                or not 1 <= len(self.cards) <= 4 or len(self.borders) != len(self.cards)):
            raise ValueError("Supply 1..4 ordered card IDs and one border ID per card as tuples")
        for values, maximum in ((self.cards, 289), (self.borders, 16),
                                ((self.red_support,), 28), ((self.blue_support,), 15)):
            minimum = 0 if maximum in (28, 15) else 1
            if any(type(value) is not int or not minimum <= value <= maximum for value in values):
                raise ValueError("Invalid card, border, or support ID")
        if not isinstance(self.mutations, tuple) or len(self.mutations) != len(self.cards):
            raise ValueError("Provide one mutation per card")
        for name in self.mutations:
            effective_mutation("None", name)


@dataclass(frozen=True)
class MatchEvaluation:
    index_a: int
    index_b: int
    p_a: float
    p_b: float
    tie: float
    unresolved: float
    expanded_states: int
    merged_states: int
    rules_status: str
    training_labels_allowed: bool
    simulation_mode: str = "branch"

    def __post_init__(self):
        if any(type(i) is not int or i < 0 for i in (self.index_a, self.index_b, self.expanded_states, self.merged_states)):
            raise ValueError("Match indices and counters must be nonnegative integers")
        if (not isinstance(self.rules_status, str) or not self.rules_status
                or type(self.training_labels_allowed) is not bool or self.simulation_mode not in ("sample", "branch")):
            raise ValueError("Invalid simulation provenance")
        masses = (self.p_a, self.p_b, self.tie, self.unresolved)
        if (any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 + 1e-9 for p in masses)
                or not math.isclose(sum(masses), 1, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Simulation probability must be finite, nonnegative, and account for all mass")

    def win_bounds(self, side):
        if self.simulation_mode != "branch":
            raise ValueError("One sample trajectory does not establish win-probability bounds")
        if type(side) is not int or side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        lower = min(1.0, self.p_a if side == 0 else self.p_b)
        return lower, min(1.0, lower + self.unresolved)


@dataclass(frozen=True)
class CrossPlay:
    teams_a: tuple[TeamSpec, ...]
    teams_b: tuple[TeamSpec, ...]
    matches: tuple[MatchEvaluation, ...]
    backend: str
    options: Options
    first_side: int

    def __post_init__(self):
        if not self.teams_a or not self.teams_b or any(not isinstance(t, TeamSpec) for t in self.teams_a + self.teams_b):
            raise ValueError("Cross-play requires nonempty TeamSpec sequences")
        if not isinstance(self.options, Options) or self.backend not in ("python", "c"):
            raise ValueError("Invalid cross-play provenance")
        self.options.validate()
        if type(self.first_side) is not int or self.first_side not in (0, 1):
            raise ValueError("first_side must be 0 or 1")
        expected = [(i, j) for i in range(len(self.teams_a)) for j in range(len(self.teams_b))]
        if (any(not isinstance(m, MatchEvaluation) or m.simulation_mode != self.options.mode for m in self.matches)
                or [(m.index_a, m.index_b) for m in self.matches] != expected):
            raise ValueError("Cross-play must contain each matchup once in A-major order with matching mode")

    def mean_win_bounds(self, side):
        """Equal-weight opposing candidates, retaining all unresolved mass."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        count = len(self.teams_a if side == 0 else self.teams_b)
        opposing = len(self.teams_b if side == 0 else self.teams_a)
        bounds = [[0.0, 0.0] for _ in range(count)]
        for match in self.matches:
            index = match.index_a if side == 0 else match.index_b
            for endpoint, value in enumerate(match.win_bounds(side)):
                bounds[index][endpoint] += value / opposing
        return tuple(map(tuple, bounds))

    def training_rows(self):
        """Future verified, fully resolved branch results only; never renormalize.

        Current kernels always return training_labels_allowed=False, so this
        method deliberately rejects every current production simulation.
        """
        if self.options.mode != "branch":
            raise ValueError("Sample trajectories are not exact probability labels")
        if any(match.training_labels_allowed is not True for match in self.matches):
            raise ValueError("Experimental simulator rules cannot produce training labels")
        if any(match.unresolved != 0 for match in self.matches):
            raise ValueError("Unresolved probability cannot be dropped or renormalized into labels")
        return tuple({"team_a": asdict(self.teams_a[match.index_a]),
                      "team_b": asdict(self.teams_b[match.index_b]),
                      "probabilities": (match.p_a, match.p_b, match.tie),
                      "first_side": self.first_side,
                      "rules_status": match.rules_status} for match in self.matches)


def evaluate_crossplay(catalog, teams_a, teams_b, *, options=None, backend="c",
                       first_side=0, library_path=None, max_matches=4096):
    """Validate all matchups, then evaluate the Cartesian product in stable order.

    C executes all matches in one call. Python uses the same per-match seed
    schedule. Seeds are assigned A-major, B-minor, independent of battle length.
    Inventory legality belongs to candidate generation/caller; no screenshot
    inference is interpreted as ownership authorization here.
    """
    if backend not in ("c", "python"):
        raise ValueError("backend must be c or python")
    if type(first_side) is not int or first_side not in (0, 1):
        raise ValueError("first_side must be 0 or 1")
    if type(max_matches) is not int or max_matches < 1:
        raise ValueError("max_matches must be positive")
    teams_a, teams_b = tuple(teams_a), tuple(teams_b)
    if not teams_a or not teams_b or len(teams_a) * len(teams_b) > max_matches:
        raise ValueError("Nonempty cross-play must fit max_matches")
    if any(not isinstance(team, TeamSpec) for team in teams_a + teams_b):
        raise ValueError("Every candidate must be a TeamSpec")
    options = Options() if options is None else options
    if not isinstance(options, Options):
        raise ValueError("options must be an Options instance")
    options.validate()
    pairs, battles = [], []
    for i, team_a in enumerate(teams_a):
        for j, team_b in enumerate(teams_b):
            pairs.append((i, j))
            battles.append(compile_battle(catalog, (team_a.cards, team_b.cards),
                           borders=(team_a.borders, team_b.borders),
                           mutations=(team_a.mutations, team_b.mutations),
                           red_supports=(team_a.red_support, team_b.red_support),
                           blue_supports=(team_a.blue_support, team_b.blue_support),
                           first_side=first_side))
    if backend == "c":
        from ..simulator.native import simulate_batch
        results = simulate_batch(battles, options, library_path=library_path)
    else:
        results = [asdict(simulate(battle, replace(options, seed=(options.seed + index) % 2**64)))
                   for index, battle in enumerate(battles)]
    matches = tuple(MatchEvaluation(i, j, simulation_mode=options.mode, **{key: result[key] for key in (
                    "p_a", "p_b", "tie", "unresolved", "expanded_states", "merged_states",
                    "rules_status", "training_labels_allowed")})
                    for (i, j), result in zip(pairs, results, strict=True))
    return CrossPlay(teams_a, teams_b, matches, backend, options, first_side)
