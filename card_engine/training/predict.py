"""Win rate of an ally team against an enemy team from the trained classifier, in both turn orders.

    python -m card_engine.training.predict \\
        --ally "Vampire Lord@GaPl" Set "Good Boy@Pl/Storm" Archer --ally-red Stormcaller@Galaxy --ally-blue "Guardian Angel@Ruby" \\
        --enemy "Immortal Witch" Archer "Good Boy" Set --enemy-blue Fate@Crystal --simulate

Cards are "Name[@Border][/Mutation]" and supports "Name[@Tier]" (see card_engine.teams). The classifier gives the
ally's win chance when the ally attacks first and when the enemy does; --simulate adds DaddyDrago's engine for both
(exact when the battle is decided by few chance points, else estimated with playouts).

Battle modes that give every enemy card the same stats: --enemy-stats HP ATK (e.g. 2.5M 400k) sets each enemy
card's starting stats and drops the enemy's borders (they only set stats). --tower FLOOR DIFFICULTY uses a tower
floor's stats (card_engine.tower), and its enemy team on fixed floors:
    python -m card_engine.training.predict --tower 105 Impossible --ally "Fate Seamstress" "Judgement Day" Parallax "Robin Hood" --ally-blue Fate --simulate
Stats enter the classifier as explicit inputs, and labels.py --fixed shards (tower battles) train it on them.
"""

import argparse
import json
import sys

import numpy as np
import torch

from ..catalog import load_catalog
from ..model.checkpoint import load_checkpoint
from ..deck import shell_words
from .. import tower
from ..teams import FIELDS, describe, parse_side, spec
from .train import RUN_DIR, Inputs, card_table


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Classifier:
    """The frozen trained model, scoring label specs."""

    def __init__(self, checkpoint=RUN_DIR / "model.checkpoint", device=None):
        self.device = device or default_device()
        self.model, self.metadata = load_checkpoint(checkpoint, map_location=self.device)
        self.model.eval().requires_grad_(False)
        self.inputs = Inputs(self.device)
        with torch.no_grad():
            self.table = card_table(self.model, self.inputs.data.description_tokens)

    def win_a(self, specs, batch=2048, fixed=None, hidden=None):
        """P(side A wins) per spec. fixed: (side, (HP, ATK, HP multiplier applies)) sets that side's starting stats.
        hidden: the side the model is not shown (incomplete mode: the field of strong teams, training.incomplete)."""
        out = []
        with torch.no_grad():
            for start in range(0, len(specs), batch):
                part = specs[start:start + batch]
                rows = {key: torch.tensor([s[key] for s in part], device=self.device) for key in FIELDS}
                if fixed is not None:  # that side's cards start at the fixed stats, as in fixed-stat labels
                    rows["fixed_side"] = torch.full((len(part),), fixed[0], device=self.device)
                    rows["fixed_stats"] = torch.tensor([fixed[1][:2]] * len(part), device=self.device)
                    rows["fixed_hp_mult"] = torch.full((len(part),), int(fixed[1][2]), device=self.device)
                if hidden is not None:
                    rows["hidden_side"] = torch.full((len(part),), hidden, device=self.device)
                out.append(self.model(**self.inputs(rows, self.table)).softmax(-1)[:, 0].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def ally_win(self, pairs, enemy_stats=None):
        """[(ally, enemy)] -> [N, 2]: the ally's win chance attacking first, and defending (enemy attacks first).
        enemy_stats: (HP, ATK, HP multiplier applies), the enemy's fixed stats (None: from its cards)."""
        first = self.win_a([spec(ally, enemy) for ally, enemy in pairs], fixed=enemy_stats and (1, enemy_stats))
        second = 1 - self.win_a([spec(enemy, ally) for ally, enemy in pairs], fixed=enemy_stats and (0, enemy_stats))
        return np.stack([first, second], 1)


    def field_win(self, teams, role):
        """The incomplete-mode win chance of each team in a role (0 attack, 1 defend) against the field it cannot see."""
        from .labels import HIDDEN_TEAM
        if role == 0:
            return self.win_a([spec(team, HIDDEN_TEAM) for team in teams], hidden=1)
        return 1 - self.win_a([spec(HIDDEN_TEAM, team) for team in teams], hidden=0)


def simulate(catalog, ally, enemy, seed=12345, enemy_stats=None):
    """The engine's ally win chance attacking first and defending, and whether each is exact."""
    from .labels import evaluate
    per_card = None if enemy_stats is None else tower.engine_stats(catalog, enemy["cards"], enemy_stats)
    fixed = lambda side: None if per_card is None else dict(fixed=(side, per_card))
    first, first_exact = evaluate(catalog, spec(ally, enemy), seed, **(fixed(1) or {}))
    second, second_exact = evaluate(catalog, spec(enemy, ally), seed + 1, **(fixed(0) or {}))
    return (first[0], second[1]), (first_exact, second_exact)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for side in ("ally", "enemy"):
        parser.add_argument(f"--{side}", nargs=4, required=side == "ally", metavar="CARD",
                            help="four cards, Name[@Border][/Mutation]")
        parser.add_argument(f"--{side}-red", help="red support, Name[@Tier]")
        parser.add_argument(f"--{side}-blue", help="blue support, Name[@Tier]")
    tower.add_arguments(parser)
    parser.add_argument("--simulate", action="store_true", help="also run DaddyDrago's engine")
    parser.add_argument("--checkpoint", default=str(RUN_DIR / "model.checkpoint"))
    parser.add_argument("--device")
    args = parser.parse_args(shell_words(sys.argv[1:]))
    catalog = load_catalog()
    ally = parse_side(catalog, args.ally, args.ally_red, args.ally_blue)
    enemy, enemy_stats = tower.enemy(catalog, args, parser)
    if enemy is None:
        parser.error("give --enemy, or --tower with a fixed floor")
    classifier = Classifier(args.checkpoint, args.device)
    model = classifier.ally_win([(ally, enemy)], enemy_stats)[0]
    report = {"ally": describe(catalog, ally), "enemy": describe(catalog, enemy, enemy_stats),
              "model": {"ally_attacks_first": round(float(model[0]), 3), "enemy_attacks_first": round(float(model[1]), 3)},
              "checkpoint_step": classifier.metadata.get("step")}
    if args.simulate:
        (first, second), (first_exact, second_exact) = simulate(catalog, ally, enemy, enemy_stats=enemy_stats)
        report["simulator"] = {"ally_attacks_first": round(first, 3), "enemy_attacks_first": round(second, 3),
                               "exact": [first_exact, second_exact]}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
