"""Win rate of an ally team against an enemy team from the trained classifier, in both turn orders.

    python -m card_engine.training.predict \\
        --ally "Vampire Lord@GaPl" Set "Good Boy@Pl/Storm" Archer --ally-red Stormcaller@Galaxy --ally-blue "Guardian Angel@Ruby" \\
        --enemy "Immortal Witch" Archer "Good Boy" Set --enemy-blue Fate@Crystal --simulate

Cards are "Name[@Border][/Mutation]" and supports "Name[@Tier]" (see card_engine.teams). The classifier gives the
ally's win chance when the ally attacks first and when the enemy does; --simulate adds DaddyDrago's engine for both
(exact when the battle is decided by few chance points, else estimated with playouts).
"""

import argparse
import json

import numpy as np
import torch

from ..catalog import load_catalog
from ..model.checkpoint import load_checkpoint
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

    def win_a(self, specs, batch=2048):
        """P(side A wins) per spec."""
        out = []
        with torch.no_grad():
            for start in range(0, len(specs), batch):
                part = specs[start:start + batch]
                rows = {key: torch.tensor([s[key] for s in part], device=self.device) for key in FIELDS}
                out.append(self.model(**self.inputs(rows, self.table)).softmax(-1)[:, 0].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def ally_win(self, pairs):
        """[(ally, enemy)] -> [N, 2]: the ally's win chance attacking first, and defending (enemy attacks first)."""
        first = self.win_a([spec(ally, enemy) for ally, enemy in pairs])
        second = 1 - self.win_a([spec(enemy, ally) for ally, enemy in pairs])
        return np.stack([first, second], 1)


def simulate(catalog, ally, enemy, seed=12345):
    """The engine's ally win chance attacking first and defending, and whether each is exact."""
    from .labels import evaluate
    first, first_exact = evaluate(catalog, spec(ally, enemy), seed)
    second, second_exact = evaluate(catalog, spec(enemy, ally), seed + 1)
    return (first[0], second[1]), (first_exact, second_exact)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for side in ("ally", "enemy"):
        parser.add_argument(f"--{side}", nargs=4, required=True, metavar="CARD", help="four cards, Name[@Border][/Mutation]")
        parser.add_argument(f"--{side}-red", help="red support, Name[@Tier]")
        parser.add_argument(f"--{side}-blue", help="blue support, Name[@Tier]")
    parser.add_argument("--simulate", action="store_true", help="also run DaddyDrago's engine")
    parser.add_argument("--checkpoint", default=str(RUN_DIR / "model.checkpoint"))
    parser.add_argument("--device")
    args = parser.parse_args()
    catalog = load_catalog()
    ally = parse_side(catalog, args.ally, args.ally_red, args.ally_blue)
    enemy = parse_side(catalog, args.enemy, args.enemy_red, args.enemy_blue)
    classifier = Classifier(args.checkpoint, args.device)
    model = classifier.ally_win([(ally, enemy)])[0]
    report = {"ally": describe(catalog, ally), "enemy": describe(catalog, enemy),
              "model": {"ally_attacks_first": round(float(model[0]), 3), "enemy_attacks_first": round(float(model[1]), 3)},
              "checkpoint_step": classifier.metadata.get("step")}
    if args.simulate:
        (first, second), (first_exact, second_exact) = simulate(catalog, ally, enemy)
        report["simulator"] = {"ally_attacks_first": round(first, 3), "enemy_attacks_first": round(second, 3),
                               "exact": [first_exact, second_exact]}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
