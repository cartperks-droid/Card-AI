"""Run a small wiring check with fresh random model weights, not team advice."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..catalog import load_catalog
from ..model import BattleModel, load_model_data, precompute_embeddings
from .pipeline import run_diagnostic_cycle
from .search import CardVariant, LegalInventory, SearchConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", choices=("python", "c"), default="c")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-mutations", action="store_true", help="Include Storm/Manga variants for eligible fixture cards")
    args = parser.parse_args()
    try:
        config = SearchConfig(candidate_count=args.candidates, steps=args.steps, seed=args.seed)
        if config.candidate_count**2 > 4096:
            raise ValueError("This diagnostic supports at most 64 candidates per side")
        torch.set_num_threads(min(4, torch.get_num_threads()))
        data = load_model_data()
        with torch.random.fork_rng():
            torch.manual_seed(args.seed)
            model = BattleModel().eval()
        embeddings = precompute_embeddings(model.description, data.description_tokens)
        # Fixture inventory, not a claim about the user's owned quantities.
        team = tuple(CardVariant(card_id, 1) for card_id in (3, 38, 44, 205))
        catalog = load_catalog()
        choices = tuple(CardVariant(v.card_id, v.border_id, mutation)
                        for v in team for mutation in (("None", "Storm", "Manga") if args.include_mutations and catalog.card(v.card_id).weather_id == 1 else ("None",)))
        inventory = LegalInventory((choices,) * 4, duplicate_policy="unique_card")
        cycle = run_diagnostic_cycle(model, embeddings, catalog, seed_team_a=team,
                  seed_team_b=team[::-1], inventory_a=inventory, inventory_b=inventory,
                  search_config=config, pack_ids=data.pack_ids, class_weights=data.class_weights, identity_keys=data.identity_keys,
                  backend=args.backend)
        report = {"purpose": "integration diagnostic; random weights, not trained team recommendations",
                  "mutation_variants_enabled": args.include_mutations, "first_side": 0, "training_labels_allowed": False,
                  "search_config": asdict(config),
                  "description_architecture": asdict(model.description.config),
                  "strategic_architecture": asdict(model.strategy.config),
                  "candidates_a": [asdict(c) for c in cycle.search_a.candidates],
                  "candidates_b": [asdict(c) for c in cycle.search_b.candidates],
                  "crossplay": [asdict(m) for m in cycle.crossplay.matches],
                  "experimental_mean_win_bounds_a": cycle.crossplay.mean_win_bounds(0),
                  "experimental_mean_win_bounds_b": cycle.crossplay.mean_win_bounds(1)}
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(f"Saved {len(cycle.crossplay.matches)} experimental matchups to {args.output.resolve()}; training labels disabled")
        else:
            print(text, end="")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
