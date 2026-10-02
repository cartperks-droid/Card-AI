"""Inspect the migrated dataset: python3 -m card_engine [--card 33]."""

import argparse
from dataclasses import asdict
import json

from .catalog import load_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Path to a generated dataset.json")
    parser.add_argument("--card", type=int, help="One-based Snap card ID")
    parser.add_argument("--border", type=int, default=1, help="One-based border ID for dataset base stats")
    parser.add_argument("--mutation", default="None", help="Exact mutation name, e.g. Storm or Manga")
    args = parser.parse_args()
    catalog = load_catalog(args.dataset)
    if args.card is not None:
        try:
            result = asdict(catalog.card(args.card))
            from .stats import base_stats
            result["base_stats_before_supports_and_abilities"] = asdict(base_stats(catalog, args.card, args.border, mutation=args.mutation))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        result = {
            "cards": len(catalog.cards), "red_supports": len(catalog.red_supports),
            "blue_supports": len(catalog.blue_supports), "borders": len(catalog.borders),
            "weathers": len(catalog.weathers), "vocabulary_including_specials": len(catalog.vocabulary),
            "simulation_ready": False,
            "next_step": "Validate gameplay rules using docs/simulator_questions.md",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
