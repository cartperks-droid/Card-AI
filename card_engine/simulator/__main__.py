"""Run the current supported card subset and inspect experimental event traces."""

import argparse
from dataclasses import asdict
import json

from ..catalog import load_catalog
from .catalog_rules import compile_battle, coverage, support_coverage
from .reference import Options, simulate


def ids(value):
    return [int(x.strip()) for x in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-a", type=ids, default=[3])
    parser.add_argument("--team-b", type=ids, default=[3])
    parser.add_argument("--borders-a", type=ids)
    parser.add_argument("--borders-b", type=ids)
    parser.add_argument("--mutations-a", help="Comma-separated exact mutation names, one per card")
    parser.add_argument("--mutations-b", help="Comma-separated exact mutation names, one per card")
    parser.add_argument("--red-support-a", type=int, default=0)
    parser.add_argument("--red-support-b", type=int, default=0)
    parser.add_argument("--blue-support-a", type=int, default=0)
    parser.add_argument("--blue-support-b", type=int, default=0)
    parser.add_argument("--first-side", type=int, choices=(0, 1), default=0)
    parser.add_argument("--mode", choices=("sample", "branch"), default="branch")
    parser.add_argument("--rounding", choices=("ceil", "unrounded", "floor", "nearest_half_up"), default="ceil")
    parser.add_argument("--backend", choices=("python", "c"), default="python")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    catalog = load_catalog()
    if args.coverage:
        rows = coverage(catalog)
        print(json.dumps({"experimental_supported": [r for r in rows if r["status"] != "unsupported"],
                          "unsupported_count": sum(r["status"] == "unsupported" for r in rows),
                          "supports": support_coverage(catalog),
                          "training_labels_allowed": False}, indent=2))
        return
    if args.trace and (args.mode != "sample" or args.backend != "python"):
        parser.error("--trace requires --mode sample --backend python")
    borders = (args.borders_a or [1]*len(args.team_a), args.borders_b or [1]*len(args.team_b))
    mutations = tuple([name.strip() for name in text.split(",")] if text is not None else ["None"] * len(team)
                      for text, team in ((args.mutations_a, args.team_a), (args.mutations_b, args.team_b)))
    try:
        battle = compile_battle(catalog, (args.team_a, args.team_b), borders=borders,
                                mutations=mutations, red_supports=(args.red_support_a, args.red_support_b),
                                blue_supports=(args.blue_support_a, args.blue_support_b), first_side=args.first_side)
        options = Options(mode=args.mode, seed=args.seed, rounding=args.rounding)
        if args.backend == "python":
            result = asdict(simulate(battle, options, trace=args.trace))
        else:
            from .native import simulate as native_simulate
            result = native_simulate(asdict(battle), asdict(options))
            result.update(rules_status="experimental_subset", training_labels_allowed=False)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
