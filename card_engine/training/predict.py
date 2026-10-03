"""Compare the trained predictor with the simulator on one matchup (side A initiates).

Example:
    python -m card_engine.training.predict --a 3,205,44,1 --b 21,104,18,10 --borders-a 8,8,8,8 --red-a 12:3
"""

import argparse
import json

import torch

from ..catalog import load_catalog
from ..model.checkpoint import load_checkpoint
from .flags import snapshot
from .labels import label_specs
from .tablebase import Tablebase
from .train import RUN_DIR, Inputs, card_table


def _ints(text, default, count=4):
    return [int(x) for x in text.split(",")] if text else [default] * count


def _support(text):
    if not text:
        return 0, 0
    support, _, tier = text.partition(":")
    return int(support), int(tier or 1)


def predict(spec, *, run_dir=RUN_DIR, device="cpu"):
    model, metadata = load_checkpoint(f"{run_dir}/model.checkpoint", map_location=device)
    model.eval()
    inputs = Inputs(device)
    rows = {key: torch.tensor([value], device=device) for key, value in spec.items()}
    with torch.no_grad():
        model_p = model(**inputs(rows, card_table(model, inputs.data.description_tokens))).softmax(-1)[0].tolist()
    probs, exact = label_specs(load_catalog(), [spec], seed=12345, tablebase=Tablebase(snapshot()))
    probs, exact = probs[0], bool(exact[0])
    finished = max(1e-12, float(probs[:2].sum()))
    return {"model": dict(zip(("A", "B"), (round(p, 3) for p in model_p))),
            "simulator": dict(zip(("A", "B"), (round(float(c) / finished, 3) for c in probs[:2]))),
            "simulator_exact": exact, "unfinished": round(float(probs[3]), 4), "checkpoint_step": metadata.get("step")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="four card IDs")
    parser.add_argument("--b", required=True)
    for side in ("a", "b"):
        parser.add_argument(f"--borders-{side}", help="four border IDs (default 1)")
        parser.add_argument(f"--mutations-{side}", help="four mutation indices (default 0 = none)")
        parser.add_argument(f"--red-{side}", help="red support id[:tier]")
        parser.add_argument(f"--blue-{side}", help="blue support id[:tier]")
    parser.add_argument("--run-dir", default=RUN_DIR, help="training run whose model.checkpoint is used")
    args = parser.parse_args()
    spec = {"cards": [], "borders": [], "mutations": [], "arts": [], "red": [], "red_tier": [], "blue": [], "blue_tier": []}
    for side in ("a", "b"):
        cards = _ints(getattr(args, side), 0)
        spec["cards"].append(cards)
        spec["borders"].append(_ints(getattr(args, f"borders_{side}"), 1))
        spec["mutations"].append(_ints(getattr(args, f"mutations_{side}"), 0))
        spec["arts"].append([0] * 4)  # Astraeus draws its art in battle
        for color in ("red", "blue"):
            support, tier = _support(getattr(args, f"{color}_{side}"))
            spec[color].append(support)
            spec[color + "_tier"].append(tier if support else 0)
    print(json.dumps(predict(spec, run_dir=args.run_dir), indent=2))


if __name__ == "__main__":
    main()
