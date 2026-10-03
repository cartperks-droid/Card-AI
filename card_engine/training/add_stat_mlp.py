"""One-off: add the zero-initialised stat MLP to a run trained before it existed. Delete after migrating.

    python -m card_engine.training.add_stat_mlp [--run-dir data/training]

The new projection and MLP output layer are zero, so the upgraded model predicts exactly as before; training then learns
to use the card stats. Adam's moments for every existing weight are kept. Originals are kept as *.pre-stat-mlp.
Files that already include the stat MLP are left alone, so it is safe to run again.
"""

import shutil
from pathlib import Path

import torch

from ..model import BattleModel
from ..model.checkpoint import save_checkpoint
from ..model.config import DescriptionConfig, StrategicConfig
from .train import RUN_DIR


STAT_PREFIXES = ("strategy.stat_projection.", "strategy.stat_mlp.")


def upgrade_checkpoint(path):
    """Return (model, upgraded)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = BattleModel(DescriptionConfig(**payload["config"]["description"]), StrategicConfig(**payload["config"]["strategic"]))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected or any(not key.startswith(STAT_PREFIXES) for key in missing):
        raise ValueError(f"{path}: does not match the model with or without the stat MLP")
    if not missing:
        return model, False
    model.train(payload.get("training", False))
    shutil.copy2(path, path.with_name(path.name + ".pre-stat-mlp"))
    save_checkpoint(model, path, metadata=payload.get("metadata", {}))
    return model, True


def upgrade_optimizer(path, model):
    """The stat projection's and MLP's parameters come last (StrategicModel registers them last): append them, without
    state. Return whether the file changed."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    group = state["optimizer"]["param_groups"][0]
    parameters = list(model.parameters())
    if len(state["optimizer"]["param_groups"]) == 1 and len(group["params"]) == len(parameters):
        return False
    new = [p for name, p in model.named_parameters() if name.startswith(STAT_PREFIXES)]
    if len(state["optimizer"]["param_groups"]) != 1 or len(group["params"]) + len(new) != len(parameters) \
            or any(a is not b for a, b in zip(parameters[-len(new):], new)):
        raise ValueError(f"{path}: optimizer does not match the model before the stat MLP")
    group["params"] = [*group["params"], *range(max(group["params"]) + 1, max(group["params"]) + 1 + len(new))]
    torch.optim.AdamW(parameters).load_state_dict(state["optimizer"])  # validates the result
    shutil.copy2(path, path.with_name(path.name + ".pre-stat-mlp"))
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return True


def main(run_dir=RUN_DIR):
    run_dir = Path(run_dir)
    model, changed = upgrade_checkpoint(run_dir / "model.checkpoint")
    report = {"model.checkpoint": changed, "trainer.pt": upgrade_optimizer(run_dir / "trainer.pt", model)}
    if (run_dir / "best.checkpoint").exists():
        report["best.checkpoint"] = upgrade_checkpoint(run_dir / "best.checkpoint")[1]
    for name, changed in report.items():
        print(f"{name}: {'upgraded' if changed else 'already includes the stat MLP'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=RUN_DIR)
    main(parser.parse_args().run_dir)
