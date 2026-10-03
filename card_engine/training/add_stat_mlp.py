"""One-off: add the zero-initialised stat MLP to a run trained before it existed. Delete after migrating.

    python -m card_engine.training.add_stat_mlp [--run-dir data/training]

The new projection and MLP output layer are zero, so the upgraded model predicts exactly as before; training then learns
to use the card stats. Adam's moments for every existing weight are kept. Originals are kept as *.pre-stat-mlp.
"""

import shutil
from pathlib import Path

import torch

from ..model import BattleModel
from ..model.checkpoint import save_checkpoint
from ..model.config import DescriptionConfig, StrategicConfig
from .train import RUN_DIR


def upgrade_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = BattleModel(DescriptionConfig(**payload["config"]["description"]), StrategicConfig(**payload["config"]["strategic"]))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected or not missing or any(not key.startswith(("strategy.stat_projection.", "strategy.stat_mlp.")) for key in missing):
        raise ValueError(f"{path}: not a checkpoint from before the stat MLP")
    model.train(payload.get("training", False))
    shutil.copy2(path, path.with_name(path.name + ".pre-stat-mlp"))
    save_checkpoint(model, path, metadata=payload.get("metadata", {}))
    return model


def upgrade_optimizer(path, model):
    """The stat projection's and MLP's parameters come last (StrategicModel registers it last): append them, without state."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    group = state["optimizer"]["param_groups"][0]
    parameters = list(model.parameters())
    new = [p for name, p in model.named_parameters() if name.startswith(("strategy.stat_projection.", "strategy.stat_mlp."))]
    if len(state["optimizer"]["param_groups"]) != 1 or len(group["params"]) + len(new) != len(parameters) \
            or any(a is not b for a, b in zip(parameters[-len(new):], new)):
        raise ValueError(f"{path}: optimizer does not match the model before the stat MLP")
    group["params"] = [*group["params"], *range(max(group["params"]) + 1, max(group["params"]) + 1 + len(new))]
    torch.optim.AdamW(parameters).load_state_dict(state["optimizer"])  # validates the result
    shutil.copy2(path, path.with_name(path.name + ".pre-stat-mlp"))
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main(run_dir=RUN_DIR):
    run_dir = Path(run_dir)
    model = upgrade_checkpoint(run_dir / "model.checkpoint")
    upgrade_optimizer(run_dir / "trainer.pt", model)
    if (run_dir / "best.checkpoint").exists():
        upgrade_checkpoint(run_dir / "best.checkpoint")
    print(f"Upgraded {run_dir}: model.checkpoint, trainer.pt and best.checkpoint now include the stat MLP")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=RUN_DIR)
    main(parser.parse_args().run_dir)
