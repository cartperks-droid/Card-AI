"""Versioned architecture+weights checkpoints; no optimizer or fabricated labels."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import torch

from .config import DescriptionConfig, StrategicConfig
from .networks import BattleModel, DescriptionModel, StrategicModel


def save_checkpoint(model, path, *, metadata=None):
    """Atomically save weights and JSON-compatible provenance supplied by caller."""
    if type(model) is BattleModel:
        kind = "battle"
        config = {"description": asdict(model.description_config),
                  "strategic": asdict(model.strategic_config)}
    elif type(model) is DescriptionModel:
        kind, config = "description", asdict(model.config)
    elif type(model) is StrategicModel:
        kind, config = "strategic", asdict(model.config)
    else:
        raise ValueError("Unsupported checkpoint model type")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON-compatible dictionary")
    # Round-trip removes arbitrary Python objects from the safe-load payload.
    metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
    payload = {"schema_version": 2, "kind": kind, "config": config,
               "state_dict": model.state_dict(), "training": model.training,
               "metadata": metadata}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, map_location="cpu"):
    """Reconstruct architecture and strictly load tensor weights; return (model, metadata)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    # No trained model exists yet, so only the current schema is accepted (no legacy shims).
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("Unsupported model checkpoint schema")
    kind, config = payload.get("kind"), payload.get("config")
    if kind == "battle":
        model = BattleModel(DescriptionConfig(**config["description"]),
                            StrategicConfig(**config["strategic"]))
    elif kind == "description":
        model = DescriptionModel(DescriptionConfig(**config))
    elif kind == "strategic":
        model = StrategicModel(StrategicConfig(**config))
    else:
        raise ValueError("Unknown checkpoint model kind")
    # Assign preserves checkpoint tensor dtypes as well as values; a double/half
    # checkpoint must not silently round through a newly constructed float model.
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    model.to(map_location)
    model.train(payload.get("training", False))
    return model, payload.get("metadata", {})
