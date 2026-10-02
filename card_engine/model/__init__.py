"""Optional PyTorch models. Importing the main card_engine package needs no torch."""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import DescriptionConfig, StrategicConfig
from .data import ModelData, load_model_data, pad_descriptions, precompute_embeddings
from .networks import (
    BattleModel, DescriptionEncoder, DescriptionModel, NarrowSelfAttention,
    OUTCOME_NAMES, SLOT_NAMES, StrategicModel,
)

__all__ = [
    "BattleModel", "DescriptionConfig", "DescriptionEncoder", "DescriptionModel",
    "StrategicConfig", "StrategicModel", "NarrowSelfAttention", "ModelData",
    "load_model_data", "pad_descriptions", "precompute_embeddings",
    "load_checkpoint", "save_checkpoint", "OUTCOME_NAMES", "SLOT_NAMES",
]
