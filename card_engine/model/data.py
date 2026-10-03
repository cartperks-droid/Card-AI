"""Load canonical description tokens and explicitly selected metadata evidence."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import Tensor

from ..schema import validate_dataset
from .config import CLASS_NAMES
from .networks import DescriptionModel, _long_ids, _range

from ..teams import ASTRAEUS_ARTS


CLASS_POLICIES = ("disabled", "verified_only", "wiki_supported", "include_candidates")


def pad_descriptions(sequences, *, pad_to=None, device=None) -> Tensor:
    """Pad complete [BOS, lexical..., CARD] rows without moving CARD past PAD."""
    sequences = [list(row) for row in sequences]
    if not sequences:
        raise ValueError("At least one description is required")
    for row in sequences:
        if (len(row) < 2 or row[0] != 439 or row[-1] != 440
                or any(type(t) is not int for t in row)
                or any(not 1 <= t <= 438 for t in row[1:-1])):
            raise ValueError("Expected canonical unpadded BOS=439, lexical IDs 1..438, CARD=440")
    required = max(map(len, sequences))
    if pad_to is None:
        pad_to = required
    if type(pad_to) is not int or pad_to < required:
        raise ValueError("pad_to must fit the longest complete description")
    result = torch.zeros(len(sequences), pad_to, dtype=torch.long, device=device)
    for index, row in enumerate(sequences):
        result[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return result


@dataclass(frozen=True)
class ModelData:
    """Tables are indexed by card_id - 1; source IDs and lineup order are stable."""

    names: tuple[str, ...]
    description_tokens: Tensor
    pack_ids: Tensor
    class_weights: Tensor | None
    class_names: tuple[str, ...]
    class_policy: str
    identity_keys: Tensor  # Permanent per-card keys (card_engine.card_keys), not positions.
    art_identity_keys: Tensor  # [1 + arts]: Astraeus's arts are separate cards (user); index 0 = no art.

    def gather(self, card_ids: Tensor) -> dict[str, Tensor]:
        """Gather descriptions/metadata in exactly the requested card-slot order."""
        _long_ids(card_ids, "card_ids")
        _range(card_ids, "card_ids", 1, len(self.names))
        if card_ids.device != self.description_tokens.device:
            raise ValueError("card_ids must be on the data-table device")
        indices = card_ids - 1
        result = {"description_tokens": self.description_tokens[indices],
                  "pack_ids": self.pack_ids[indices],
                  "identity_keys": self.identity_keys[indices]}
        if self.class_weights is not None:
            result["class_weights"] = self.class_weights[indices]
        return result


def load_model_data(path=None, *, class_names=CLASS_NAMES, class_policy="verified_only",
                    pad_to=None, device=None) -> ModelData:
    """Default excludes all unverified class candidates.

    wiki_supported/include_candidates are explicit experimental opt-ins, not
    claims that the simulator has verified those classes. Pack IDs come from
    the cleaned user-supported metadata; no pack is inferred from card names.
    """
    if class_policy not in CLASS_POLICIES:
        raise ValueError(f"class_policy must be one of {CLASS_POLICIES}")
    class_names = tuple(class_names)
    if len(set(class_names)) != len(class_names) or any(not isinstance(n, str) or not n for n in class_names):
        raise ValueError("Class vocabulary must contain unique nonempty names")
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data/clean/dataset.json"
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(dataset)
    descriptions = pad_descriptions([row["token_ids"] for row in dataset["tokens"]["rows"]],
                                   pad_to=pad_to, device=device)
    packs = []
    class_weights = torch.zeros(len(dataset["cards"]), len(class_names), device=device) if class_names else None
    class_index = {name: index for index, name in enumerate(class_names)}
    for index, card in enumerate(dataset["cards"]):
        pack_id = card.get("pack_id") or 0
        if type(pack_id) is not int or not 0 <= pack_id <= 14:
            raise ValueError(f"Card {index + 1} has an invalid pack ID")
        packs.append(pack_id)
        if class_weights is None or class_policy == "disabled":
            continue
        metadata = card.get("metadata", {})
        members = set()
        if metadata.get("gameplay_classes_verified") is True:
            members.update(card.get("classes") or [])
        if class_policy in ("wiki_supported", "include_candidates"):
            members.update(row["class_name"] for row in metadata.get("sourced_class_memberships", [])
                           if row.get("status") == "wiki_supported_unverified_current")
        if class_policy == "include_candidates":
            members.update(metadata.get("gameplay_class_candidates", []))
        for member in members:
            if member in class_index:
                class_weights[index, class_index[member]] = 1.0
    from ..card_keys import assign_identity_keys
    art_keys = [0] + assign_identity_keys([f"Astraeus ({art})" for art in ASTRAEUS_ARTS])
    return ModelData(tuple(row["name"] for row in dataset["cards"]), descriptions,
                     torch.tensor(packs, dtype=torch.long, device=device), class_weights,
                     class_names, class_policy,
                     torch.tensor([card["identity_key"] for card in dataset["cards"]], dtype=torch.long, device=device),
                     torch.tensor(art_keys, dtype=torch.long, device=device))


def precompute_embeddings(model: DescriptionModel, token_ids: Tensor, *, batch_size=64) -> Tensor:
    """Return a detached CPU snapshot, not a permanently valid training cache.

    Recompute after description-model updates. No parameters are frozen and
    the model's previous train/eval mode is restored, even when encoding fails.
    """
    if not isinstance(model, DescriptionModel):
        raise ValueError("precompute_embeddings expects a DescriptionModel")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    _long_ids(token_ids, "token_ids")
    if token_ids.ndim != 2 or token_ids.shape[0] < 1:
        raise ValueError("token_ids must contain a nonempty batch")
    was_training = model.training
    device = next(model.parameters()).device
    try:
        model.eval()
        with torch.no_grad():
            return torch.cat([model(part.to(device)).cpu() for part in token_ids.split(batch_size)], dim=0)
    finally:
        model.train(was_training)
