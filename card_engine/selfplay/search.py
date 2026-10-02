"""Differentiate through strategic inputs, then sample and rescore legal teams.

Inventory is an explicit caller assertion, never inferred from screenshot
brightness or class evidence. Supports and the opposing lineup stay fixed.
Scores from any checkpoint remain model estimates, not simulator labels.
"""

from dataclasses import dataclass
import hashlib
import math
import random

import torch
from torch import Tensor

from ..model.networks import BattleModel, StrategicModel
from ..mutations import MUTATION_NAMES, effective_mutation


@dataclass(frozen=True, order=True)
class CardVariant:
    card_id: int
    border_id: int
    mutation: str = "None"

    def __post_init__(self):
        if type(self.card_id) is not int or self.card_id < 1:
            raise ValueError("card_id must be a positive integer")
        if type(self.border_id) is not int or not 1 <= self.border_id <= 16:
            raise ValueError("border_id must be in 1..16")
        effective_mutation("None", self.mutation)


@dataclass(frozen=True)
class LegalInventory:
    """Four ordered pools of caller-authorized card/border variants.

    duplicate_policy is deliberately required: allow, unique_card, or
    unique_variant. Quantities and other game-specific restrictions must be
    represented by these pools/policy or checked outside this API.
    """

    slots: tuple[tuple[CardVariant, ...], ...]
    duplicate_policy: str

    def __post_init__(self):
        if self.duplicate_policy not in ("allow", "unique_card", "unique_variant"):
            raise ValueError("Choose duplicate_policy: allow, unique_card, or unique_variant")
        if (not isinstance(self.slots, tuple) or len(self.slots) != 4
                or any(not isinstance(row, tuple) or not row for row in self.slots)):
            raise ValueError("Inventory requires four nonempty tuple pools in lineup order")
        if any(not isinstance(v, CardVariant) for row in self.slots for v in row):
            raise ValueError("Inventory choices must be CardVariant objects")
        if any(len(set(row)) != len(row) for row in self.slots):
            raise ValueError("Duplicate inventory entries would bias sampling")
        if not _can_complete(self, 0, set()):
            raise ValueError("Inventory has no complete team under the duplicate policy")

    def permits(self, team):
        if len(team) != 4 or any(v not in self.slots[i] for i, v in enumerate(team)):
            return False
        keys = [_variant_key(v, self.duplicate_policy) for v in team]
        return self.duplicate_policy == "allow" or len(set(keys)) == 4


@dataclass(frozen=True)
class SearchConfig:
    candidate_count: int = 32
    steps: int = 24
    learning_rate: float = 0.1
    nearest_k: int = 8
    distance_temperature: float = 0.25
    initial_noise: float = 0.02
    anchor_penalty: float = 0.001
    seed: int = 0

    def __post_init__(self):
        for name in ("candidate_count", "nearest_k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        for name in ("learning_rate", "distance_temperature", "initial_noise", "anchor_penalty"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.distance_temperature == 0:
            raise ValueError("learning_rate and distance_temperature must be positive")


@dataclass(frozen=True)
class SearchCandidate:
    team: tuple[CardVariant, ...]
    model_probabilities: tuple[float, float]
    target_score: float
    seed: int


@dataclass(frozen=True)
class SearchResult:
    """Candidate order is seed order, not rank order; repeats are possible."""

    candidates: tuple[SearchCandidate, ...]
    relaxed_embeddings: Tensor
    target_side: int
    score_status: str = "uncalibrated_model_estimates"
    training_labels_allowed: bool = False
    relaxation: str = "description_plus_border_mutation_pack_and_selected_classes"


def _variant_key(variant, policy):
    return variant.card_id if policy == "unique_card" else variant


def _can_complete(inventory, start, used):
    """Four-slot bipartite matching prevents a greedy choice from dead-ending."""
    if inventory.duplicate_policy == "allow":
        return True
    matched = {}

    def augment(slot, seen):
        for variant in inventory.slots[slot]:
            key = _variant_key(variant, inventory.duplicate_policy)
            if key in used or key in seen:
                continue
            seen.add(key)
            if key not in matched or augment(matched[key], seen):
                matched[key] = slot
                return True
        return False

    return all(augment(slot, set()) for slot in range(start, 4))


def _child_seed(seed, index, purpose):
    digest = hashlib.blake2b(f"{seed}:{index}:{purpose}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _team(team, name, card_count):
    team = tuple(team)
    if (len(team) != 4 or any(not isinstance(v, CardVariant) for v in team)
            or any(v.card_id > card_count for v in team)):
        raise ValueError(f"{name} must contain four valid CardVariant objects")
    return team


def _support_ids(values, name, maximum, device):
    values = tuple(values)
    if len(values) != 2 or any(type(v) is not int or not 0 <= v <= maximum for v in values):
        raise ValueError(f"{name} must contain side A/B IDs in 0..{maximum}")
    return torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)


def generate_candidates(
    model: StrategicModel | BattleModel,
    card_embeddings: Tensor,
    *,
    seed_team,
    opponent_team,
    inventory: LegalInventory,
    target_side: int,
    red_support_ids=(0, 0),
    blue_support_ids=(0, 0),
    pack_ids: Tensor | None = None,
    class_weights: Tensor | None = None,
    identity_keys: Tensor | None = None,
    intrinsic_weathers=None,
    config: SearchConfig | None = None,
) -> SearchResult:
    """Optimize four ordered latent slots and return discrete legal candidates.

    card_embeddings is a [card_count, width] snapshot indexed by card_id - 1;
    regenerate it after description-model updates. pack_ids and class_weights
    follow the same row order. Class evidence selection is the caller's choice
    (load_model_data defaults to verified_only). Each candidate uses independent
    local noise/sampling seeds, preserving existing global random state.

    The relaxed vector contains description + border + pack + class + identity terms.
    During relaxation the model receives border ID 1, with that embedding
    subtracted from its input, and no separate pack/class metadata. This exactly
    reproduces the current additive metadata rule at every discrete variant.
    Nearest neighbors use RMS Euclidean distance in this combined space. For
    each slot, sample among its nearest k choices that permit a legal remaining
    lineup, with probability proportional to exp(-distance / temperature).
    Each candidate is rescored with its actual IDs and original metadata.

    Only input vectors change; parameters, existing parameter gradients,
    requires_grad flags and every module's train/eval mode are preserved. This
    function supports the existing additive StrategicModel metadata interface;
    changing that interface requires a matching search implementation.
    """
    strategy = model.strategy if isinstance(model, BattleModel) else model
    if torch.is_inference_mode_enabled():
        raise ValueError("Gradient search cannot run inside torch.inference_mode")
    if not isinstance(strategy, StrategicModel):
        raise ValueError("model must be StrategicModel or BattleModel")
    if not isinstance(inventory, LegalInventory):
        raise ValueError("Explicit LegalInventory is required")
    config = config or SearchConfig()
    if not isinstance(config, SearchConfig):
        raise ValueError("config must be SearchConfig")
    if type(target_side) is not int or target_side not in (0, 1):
        raise ValueError("target_side must be 0 (A) or 1 (B)")
    device, dtype = strategy.bos.device, strategy.bos.dtype
    if (not isinstance(card_embeddings, Tensor) or not card_embeddings.is_floating_point()
            or card_embeddings.ndim != 2 or card_embeddings.shape[0] < 1
            or card_embeddings.shape[1] != strategy.config.width
            or not bool(torch.isfinite(card_embeddings).all())):
        raise ValueError("card_embeddings must be a finite [card_count, model_width] tensor")
    table = card_embeddings.detach().to(device=device, dtype=dtype)
    if not bool(torch.isfinite(table).all()):
        raise ValueError("card_embeddings overflow the model dtype")
    count = len(table)
    seed_team = _team(seed_team, "seed_team", count)
    opponent_team = _team(opponent_team, "opponent_team", count)
    if not inventory.permits(seed_team):
        raise ValueError("seed_team is not permitted by the explicit inventory")
    if any(v.card_id > count for slot in inventory.slots for v in slot):
        raise ValueError("Inventory card ID exceeds the embedding table")
    mutation_indices = {name: index for index, name in enumerate(strategy.config.mutation_names)} or {"None": 0}
    all_variants = (*seed_team, *opponent_team, *(v for slot in inventory.slots for v in slot))
    if intrinsic_weathers is None and any(v.mutation != "None" for v in all_variants):
        raise ValueError("Mutation search requires explicit intrinsic_weathers eligibility for the embedding table")
    intrinsic_weathers = ("None",) * count if intrinsic_weathers is None else tuple(intrinsic_weathers)
    if len(intrinsic_weathers) != count or any(not isinstance(n, str) or n not in (*MUTATION_NAMES, "Rapture") for n in intrinsic_weathers):
        raise ValueError("intrinsic_weathers must contain one known weather name per embedding row")

    def mutation_name(variant):
        return effective_mutation(intrinsic_weathers[variant.card_id - 1], variant.mutation)

    if any(mutation_name(v) not in mutation_indices for v in all_variants):
        raise ValueError("A chosen mutation is missing from the model's saved mutation vocabulary")
    if pack_ids is None:
        pack_ids = torch.zeros(count, dtype=torch.long, device=device)
    if (not isinstance(pack_ids, Tensor) or pack_ids.dtype != torch.long
            or tuple(pack_ids.shape) != (count,)
            or bool(((pack_ids < 0) | (pack_ids > 14)).any())):
        raise ValueError("pack_ids must be a card_count torch.long vector in 0..14")
    packs = pack_ids.detach().to(device)
    identities = None
    if identity_keys is not None and strategy.identity_embedding is not None:
        if (not isinstance(identity_keys, Tensor) or identity_keys.dtype != torch.long or tuple(identity_keys.shape) != (count,)
                or bool(((identity_keys < 0) | (identity_keys > strategy.config.identity_capacity)).any())):
            raise ValueError("identity_keys must be a card_count torch.long vector within the model's identity capacity")
        identities = identity_keys.detach().to(device)
    classes = None
    if class_weights is not None:
        if (strategy.class_embedding is None or not isinstance(class_weights, Tensor)
                or not class_weights.is_floating_point()
                or tuple(class_weights.shape) != (count, len(strategy.config.class_names))
                or not bool(torch.isfinite(class_weights).all())
                or bool((class_weights < 0).any())):
            raise ValueError("class_weights must match the model class vocabulary and be finite/nonnegative")
        classes = class_weights.detach().to(device=device, dtype=dtype)
        if not bool(torch.isfinite(classes).all()):
            raise ValueError("class_weights overflow the model dtype")
    red = _support_ids(red_support_ids, "red_support_ids", 28, device)
    blue = _support_ids(blue_support_ids, "blue_support_ids", 15, device)

    def actual_inputs(teams):
        # teams is an ordered batch of (side A, side B) lineups.
        ids = torch.tensor([[[v.card_id for v in side] for side in pair] for pair in teams],
                           dtype=torch.long, device=device) - 1
        borders = torch.tensor([[[v.border_id for v in side] for side in pair] for pair in teams],
                               dtype=torch.long, device=device)
        metadata = {"pack_ids": packs[ids]}
        metadata["mutation_ids"] = torch.tensor([[[mutation_indices[mutation_name(v)] for v in side]
                                                   for side in pair] for pair in teams], dtype=torch.long, device=device)
        if classes is not None:
            metadata["class_weights"] = classes[ids]
        if identities is not None:
            metadata["identity_keys"] = identities[ids]
        return table[ids], borders, metadata

    def combine(variants):
        ids = torch.tensor([v.card_id - 1 for v in variants], device=device)
        borders = torch.tensor([v.border_id - 1 for v in variants], device=device)
        combined = table[ids] + strategy.border_embedding(borders)
        if strategy.mutation_embedding is not None:
            mutation_ids = torch.tensor([mutation_indices[mutation_name(v)] for v in variants], device=device)
            mutations = strategy.mutation_embedding(mutation_ids)
            combined = combined + torch.where(mutation_ids[:, None] > 0, mutations, 0.0)
        pack_values = strategy.pack_embedding(packs[ids].clamp_min(1) - 1)
        combined = combined + torch.where(packs[ids, None] > 0, pack_values, 0.0)
        if classes is not None:
            combined = combined + classes[ids] @ strategy.class_embedding.weight
        if identities is not None:
            combined = combined + strategy.identity_embedding(identities[ids])
        return combined.detach()

    modes = [(module, module.training) for module in strategy.modules()]
    try:
        strategy.eval()
        with torch.no_grad():
            anchor = combine(seed_team)
            opponent = combine(opponent_team)
            choices = tuple(combine(slot) for slot in inventory.slots)
            base_border = strategy.border_embedding.weight[0].detach()
            if not all(bool(torch.isfinite(v).all()) for v in (anchor, opponent, *choices)):
                raise ValueError("Combined card metadata contains non-finite values")
        # Optimize each candidate separately so changing candidate_count cannot
        # change an existing candidate through batch-size arithmetic differences.
        relaxed = []
        candidates = []
        for index in range(config.candidate_count):
            noise_generator = torch.Generator(device="cpu")
            noise_generator.manual_seed(_child_seed(config.seed, index, "noise"))
            noise = torch.randn(anchor.shape, generator=noise_generator, dtype=torch.float64)
            latent = (anchor + config.initial_noise * noise.to(device=device, dtype=dtype)).detach()
            if not bool(torch.isfinite(latent).all()):
                raise ValueError("Non-finite initial search representation")
            with torch.enable_grad():
                for _ in range(config.steps):
                    latent.requires_grad_(True)
                    sides = [opponent, opponent]
                    sides[target_side] = latent
                    inputs = torch.stack(sides).unsqueeze(0) - base_border
                    logits = strategy(inputs, torch.ones((1, 2, 4), dtype=torch.long, device=device), red, blue)
                    objective = logits.log_softmax(-1)[0, target_side]
                    objective = objective - config.anchor_penalty * (latent - anchor).square().mean()
                    gradient, = torch.autograd.grad(objective, latent)
                    if not bool(torch.isfinite(gradient).all()):
                        raise ValueError("Non-finite search gradient")
                    # Normalize each slot's gradient to avoid model-width scaling.
                    scale = gradient.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
                    latent = (latent + config.learning_rate * gradient / scale).detach()
                    if not bool(torch.isfinite(latent).all()):
                        raise ValueError("Non-finite relaxed card representation")
            sampling_seed = _child_seed(config.seed, index, "sample")
            rng = random.Random(sampling_seed)
            team, used = [], set()
            for slot in range(4):
                distances = (choices[slot].double() - latent[slot].double()).square().mean(-1).sqrt().cpu().tolist()
                if not all(math.isfinite(distance) for distance in distances):
                    raise ValueError("Non-finite nearest-neighbor distances")
                viable = []
                for choice_index, variant in enumerate(inventory.slots[slot]):
                    key = _variant_key(variant, inventory.duplicate_policy)
                    if inventory.duplicate_policy != "allow" and key in used:
                        continue
                    if _can_complete(inventory, slot + 1, used | {key}):
                        viable.append((distances[choice_index], choice_index, variant, key))
                nearest = sorted(viable, key=lambda item: (item[0], item[1]))[:config.nearest_k]
                if not nearest:
                    raise RuntimeError("Inventory feasibility changed during search")
                minimum = nearest[0][0]
                weights = [math.exp(-(item[0] - minimum) / config.distance_temperature) for item in nearest]
                selected = rng.choices(nearest, weights=weights, k=1)[0]
                team.append(selected[2])
                used.add(selected[3])
            team = tuple(team)
            sides = [opponent_team, opponent_team]
            sides[target_side] = team
            with torch.no_grad():
                inputs, borders, metadata = actual_inputs([sides])
                probability = strategy(inputs, borders, red, blue, **metadata).softmax(-1)[0]
                if not bool(torch.isfinite(probability).all()):
                    raise ValueError("Non-finite model score for a discrete candidate")
            probabilities = tuple(float(v) for v in probability.cpu())
            candidates.append(SearchCandidate(team, probabilities, probabilities[target_side], sampling_seed))
            relaxed.append(latent.detach().cpu())
        return SearchResult(tuple(candidates), torch.stack(relaxed), target_side)
    finally:
        # Assign flags directly to preserve intentionally mixed submodule modes.
        for module, training in modes:
            module.training = training
