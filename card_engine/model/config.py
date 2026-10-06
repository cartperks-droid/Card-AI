"""Architectures, independent of simulator rules and training labels."""

from dataclasses import dataclass
from ..mutations import MUTATION_NAMES


# Engine-side card classes (user-verified 2026-09-30); "sin" marks the seven-sin bosses.
CLASS_NAMES = ("undead", "demon", "dragon", "avian", "toy", "rng", "friendship", "sin")


@dataclass(frozen=True)
class DescriptionConfig:
    vocabulary_size: int = 441
    pad_id: int = 0
    bos_id: int = 439
    card_id: int = 440
    width: int = 128
    layers: int = 4  # User: 2 blocks is minimal for rule-based semantics.
    heads: int = 4
    feedforward_width: int = 512
    max_length: int = 64
    projection_hidden_width: int = 768
    output_width: int = 768
    dropout: float = 0.0

    def __post_init__(self):
        _positive(self.width, self.layers, self.heads, self.feedforward_width,
                  self.max_length, self.projection_hidden_width, self.output_width)
        if (self.vocabulary_size, self.pad_id, self.bos_id, self.card_id) != (441, 0, 439, 440):
            raise ValueError("Canonical vocabulary is 441 entries: PAD=0, BOS=439, CARD=440")
        if self.width % self.heads:
            raise ValueError("Description width must be divisible by heads")
        if self.max_length < 2 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid maximum length or dropout")


@dataclass(frozen=True)
class StrategicConfig:
    width: int = 768
    layers: int = 4
    heads: int = 8
    attention_width: int = 192
    feedforward_width: int = 3072
    dropout: float = 0.0
    # The order defines class-weight columns; memberships come from the user-verified data.
    class_names: tuple[str, ...] = CLASS_NAMES
    mutation_names: tuple[str, ...] = MUTATION_NAMES
    # Learned card-identity embedding, indexed by permanent identity keys (card_engine.card_keys),
    # never by list position. Spare rows let new cards join without resizing; 0 disables it.
    identity_capacity: int = 1024
    # Training-only: chance to drop a card's identity vector, so descriptions stay informative.
    identity_dropout: float = 0.1
    # L2 weight on identity vectors, added to the training loss via StrategicModel.identity_penalty():
    # memorisation stays possible but must pay for itself, so most understanding comes from language.
    identity_l2: float = 1e-3
    # GELU upscale width of the residual MLP that adds each card's normalized (HP, ATK) to its token.
    stat_hidden_width: int = 3072
    # Stats beside the card, not on it (user, 2026-10-06): > 0 gives the stats their own last stat_width channels of
    # each card token and the card (description, identity, classes, border, ...) a projection onto the rest, each part
    # layer-normalised on its own, so a huge stat gap cannot drown out the abilities. 0 adds both in one space.
    stat_width: int = 0
    # The engine reads a card's pack and mutation weather only for its stats and the stat supports that boost it,
    # all inside the card's stats already (2026-10-06): without these embeddings neither can stand in for identity.
    pack_embedding: bool = True
    mutation_embedding: bool = True

    def __post_init__(self):
        _positive(self.width, self.layers, self.heads, self.attention_width,
                  self.feedforward_width, self.stat_hidden_width)
        if self.attention_width % self.heads:
            raise ValueError("Internal attention width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        if type(self.stat_width) is not int or not 0 <= self.stat_width < self.width:
            raise ValueError("stat_width must be an integer in [0, width)")
        if (type(self.identity_capacity) is not int or self.identity_capacity < 0 or not 0 <= self.identity_dropout < 1
                or not self.identity_l2 >= 0):
            raise ValueError("identity_capacity must be a nonnegative integer and identity_dropout in [0, 1)")
        if not isinstance(self.class_names, tuple):
            raise ValueError("class_names must be an ordered tuple")
        if len(set(self.class_names)) != len(self.class_names) or any(
                not isinstance(n, str) or not n for n in self.class_names):
            raise ValueError("Class names must be unique nonempty strings")
        if (not isinstance(self.mutation_names, tuple) or any(not isinstance(n, str) or n not in MUTATION_NAMES for n in self.mutation_names)
                or len(set(self.mutation_names)) != len(self.mutation_names)
                or (self.mutation_names and self.mutation_names[0] != "None")):
            raise ValueError("Mutation names must be an explicit unique tuple starting with None, or empty for legacy models")


def _positive(*values):
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError("Architecture dimensions must be positive integers")
