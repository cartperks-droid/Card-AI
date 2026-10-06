"""Torch-only description and ordered-team transformers.

Model outputs are uncalibrated until trained on verified battle probabilities.
No battle outcomes are supplied or inferred by this module.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DescriptionConfig, StrategicConfig


OUTCOME_NAMES = ("a_win", "b_win")  # no ties: the attacker (A) loses if both sides are wiped out (user)
SLOT_NAMES = (
    "BOS", "ally_red", "ally_blue", "ally_card_1", "ally_card_2",
    "ally_card_3", "ally_card_4", "enemy_red", "enemy_blue", "enemy_card_1",
    "enemy_card_2", "enemy_card_3", "enemy_card_4", "MODE", "PREDICT",
)


def _long_ids(value: Tensor, name: str, shape=None):
    if not isinstance(value, Tensor) or value.dtype != torch.long:
        raise ValueError(f"{name} must be a torch.long tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")


def _range(value: Tensor, name: str, low: int, high: int):
    if bool(((value < low) | (value > high)).any()):
        raise ValueError(f"{name} must be in {low}..{high}")


def _visibility(mask, shape, device, name):
    if mask is None:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape):
        raise ValueError(f"{name} must be a bool tensor of shape {tuple(shape)}")
    if mask.device != device:
        raise ValueError(f"{name} must be on the input device")
    return mask


class NarrowSelfAttention(nn.Module):
    """Project model-width states to a separately sized Q/K/V attention space."""

    def __init__(self, width, attention_width, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_width = attention_width // heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * attention_width)
        self.output = nn.Linear(attention_width, width)

    def forward(self, x, *, key_valid=None, causal=False):
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        allowed = None if key_valid is None else key_valid[:, None, None, :]
        if causal:
            causal_allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            allowed = causal_allowed if allowed is None else allowed & causal_allowed
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(result.transpose(1, 2).reshape(batch, length, -1))


class TransformerBlock(nn.Module):
    def __init__(self, width, attention_width, heads, feedforward_width, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = NarrowSelfAttention(width, attention_width, heads, dropout)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, *, key_valid=None, causal=False):
        x = x + self.residual_dropout(self.attention(
            self.attention_norm(x), key_valid=key_valid, causal=causal))
        return x + self.residual_dropout(self.feedforward(self.feedforward_norm(x)))


class DescriptionEncoder(nn.Module):
    """Causal 128-D encoder returning CARD's state immediately after real tokens."""

    def __init__(self, config: DescriptionConfig | None = None):
        super().__init__()
        self.config = config or DescriptionConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocabulary_size, c.width, padding_idx=c.pad_id)
        self.position_embedding = nn.Embedding(c.max_length, c.width)
        self.blocks = nn.ModuleList([
            TransformerBlock(c.width, c.width, c.heads, c.feedforward_width, c.dropout)
            for _ in range(c.layers)
        ])

    def forward(self, token_ids: Tensor) -> Tensor:
        _long_ids(token_ids, "token_ids")
        c = self.config
        if token_ids.ndim != 2 or token_ids.shape[0] < 1 or not 2 <= token_ids.shape[1] <= c.max_length:
            raise ValueError(f"token_ids must be [batch, length] with length 2..{c.max_length}")
        _range(token_ids, "token_ids", 0, c.vocabulary_size - 1)
        valid = token_ids.ne(c.pad_id)
        lengths = valid.sum(-1)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        if bool((lengths < 2).any()) or not torch.equal(valid, positions[None, :] < lengths[:, None]):
            raise ValueError("Descriptions need at least BOS/CARD and only trailing padding")
        if not bool(token_ids[:, 0].eq(c.bos_id).all()) or not bool(token_ids.eq(c.bos_id).sum(-1).eq(1).all()):
            raise ValueError("Each description must contain BOS exactly once at its beginning")
        card_positions = lengths - 1
        if not bool(token_ids.eq(c.card_id).sum(-1).eq(1).all()) or not bool(
                token_ids.gather(1, card_positions[:, None]).eq(c.card_id).all()):
            raise ValueError("CARD must occur exactly once, immediately before padding")
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x, key_valid=valid, causal=True)
        return x[torch.arange(x.shape[0], device=x.device), card_positions]


class DescriptionModel(nn.Module):
    """Description encoder plus LayerNorm -> Linear -> GELU -> Linear projection."""

    def __init__(self, config: DescriptionConfig | None = None):
        super().__init__()
        self.config = config or DescriptionConfig()
        c = self.config
        self.encoder = DescriptionEncoder(c)
        self.projection = nn.Sequential(
            nn.LayerNorm(c.width), nn.Linear(c.width, c.projection_hidden_width),
            nn.GELU(), nn.Linear(c.projection_hidden_width, c.output_width),
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.projection(self.encoder(token_ids))


class StrategicModel(nn.Module):
    """Ordered battle transformer. Card representations stay differentiable.

    Shapes: cards [B, 2, 4, width], borders/packs [B, 2, 4], each support
    namespace [B, 2], support_tiers [B, 2, 2] (side, red/blue; 1..5, 0 unknown),
    classes [B, 2, 4, class_count]. Side 0 is ally/A.
    Border IDs are 1..16. Support IDs are 1..28/15; 0 means absent.
    Pack IDs are 1..14; 0 means unknown. Visibility has True for observed slots.
    Hidden identities, including their metadata, are ignored before lookup.
    card_stats [B, 2, 4, 2] are each card's (HP, ATK) as it enters the battle; only their
    ratios across all visible cards of both teams reach the model (see `normalized_stats`).
    """

    def __init__(self, config: StrategicConfig | None = None):
        super().__init__()
        self.config = config or StrategicConfig()
        c = self.config
        self.border_embedding = nn.Embedding(16, c.width)
        self.mutation_embedding = (nn.Embedding(len(c.mutation_names), c.width, padding_idx=0)
                                   if c.mutation_names and c.mutation_embedding else None)
        self.red_support_embedding = nn.Embedding(28, c.width)
        self.blue_support_embedding = nn.Embedding(15, c.width)
        # Support border quality: tier 1 base .. 5 Galaxy, one table per colour (0 = unknown/absent).
        self.support_tier_embedding = nn.Embedding(2 * 5, c.width)
        self.pack_embedding = nn.Embedding(14, c.width) if c.pack_embedding else None
        self.class_embedding = nn.Embedding(len(c.class_names), c.width) if c.class_names else None
        # Row 0 means "no identity" (hidden or unknown). Zero-initialised: a new card starts
        # from its description alone and learns card-specific interactions over training.
        self.identity_embedding = (nn.Embedding(c.identity_capacity + 1, c.width, padding_idx=0)
                                   if c.identity_capacity else None)
        if self.identity_embedding is not None:
            nn.init.zeros_(self.identity_embedding.weight)
        self.position_embedding = nn.Embedding(len(SLOT_NAMES), c.width)
        self.mode_embedding = nn.Embedding(2, c.width)
        self.bos = nn.Parameter(torch.empty(c.width))
        self.predict = nn.Parameter(torch.empty(c.width))
        self.hidden_card = nn.Parameter(torch.empty(c.width))
        self.hidden_support = nn.Parameter(torch.empty(2, c.width))
        for value in (self.bos, self.predict, self.hidden_card, self.hidden_support):
            nn.init.normal_(value, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(c.width, c.attention_width, c.heads, c.feedforward_width, c.dropout)
            for _ in range(c.layers)
        ])
        self.outcome_head = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, len(OUTCOME_NAMES)))
        # Residual stat MLP over every card token (user): the normalized (HP, ATK) are projected to the token
        # width, and that projection skips around the MLP to its output; the sum is added to the card token.
        # The projection and the MLP's output layer start at zero, so adding them to a trained model leaves
        # its predictions unchanged until training puts the stats to use (the skip still passes gradient).
        # Registered last: these parameters come after all others (training.add_stat_mlp relies on this).
        stat_width = c.stat_width or c.width
        self.stat_projection = nn.Linear(2, stat_width)
        self.stat_mlp = nn.Sequential(nn.Linear(stat_width, c.stat_hidden_width), nn.GELU(),
                                      nn.Linear(c.stat_hidden_width, stat_width))
        if c.stat_width:  # stats in their own channels: the card is projected onto the others, each part normalised
            self.card_projection = nn.Linear(c.width, c.width - c.stat_width)
            self.card_norm = nn.LayerNorm(c.width - c.stat_width)
            self.stat_norm = nn.LayerNorm(c.stat_width)
        elif c.stat_tokens:  # a stat token per card: its own positions, and a stand-in for unseen cards' stats
            self.stat_position_embedding = nn.Embedding(8, c.width)
            self.hidden_stat = nn.Parameter(torch.empty(c.width))
            nn.init.normal_(self.hidden_stat, std=0.02)
        else:  # added to the card: starting at zero leaves a trained model unchanged (add_stat_mlp)
            for layer in (self.stat_projection, self.stat_mlp[-1]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def stat_vectors(self, normalized: Tensor) -> Tensor:
        projected = self.stat_projection(normalized)
        return projected + self.stat_mlp(projected)

    def with_stats(self, cards: Tensor, normalized: Tensor) -> tuple[Tensor, Tensor | None]:
        """(card tokens, stat tokens or None) from card tokens [..., W] and their normalized stats [..., 2]: stats as
        tokens of their own (stat_tokens), in the card token's own channels (stat_width), or added to it."""
        if self.config.stat_tokens:
            return cards, self.stat_vectors(normalized)
        if self.config.stat_width:
            return torch.cat([self.card_norm(self.card_projection(cards)), self.stat_norm(self.stat_vectors(normalized))], -1), None
        return cards + self.stat_vectors(normalized), None

    @staticmethod
    def normalized_stats(card_stats: Tensor, visible: Tensor) -> Tensor:
        """log(HP), log(ATK) minus their shared mean over the visible cards of both teams.

        No interaction depends on absolute stat sizes (user), so only ratios remain. HP and ATK share one
        scale, which keeps HP-to-ATK ratios (hits to kill) while scaling every stat together changes nothing.
        """
        logs = card_stats.log()
        weight = visible[..., None].to(logs.dtype).expand_as(logs)
        mean = (logs * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3)).clamp_min(1)
        return torch.where(visible[..., None], logs - mean[:, None, None, None], 0.0)

    def _metadata_ids(self, ids, shape, name, visible, low, high, hidden_default):
        _long_ids(ids, name, shape)
        if ids.device != visible.device:
            raise ValueError(f"{name} must be on the input device")
        safe = torch.where(visible, ids, hidden_default)
        _range(safe, name, low, high)
        return safe

    def identity_penalty(self) -> Tensor:
        """Training-loss term: identity_l2 * sum of squared identity vectors (zero if disabled)."""
        if self.identity_embedding is None or not self.config.identity_l2:
            return self.bos.new_zeros(())
        return self.config.identity_l2 * self.identity_embedding.weight.square().sum()

    def card_part(self, card_embeddings: Tensor, pack_ids: Tensor | None = None,
                  class_weights: Tensor | None = None, identity_keys: Tensor | None = None) -> Tensor:
        """The card-determined part of a card token: description + pack + classes + identity (ids already checked)."""
        cards = card_embeddings
        if pack_ids is not None and self.pack_embedding is not None:
            cards = cards + torch.where(pack_ids[..., None].gt(0), self.pack_embedding(pack_ids.clamp_min(1) - 1), 0.0)
        if class_weights is not None:
            cards = cards + class_weights @ self.class_embedding.weight
        if identity_keys is not None and self.identity_embedding is not None:
            identities = self.identity_embedding(identity_keys)
            if self.training and self.config.identity_dropout:
                keep = torch.rand(identity_keys.shape, device=identity_keys.device) >= self.config.identity_dropout
                identities = identities * keep[..., None].to(identities.dtype)
            cards = cards + identities
        return cards

    def mutation_vectors(self, mutation_ids: Tensor) -> Tensor:
        if self.mutation_embedding is None:
            return torch.zeros(*mutation_ids.shape, self.config.width, device=mutation_ids.device)
        return torch.where(mutation_ids[..., None].gt(0), self.mutation_embedding(mutation_ids), 0.0)

    def support_vectors(self, index: int, ids: Tensor, tiers: Tensor | None) -> Tensor:
        """Support tokens for one colour (0 red, 1 blue); id 0 = absent."""
        embedding = (self.red_support_embedding, self.blue_support_embedding)[index]
        values = embedding(ids.clamp_min(1) - 1)
        if tiers is not None:
            values = values + torch.where(tiers[..., None].gt(0), self.support_tier_embedding(index * 5 + tiers.clamp_min(1) - 1), 0.0)
        return torch.where(ids[..., None].gt(0), values, 0.0)

    def assemble(self, cards: Tensor, supports: list[Tensor], mode_ids: Tensor, stats: Tensor | None = None) -> Tensor:
        """The token sequence from final card tokens [B, 2, 4, W] and support tokens (red, blue) [B, 2, W]; stat
        tokens [B, 2, 4, W] (stat_tokens) go before MODE and PREDICT, each at its card's own stat position."""
        batch = cards.shape[0]
        sequence = torch.cat([
            self.bos.expand(batch, 1, -1),
            supports[0][:, 0:1], supports[1][:, 0:1], cards[:, 0],
            supports[0][:, 1:2], supports[1][:, 1:2], cards[:, 1],
            self.mode_embedding(mode_ids)[:, None], self.predict.expand(batch, 1, -1),
        ], dim=1)
        sequence = sequence + self.position_embedding(torch.arange(len(SLOT_NAMES), device=cards.device))[None]
        if stats is None:
            return sequence
        stats = stats.flatten(1, 2) + self.stat_position_embedding(torch.arange(8, device=cards.device))[None]
        return torch.cat([sequence[:, :-2], stats, sequence[:, -2:]], dim=1)

    def outcome(self, sequence: Tensor) -> Tensor:
        """Logits (A win, B win) for an assembled sequence."""
        for block in self.blocks:
            sequence = block(sequence)
        return self.outcome_head(sequence[:, -1])

    def build_sequence(self, card_embeddings: Tensor, border_ids: Tensor,
                       red_support_ids: Tensor, blue_support_ids: Tensor, *,
                       pack_ids: Tensor | None = None, class_weights: Tensor | None = None,
                       mutation_ids: Tensor | None = None, identity_keys: Tensor | None = None,
                       card_visible: Tensor | None = None, support_visible: Tensor | None = None,
                       mode_ids: Tensor | None = None, support_tiers: Tensor | None = None,
                       card_stats: Tensor | None = None) -> Tensor:
        c = self.config
        if not isinstance(card_embeddings, Tensor) or not card_embeddings.is_floating_point():
            raise ValueError("card_embeddings must be floating-point")
        if card_embeddings.ndim != 4 or tuple(card_embeddings.shape[1:]) != (2, 4, c.width) or card_embeddings.shape[0] < 1:
            raise ValueError(f"card_embeddings must be [batch, 2, 4, {c.width}]")
        if card_embeddings.device != self.bos.device or card_embeddings.dtype != self.bos.dtype:
            raise ValueError("card_embeddings must have the same device and dtype as the model")
        batch, device = card_embeddings.shape[0], card_embeddings.device
        cv = _visibility(card_visible, (batch, 2, 4), device, "card_visible")
        sv = _visibility(support_visible, (batch, 2, 2), device, "support_visible")
        cards = torch.where(cv[..., None], card_embeddings, 0.0)
        if not bool(torch.isfinite(cards).all()):
            raise ValueError("Visible card embeddings must be finite")
        border_ids = self._metadata_ids(border_ids, (batch, 2, 4), "border_ids", cv, 1, 16, 1)
        cards = cards + self.border_embedding(border_ids - 1)
        if mutation_ids is not None:
            mutation_ids = self._metadata_ids(mutation_ids, (batch, 2, 4), "mutation_ids", cv,
                                               0, max(0, len(c.mutation_names)-1), 0)
            cards = cards + self.mutation_vectors(mutation_ids)
        if pack_ids is not None:
            pack_ids = self._metadata_ids(pack_ids, (batch, 2, 4), "pack_ids", cv, 0, 14, 0)
        if class_weights is not None:
            expected = (batch, 2, 4, len(c.class_names))
            if self.class_embedding is None:
                raise ValueError("Class metadata is disabled; configure an explicit class vocabulary")
            if (not isinstance(class_weights, Tensor) or tuple(class_weights.shape) != expected
                    or not class_weights.is_floating_point() or class_weights.device != device
                    or class_weights.dtype != cards.dtype):
                raise ValueError(f"class_weights must be floating-point {expected} on the model device/dtype")
            class_weights = torch.where(cv[..., None], class_weights, 0.0)
            if not bool(torch.isfinite(class_weights).all()) or bool((class_weights < 0).any()):
                raise ValueError("Visible class weights must be finite and nonnegative")
        if identity_keys is not None and self.identity_embedding is not None:
            identity_keys = self._metadata_ids(identity_keys, (batch, 2, 4), "identity_keys", cv, 0, c.identity_capacity, 0)
        else:
            identity_keys = None
        cards = self.card_part(cards, pack_ids, class_weights, identity_keys)
        stat_tokens = None
        if card_stats is None and (c.stat_width or c.stat_tokens):
            raise ValueError("A model with stat_width needs card_stats")
        if card_stats is not None:
            if (not isinstance(card_stats, Tensor) or tuple(card_stats.shape) != (batch, 2, 4, 2)
                    or card_stats.device != device or card_stats.dtype != cards.dtype):
                raise ValueError("card_stats must be [batch, 2, 4, 2] (HP, ATK) on the model device/dtype")
            if not bool((torch.where(cv[..., None], card_stats, 1.0) > 0).all()):
                raise ValueError("Visible card stats must be positive")
            normalized = self.normalized_stats(torch.where(cv[..., None], card_stats, 1.0), cv)
            cards, stat_tokens = self.with_stats(cards, normalized)
        # Replace the entire identity+border+pack+class representation, not only identity.
        cards = torch.where(cv[..., None], cards, self.hidden_card)
        if stat_tokens is not None:
            stat_tokens = torch.where(cv[..., None], stat_tokens, self.hidden_stat)
        supports = []
        for index, (ids, embedding, count, name) in enumerate((
                (red_support_ids, self.red_support_embedding, 28, "red_support_ids"),
                (blue_support_ids, self.blue_support_embedding, 15, "blue_support_ids"))):
            ids = self._metadata_ids(ids, (batch, 2), name, sv[:, :, index], 0, count, 0)
            tiers = (None if support_tiers is None else
                     self._metadata_ids(support_tiers[..., index], (batch, 2), "support_tiers", sv[:, :, index], 0, 5, 0))
            values = self.support_vectors(index, ids, tiers)
            supports.append(torch.where(sv[:, :, index, None], values, self.hidden_support[index]))
        if mode_ids is None:
            # Mode 0 = observed enemy; mode 1 = at least one hidden enemy slot.
            mode_ids = (~cv[:, 1]).any(-1).logical_or((~sv[:, 1]).any(-1)).long()
        _long_ids(mode_ids, "mode_ids", (batch,))
        if mode_ids.device != device:
            raise ValueError("mode_ids must be on the input device")
        _range(mode_ids, "mode_ids", 0, 1)
        return self.assemble(cards, supports, mode_ids, stat_tokens)

    def forward(self, card_embeddings: Tensor, border_ids: Tensor,
                red_support_ids: Tensor, blue_support_ids: Tensor, **metadata) -> Tensor:
        """Return logits ordered (A win, B win), not calibrated probabilities."""
        # Full strategic context is visible to PREDICT; lineup positions never move.
        return self.outcome(self.build_sequence(card_embeddings, border_ids, red_support_ids,
                                                blue_support_ids, **metadata))

    def probabilities(self, *args, **kwargs) -> Tensor:
        return self(*args, **kwargs).softmax(-1)


class BattleModel(nn.Module):
    """Compose description learning and strategy, or accept search-slot vectors."""

    def __init__(self, description_config: DescriptionConfig | None = None,
                 strategic_config: StrategicConfig | None = None):
        super().__init__()
        self.description_config = description_config or DescriptionConfig()
        self.strategic_config = strategic_config or StrategicConfig()
        if self.description_config.output_width != self.strategic_config.width:
            raise ValueError("Description output width must match strategic width")
        self.description = DescriptionModel(self.description_config)
        self.strategy = StrategicModel(self.strategic_config)

    def forward(self, *, border_ids, red_support_ids, blue_support_ids,
                description_tokens=None, card_embeddings=None, **metadata):
        if (description_tokens is None) == (card_embeddings is None):
            raise ValueError("Provide exactly one of description_tokens and card_embeddings")
        if description_tokens is not None:
            _long_ids(description_tokens, "description_tokens")
            if description_tokens.ndim != 4 or tuple(description_tokens.shape[1:3]) != (2, 4):
                raise ValueError("description_tokens must be [batch, 2, 4, length]")
            batch, _, _, length = description_tokens.shape
            if length < 2:
                raise ValueError("Descriptions require at least BOS and CARD")
            visible = _visibility(metadata.get("card_visible"), (batch, 2, 4),
                                  description_tokens.device, "card_visible")
            hidden_tokens = torch.zeros_like(description_tokens)
            hidden_tokens[..., 0] = self.description_config.bos_id
            hidden_tokens[..., 1] = self.description_config.card_id
            tokens = torch.where(visible[..., None], description_tokens, hidden_tokens)
            card_embeddings = self.description(tokens.reshape(-1, length)).reshape(
                batch, 2, 4, self.strategic_config.width)
        return self.strategy(card_embeddings, border_ids, red_support_ids,
                             blue_support_ids, **metadata)

    def probabilities(self, **kwargs):
        return self(**kwargs).softmax(-1)
