# PyTorch model layer

This package implements the intended architectures and input contracts. Weights
start randomly initialized. Its two outputs are **A win, B win** logits (there are no ties: the attacker A loses if both sides are wiped out);
softmax produces normalized scores, which are not calibrated battle probabilities
until trained against verified simulator results. No labels or training runs are
created by this package.

The description model uses width 128, two causal layers, four heads, a 512-wide
feedforward, and learned positions. Canonical sequences are
`BOS=439, real lexical tokens, CARD=440, PAD=0...`. CARD stays immediately after
the last real token. Its 128-dimensional state passes through
LayerNorm → Linear(768) → GELU → Linear(768).

The strategic model uses width 768, four layers, eight heads, attention width 192
(24 per head), and a 3072-wide feedforward. Its fixed 15-slot layout is:

```text
BOS
ally red support, ally blue support, ally cards 1–4
enemy red support, enemy blue support, enemy cards 1–4
MODE, PREDICT
```

The strategy transformer uses full attention across that fixed sequence. Card
order is never sorted or permuted. Border embeddings have exactly **16** rows;
Red and Blue supports use separate **28** and **15** row tables. Optional pack
metadata uses 14 rows. Source IDs remain one-based; zero support means absent,
and zero pack means unknown. The module checks tensor shapes, ranges, and masks,
but team legality and inventory ownership belong to the game layer.

```python
import torch
from card_engine.model import BattleModel, load_model_data

data = load_model_data()  # Uses actual cleaned descriptions; no class guesses.
model = BattleModel().eval()
cards = torch.tensor([[[3, 4, 5, 141], [3, 4, 5, 141]]])
inputs = data.gather(cards)
inputs.update(
    border_ids=torch.ones(1, 2, 4, dtype=torch.long),
    red_support_ids=torch.ones(1, 2, dtype=torch.long),
    blue_support_ids=torch.zeros(1, 2, dtype=torch.long),
)
with torch.no_grad():
    logits = model(**inputs)  # Shape [1, 3]; random, untrained scores.
```

`card_visible[B,2,4]` and `support_visible[B,2,2]` use True for known slots. Side
index 0 means ally/A; support index 0 is Red and 1 is Blue. A hidden card replaces
its entire identity, border, mutation, pack, and class representation with a learned hidden
token. Hidden support IDs are also ignored. Hidden content cannot influence the
representation or receive gradient. Positions remain present. The composed model
sanitizes hidden description tokens before encoding as well. Unless supplied,
MODE is 1 when any enemy slot is hidden and 0 otherwise.

Class embeddings are disabled by default. To experiment with them, supply the
same explicit ordered `class_names` tuple to `StrategicConfig` and
`load_model_data`. The default `verified_only` policy excludes current inferred
memberships; `wiki_supported` and `include_candidates` are explicit experimental
opt-ins. These policies affect neural metadata only and do not validate simulator
rules. Pass class weights as floating multi-hot or nonnegative weighted values.

For differentiable team search, call `StrategicModel` (or `BattleModel` with
`card_embeddings=`) using leaf tensors of shape `[B,2,4,768]` with
`requires_grad=True`. The output remains differentiable with respect to visible
slot representations. Discrete ID selection and legal-team search are outside
this package.

`precompute_embeddings(description_model, token_ids)` returns a detached CPU
snapshot, restoring the previous train/eval mode. It becomes stale whenever
description weights change. `save_checkpoint` and `load_checkpoint` preserve
architecture, model weights, mode, and optional JSON provenance; optimizer state,
training-loop recovery, and RNG state are not included. Record the dataset version
and selected class policy in checkpoint metadata when running experiments.

Mutation metadata uses an ordered 12-name vocabulary including neutral None.
`mutation_ids[B,2,4]` indexes that vocabulary; None adds no embedding. Intrinsic
weather remains part of card identity and never becomes a mutation token. The
game/search layers enforce eligibility. Checkpoint schema 2 pins this vocabulary;
legacy schema 1 models load without mutation parameters and reject mutated search.

## Where each input is aggregated

Two transformers, each with its own learned positional embedding.

**Language transformer** (`DescriptionEncoder`, 128-D, learned positions 0..63). Sequence: `BOS`, description word tokens, `CARD`. Only language lives here. The `CARD` token's final state is projected to 768-D and becomes that card's slot vector.

**Card engine transformer** (`StrategicModel`, 768-D, learned slot positions 0..14). Sequence: `BOS`, ally red support, ally blue support, ally cards 1-4, enemy red support, enemy blue support, enemy cards 1-4, `MODE`, `PREDICT`. Each card slot is the sum of its description vector and the engine-side embeddings: border, mutation, pack, class (off by default) and card identity. A hidden card's whole sum is replaced by the hidden-card vector.

**Card identity embedding.** Indexed by the permanent identity key (`card_engine/card_keys.py`, `data/annotations/card_keys.json`), never by list position, so a card inserted into a pack does not shift any other card. It has spare rows (`identity_capacity`, default 1024) for new cards, starts at zero so a new card begins from its description alone, is dropped at random during training (`identity_dropout`), and carries an L2 cost (`identity_l2`, added to the training loss through `StrategicModel.identity_penalty()`), so most understanding comes from the description while memorisation stays possible. It lets the model memorise card-specific interactions that the text cannot express.
