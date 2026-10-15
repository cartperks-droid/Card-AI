# Testing a new learning rule on the battle transformer

This is a brief for testing an alternative to backpropagation on this project's model. Paste it into the chat that
works on the learning rule. The bench script `scripts/learning_rule_bench.py` does the plumbing. A rule is one Python
file with one function in it.

## The task the model learns

The model predicts who wins a battle in a Roblox card game (a "Snap!"-style 4v4 auto-battler). Each side has:
- 4 cards in lineup order. Each card has a border (16 kinds) and a mutation (12 kinds, including none).
- a red support and a blue support, each with a tier from 1 to 5.

Side A always attacks first.

The labels come from an exact game engine, not from humans. Each battle's target is the engine's estimated pair of
probabilities **(P(A wins), P(B wins))**. Many battles are deterministic (a target of 1/0). Draws count as A losing, so
there are only two outcomes.

What makes it hard: the **upsets**, where the side with weaker raw stats wins because of card abilities. They are
about 15% of the current held-out battles; it depends on the mix. Picking the stronger side by stats alone gets about
85–91% right, so only the upsets show whether the model understands the abilities.

There are about 92 million labelled battles. The store keeps growing.

## Architecture (PyTorch, `card_engine/model/networks.py`)

The model, `BattleModel`, has two transformers.

1. **Description transformer** (`DescriptionModel`). It reads each card's ability text, tokenized with a 441-word
   vocabulary, and turns it into one vector per card. In training it runs once per step over all ~290 cards. The
   resulting card table is then indexed by the battle's cards.

2. **Strategic transformer** (`StrategicModel`). It reads one battle as 15 tokens:

   `BOS, A red, A blue, A card 1..4, B red, B blue, B card 1..4, MODE, PREDICT`

   - **Card tokens:** each card token is built by adding:
     - the card's description vector;
     - learned embeddings for border, mutation, pack, class and identity;
     - a residual stat MLP (GELU) over the card's HP and ATK compared with the other 7 cards (37 inputs).
   - **Other tokens:**
     - Support tokens are an embedding of the support plus one of its tier.
     - Every token also gets a learned position embedding.
   - **Blocks:** pre-norm transformer blocks: `x + Attn(LN(x))`, then `x + MLP(LN(x))`. The attention projects
     width → a narrower Q/K/V width (`attention_width`) and back. The MLP is width → 4×width → width with GELU.
   - **Mask:** none (no causal mask).
   - **Output head:** `LayerNorm → Linear(width, 2)` on the PREDICT token. This gives the logits (A win, B win).
   - **Stat prior:** the head's output is a *residual* on a fixed prior. The A-win log-odds is
     `bias + scale·tanh(slope·Δ/scale) + head residual`, where Δ is the log ratio of the two sides' total
     √(HP×ATK). The three prior numbers are fitted when a run starts and are not trained. The head starts at zero, so
     at initialisation the model predicts from the stats alone.

**The production run** (`data/training_10s`) has:
- a 10-layer strategic transformer, width 256, 8 heads, attention width 128, MLP width 1024;
- about 7.4M parameters in the strategic transformer, plus about 1M in the description transformer.

It trained to step ~885,000 with batch 512, using:
- **Optimizer:** AdamW (lr 3e-4, weight decay 0.05), gradient norm clipped at 1.0, 1,000 warmup steps. A cosine decay
  ran from step 330k to 370k, and the learning rate has held at 0.33× since.
- **Regularisation:** dropout 0.1 on the residual and MLP paths, plus an EMA of the weights.
- **Description transformer:** it was frozen and thawed in spells during training. It is frozen for good now.

## Loss and metrics

- **Loss** (soft-label cross-entropy): `-(target * logits.log_softmax(-1)).sum(-1).mean()`. The trainer also adds a
  small L2 penalty on the identity embeddings.
- **Metrics:** `card_engine.training.train.evaluate` reports these on held-out battles:
  - `kl`: the loss minus the targets' own entropy, i.e. the part the model could still remove. **This is the main
    number.**
  - `accuracy`, `decisive_accuracy` (deterministic battles only).
  - `upset_accuracy` and `upset_kl`: the battles abilities win. **This is the hardest part.**
  - `favourite_kl`: KL on the battles the stat favourite wins.
  - `probabilistic_error`: the mean |predicted − target| win chance on random battles.

**The production model at step ~884,000** (AdamW, full data) scores:

| KL | accuracy | upset accuracy | upset KL | probability error |
|---|---|---|---|---|
| ~0.05 | 0.966 | ~0.90 | ~0.13–0.15 | ~0.11 |

These are only a ceiling. A small bench run won't come near them. **Compare a new rule against `--rule backprop` on
the bench, with the same settings and seed.**

## The bench: `scripts/learning_rule_bench.py`

What it does:
- It reads a fixed subset of the label store (`--shards N`, about 2,000 battles each, plus a tenth as many held-out
  shards).
- It builds a model: a fresh small one (`--layers 4 --width 128` by default, about 0.9M parameters), or a trained one
  with `--from CHECKPOINT`.
- It runs `--steps` updates. Every rule sees **the same batches in the same order** (`--seed`).
- It evaluates every `--eval-every` steps and prints one JSON line per evaluation (`--out FILE` also appends them).
- It never writes to a run folder, so it can run beside the trainer. Use `--device cpu` then, because the trainer uses
  the GPU.

```
python3 scripts/learning_rule_bench.py --rule backprop --steps 3000 --out bench.jsonl
python3 scripts/learning_rule_bench.py --rule my_rule.py --lr 1e-3 --steps 3000 --out bench.jsonl
python3 scripts/learning_rule_bench.py --rule my_rule.py --layers 10 --width 256 --shards 400 --steps 20000
python3 scripts/learning_rule_bench.py --rule my_rule.py --from data/training_10s/ema.checkpoint --freeze-language
```

Options:
- `--freeze-language` keeps the description transformer fixed. Its card vectors are computed once, so only the
  strategic transformer learns. Use it to test the rule on the transformer blocks alone.
- `--kinds` chooses which shards to read: `shard` (random battles, the default with `fixed`), `fixed` (fixed-stat
  battles), and `hard`/`found`/`pvp` (the hard examples).
- `--rule-args ...` passes extra settings to the rule as `args.rule_args`.

## Writing a rule

A rule is a Python file that defines `make_rule(model, args)`. It returns an object with
`step(forward, batch) -> float`. That method updates `model`'s parameters in place and returns the batch loss.

The arguments:
- `forward(batch)` runs the full model (the description transformer unless frozen, then the strategic one) with
  autograd on. It returns logits of shape [B, 2]. Wrap it in `torch.no_grad()` for a rule that doesn't use autograd.
- `batch["target"]` is the [B, 2] soft target. The other keys are the battle's integer fields; the model's inputs are
  built from them inside `forward`.
- `loss(logits, batch)` is the trainer's loss. Get it with `from learning_rule_bench import loss`.

Parts of the model a rule may need:
- the blocks: `model.strategy.blocks` (a ModuleList of `TransformerBlock`, each with `.attention.qkv`,
  `.attention.output`, `.feedforward[0]`, `.feedforward[3]` and two LayerNorms);
- the head: `model.strategy.outcome_head`;
- the stat MLP: `model.strategy.stat_projection` and `model.strategy.stat_mlp`;
- the embeddings, and `model.description` (the description transformer).

Forward hooks on these modules give layer-local activations, for rules that learn from local signals.

Here is the reference update, as the trainer does it:

```python
import torch
from learning_rule_bench import loss

class Backprop:
    def __init__(self, model, args):
        self.model = model
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=args.lr, weight_decay=0.05)

    def step(self, forward, batch):
        value = loss(forward(batch), batch)
        self.optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(value.detach())

def make_rule(model, args):
    return Backprop(model, args)
```

## Things to know

- **Residual on a prior:** the output head starts at zero on top of the stat prior. Before any learning, the model is
  already at the stat baseline: the stat favourite's accuracy, with upset accuracy near 0. A rule earns its keep by raising
  `upset_accuracy` and lowering `upset_kl`, not plain accuracy.
- **Embeddings:** the identity embeddings start at zero. Lookup tables (border, support, identity and so on) only get
  a signal for the rows used in a batch.
- **Dropout:** the bench uses the model's default dropout (0 for a fresh model). The production run used 0.1.
- **Precision:** the production trainer can run in bf16 autocast; the bench runs in float32.
- **Batch sampling:** the production trainer draws set shares of each batch from the hard examples, upsets and
  fixed-stat battles (`--mix`). The bench samples uniformly from the shards it read. `--kinds hard found` makes it
  harder.
- **Rules of the data:** never change labels, targets or the loss to make a rule look better. The targets are the
  engine's answers.
