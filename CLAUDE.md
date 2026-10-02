# Card engine: project handoff

This project simulates a Roblox "Snap!"-style card game and trains an AI to predict and build winning teams. Rules come from the user's in-game observations: videos, screenshots and stated facts. **Every rule must be evidence-derived.** Ask the user rather than guess, and never mark a prediction as an observation.

## Layout
- `card_engine/simulator/reference.py` is the Python reference engine. `sim/card_sim.c` and `sim/card_sim.h` are the C kernel (called through `card_engine/simulator/native.py`). **The two must agree exactly**: parity tests compare their outcome probabilities and state counts. Bump the kernel version string in both `card_sim.c` and `native.py` whenever the C code changes.
- `card_engine/simulator/catalog_rules.py` maps all 289 cards to rule fields and holds the support tables (`RED_SUPPORTS`, `BLUE_SUPPORTS`, tiers 1 base to 5 Galaxy).
- `card_engine/data_corrections.py` and `card_engine/import_snap.py` hold the user's corrections: classes, weathers, border rarities. Re-import with `python -m card_engine.import_snap --raw-dir data/raw --output-dir data/clean`.
- `card_engine/training/` holds the AI pipeline:
  - `labels.py`: battle labels from the C kernel's best-first search.
  - `train.py`: the classifier trainer.
  - `counter.py`: counter-team search.
  - `tablebase.py`: stores exact outcomes.
  - `predict.py`: compares the model with the simulator on one matchup.
- `card_engine/model/` has two transformers: a description (language) encoder and a strategic encoder. The head gives **two outcomes, A win and B win. There are no ties**: the attacker A loses if both sides are wiped out.
- Player tools:
  - `card_engine/deck.py`: the user's own collection (`data/my_deck.json`).
  - `card_engine/restricted.py`: player-base availability (`data/restricted_deck.json`).
  - `card_engine/ownership.py`: team-ownership likelihood (copy counts, a 2-D independence parameter, a progression mixture).
  - `card_engine/availability.py`: weather-availability rules.
  - `card_engine/class_report.py`: class membership report.
- Docs:
  - `docs/simulator_questions.md`: the rule log.
  - `docs/daily_checks/queue.md`: open in-game checks.
  - `docs/training.md`: the training pipeline.
  - `docs/classes.md`: class memberships.
  - `docs/card_forms.md`: card forms.

## Commands
```sh
python -c 'from card_engine.simulator.native import build_library; print(build_library())'   # build the C kernel first
python -m unittest $(ls tests/test_*.py | sed 's#/#.#;s#\.py$##')                            # all tests
python -m card_engine.training.labels --shards 1000 --rows 2000 --workers N                    # generate labels
python -m card_engine.training.train --batch-size 512 --weight-decay 0.05 --dropout 0.1 --freeze-language-at 130000
```

## Training design (agreed with the user)
- **Data:** the classifier trains on the *theoretical* space: all cards, borders, mutations, supports and tiers, with side A always attacking first. Ownership and availability models are never used in training.
- **Labels:** each battle is searched best-first in branch mode. The largest probabilities are expanded first, under a node budget. Chance rolls with tiny branches are sampled. Leftover branches are resolved by pooled playouts, as many as reducing the variance needs. Exact results go into the tablebase; estimated ones are never stored and are recomputed each time.
- **Targets:** the estimated win probabilities, as soft labels. Metrics:
  - KL over the irreducible entropy;
  - accuracy on deterministic battles;
  - error on random battles' win chance;
  - accuracy on "upsets", where abilities overturn the stat favourite.
- **Regularisation:** dropout only at the GELU upscale. The language transformer freezes at a set step.
- **Grokking:** probes (fixed training and validation subsets, plus the weight norm) are logged. The dataset keeps growing.
- **Next phase, only once the classifier is excellent:** a generator.
  - It starts from noise in the slot space and runs gradient ascent on the frozen classifier toward a "win = 1" target. Weights never change.
  - Card, border and mutation are decoded by an exact factorised nearest-k match, filtered to a pool: own deck, restricted deck, or everything.
  - A commitment penalty and an entropy check stop slots blurring between several cards.
  - Enemies are generated broadly (high temperature); 32 counters are generated tightly; the simulator verifies every team.
- **Label location:** labels live in `data/labels/<kernel version>_<rules hash>/`. A rule change starts a new directory automatically.

## Working with the user
- Fast, large-batch progress. Resolve every interaction rather than refusing it.
- Read the user's videos with frame OCR (macOS Vision); don't guess.
- Don't confuse game display bugs with game logic. Known bug: when cards die in the deck, the shown lineup shifts.
- No legacy clutter or compatibility shims.
- The user owns no Limited or seasonal cards.
