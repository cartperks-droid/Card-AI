# Card engine: project handoff

This project simulates a Roblox "Snap!"-style card game and trains an AI to predict and build winning teams. Rules come from the user's in-game observations: videos, screenshots and stated facts. **Every rule must be evidence-derived.** Ask the user rather than guess, and never mark a prediction as an observation.

## Layout
- **Battle engine: DaddyDrago's** (user, 2026-10-03: his engine, stats and abilities are more accurate; it replaced our own simulator). It is TypeScript from github.com/daddydrag0/CardRngExpansionDepths. His repo has no license file, but he gave the user permission to share it (user, 2026-10-03: "he said it was ok and he open sourced it for the reason of sharing it"). His TypeScript is still fetched rather than committed: `sim_js/setup.sh` clones it at the pinned `sim_js/ENGINE_COMMIT` into the gitignored `sim_js/vendor/`, patches it and installs `tsx`. It needs Node, git and a C compiler. Rerun it after changing `ENGINE_COMMIT`, `codemod.mjs`, `gen_tables.ts` or `sim_c/`.
  - `sim_js/codemod.mjs` turns each of his random draws into a chance point and fails loudly if any are left unconverted.
  - `sim_js/search.ts` is the best-first chance-tree search with pooled playouts. It also applies per-battle tweaks: fixed, scaled or ability-stripped cards.
  - `sim_js/worker.ts` answers one JSON request per line; `state` exports a battle's start for the C engine.
  - `sim_c/` is the C port of his battle code (`engine.c`) and of `search.ts` (`search.c`). It gives bit-identical answers, about 11× faster. `sim_js/gen_tables.ts` writes his card data into the gitignored `sim_c/build/tables.h`, and `setup.sh` compiles `sim_c/build/libcardsim`. Both are generated at setup, so they aren't committed. `sim_c/*.c` translates his code, with his permission (above).
  - `card_engine/simulator/kernel.py` loads that start state into C and runs the search. Labels, counters and verification use it. `tests/test_kernel.py` checks it against the TypeScript.
  - `card_engine/simulator/drago.py` maps our cards, borders, mutations and supports to his names (`CARD_ALIASES`) and runs one worker per process. Its functions are `evaluate`, `initial_stats` and `stat_tables`.
  - Support tiers: 1 base, 2 Platinum, 3 Crystal, 4 Ruby, 5 Galaxy. His engine lacks Ruby; `codemod.mjs` adds it in `auras.label.ts`. Red supports use a rarity ×500, fitted to the user's binder (IMG_0350–0354) and confirmed by the user (The One Ring 256%, Dinosaur King 207%/334%); blue supports use their base values. Astraeus: each art is its own deterministic card (user), so every Astraeus has an art (`arts` 1–7, `"Astraeus+Virgo"`). `search.ts` presets its Constellar ability; his engine would otherwise draw one, like Glamour, which distorts the card.
- `card_engine/data_corrections.py` and `card_engine/import_snap.py` hold the user's corrections: classes, weathers, border rarities. Re-import with `python -m card_engine.import_snap --raw-dir data/raw --output-dir data/clean`.
- `card_engine/training/` holds the AI pipeline:
  - `labels.py`: battle labels from the best-first search over his engine.
  - `train.py`: the classifier trainer.
  - `counter.py`: counter-team search.
  - `tablebase.py`: stores exact outcomes.
  - `predict.py`: the classifier's win rate for named teams, both turn orders (`--simulate` adds the engine).
  - `generate.py`: the generator (design below): counters to a named enemy or to broadly generated enemies, verified by the engine.
- `card_engine/teams.py`: teams by name: `"Name[@Border][/Mutation]"` cards, `"Name[@Tier]"` supports.
- `card_engine/model/` has two transformers: a description (language) encoder and a strategic encoder. The head gives **two outcomes, A win and B win. There are no ties**: a draw (both sides wiped out, or his 2,000-turn cap) counts as the attacker A's loss.
- Player tools:
  - `card_engine/deck.py`: the user's own collection (`data/my_deck.json`), and with `--custom` the custom pool (`data/custom_pool.json`) for suggestions tailored to one collection.
  - `card_engine/restricted.py`: player-base availability (`data/restricted_deck.json`).
  - `card_engine/ownership.py`: team-ownership likelihood (copy counts, a 2-D independence parameter, a progression mixture).
  - `card_engine/availability.py`: weather-availability rules.
  - `card_engine/class_report.py`: class membership report.
- Docs:
  - `docs/simulator_questions.md`: the rule log.
  - `docs/daily_checks/queue.md`: open in-game checks.
  - `docs/training.md`: the training pipeline.
  - `docs/runpod.md`: the pod run, Mac labelling and syncing, progress checks, the stop timer. Read it first when resuming the run.
  - `docs/classes.md`: class memberships.
  - `docs/card_forms.md`: card forms.

## Commands
```sh
bash sim_js/setup.sh                                                                           # fetch and patch DaddyDrago's engine, build the C engine
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
- **Generator** (`training/generate.py`; built 2026-10-03, its results are only as good as the classifier):
  - It starts from noise in the slot space and runs gradient ascent on the frozen classifier toward a "win = 1" target. Weights never change.
  - One role per run (user, 2026-10-04): `--role attack` (default) or `--role defend`. Attacking and defending are separate situations, never averaged.
  - Card, border and mutation are decoded by an exact factorised nearest-k match, filtered to a pool: own deck, custom pool, restricted deck, or everything.
  - Masks (user, 2026-10-04): borderless, unmutated and base-tier supports by default, plus an optional minimum card rarity. Each is widened on request. They narrow the open pools only; cards put in a deck override them.
  - A commitment penalty and an entropy check stop slots blurring between several cards.
  - Enemies are generated broadly (high temperature); 32 counters are generated tightly; the simulator verifies every team.
- **Label storage:** every shard lives in `data/labels/store/`, stamped with a rules snapshot ID (`training/flags.py`).
  - **Automatic invalidation:** each snapshot records per-entity fingerprints, taken from his data and code: cards, supports at each tier, borders, mutations, aura logic, the random-ability pool, and the engine core. When an entity changes, only the rows involving it drop out.
  - **Core changes:** an engine-code change must be declared with `python -m card_engine.training.flags declare --cards ... | --all | --none --note "..."`. Until then, older labels are held back. Declarations chain, so each engine version is declared once, from the version before it (the default). `flags status` shows the state; the trainer keeps its loaded rows while everything is held back.
  - **Storage:** nothing extra is stored per row.

## Working with the user
- Fast, large-batch progress. Resolve every interaction rather than refusing it.
- Read the user's videos with frame OCR (macOS Vision); don't guess.
- Don't confuse game display bugs with game logic. Known bug: when cards die in the deck, the shown lineup shifts.
- No legacy clutter or compatibility shims.
- The user owns no Limited or seasonal cards.
