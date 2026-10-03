# Card engine

The Snap! project is organized into a reproducible dataset, PyTorch models, and a label search around DaddyDrago's battle engine. The cleaned catalogue contains **289 cards, 28 Red supports, 15 Blue supports, 16 borders, and 14 packs**, in the original index order. His engine implements every one of them.

## Inspect and reproduce

Run from this directory:

```sh
python3 -m card_engine --card 33 --border 1
python3 -m card_engine.import_snap --raw-dir data/raw --output-dir data/clean
python3 -m card_engine.reports
python3 -m card_engine.mutations
python3 -m unittest discover -s tests -v
```

Data tools use the Python standard library. Model and training tests additionally require PyTorch (`pip install -e '.[model]'`). Battle labels need Node and `bash sim_js/setup.sh`.

Useful outputs:

- `data/review/catalog_overview.csv`: every card, its pack, inferred classes, availability evidence, description, and provisional borderless stats.
- `data/review/mutation_fit.csv`: all 28 supplied mutation snapshots, fitted factors, and rounding bounds.
- `data/review/engine_mapping.csv`: every card and support, its name in DaddyDrago's engine, and whether his engine implements it.
- `data/review/source_conflicts.json`: differences between screenshot rarity labels and saved values, without silently changing either source.
- `data/clean/dataset.json`: canonical machine-readable data.
- `data/clean/archive/`: lossless persistent-variable and CSV evidence.
- `data/clean/manifest.json`, `corrections.json`, and `validation.json`: provenance and import checks.

Raw XML, supplied CSVs, and screenshots are preserved in `data/raw/`. New files whose names collide with older attachments are under `data/raw/user_2026-09-27/`. All source IDs remain one-based; card position is never alphabetical. Red/Blue support IDs are separate namespaces. Border 1 means no border.

## Applied corrections

**Card Modifier** scales both HP and attack. Both rapture cards have the confirmed borderless values **2560 HP / 1280 ATK**. Weather multipliers describe intrinsic card stats; border rarity contributes to effective rarity. The generic starting-stat formula remains subject to per-card exceptions.

Later user corrections are reapplied on every import with per-field provenance: Bad Boys is Base and Pandora is Eclipse; all eight Bosses cards have a 2× Card Modifier. Toy Bear remains Snow at rarity 400 million, and its reported 1943 ATK / 3886 HP is fitted with a separate ⅔ Card Modifier. Jason’s 4788 ATK / 9575 HP is fitted with a 2× modifier. The causes of those two fitted factors and their variant scaling remain unverified. Santa Claus (Aurora), Wandering Snowman and Santa Claws (Snow), The Blood Countess (Blood Rain), and Heaven’s Armor (Rapture) retain their confirmed mappings. Raw inputs remain unchanged.

The canonical token data uses the supplied untampered matrix and ordered vocabulary. Demon Cultivator, zero-based card index **119**, says **“awaken after 2 turns”**. Removing the unused `rd` slot shifts later token IDs down by one; other lexical IDs and vocabulary order are preserved. There are **438 lexical tokens**, PAD=0, BOS=439, CARD=440. CARD follows the last real token and precedes padding. Source text and original token files remain archived.

Only the authorized trailing weather `1` and proven duplicate trailing card/Blue-support descriptions are omitted from normalized data. Visual class assignments and lit-card availability remain labeled as inferences. Packs follow the supplied screenshot order. Inferred classes are excluded from neural inputs by default and are not simulator class rules.

## Battle engine

Battles run on DaddyDrago's engine (github.com/daddydrag0/CardRngExpansionDepths, user decision 2026-10-03). It has no license, so it is never copied here. `bash sim_js/setup.sh` clones it at the pinned commit into the gitignored `sim_js/vendor/`. It then patches every random draw into a chance point (`sim_js/codemod.mjs`). `sim_js/search.ts` searches the chance tree best-first and resolves the rest with pooled playouts. `card_engine/simulator/drago.py` translates our cards, borders, mutations and supports into his loadouts. A draw counts as the attacker's loss. Our own Python/C simulator was deleted when his engine replaced it; the rule observations behind it stay in `docs/simulator_questions.md`.

## Mutations

The 28 supplied Black Cat, Dancer, and Arthur snapshots all match a mutation factor applied before ceiling the base stats. Known candidate factors are None 1, Storm 1.1, Snow 1.2, Aurora 1.3, Shroud 1.5, Meteor Shower 1.8, Time Storm 2, Eclipse 2.5, Virus 3, Blood Rain 3.5, Armageddon 4, and Manga 4.5. Rounded observations bound these factors rather than proving unique exact values. Original intrinsic weather multipliers remain separate.

```sh
python3 -m card_engine --card 69 --mutation Storm
```

Weather cards retain their separate intrinsic multiplier and are ineligible for mutations. They receive no mutation token. Illegal combinations fail explicitly. The user confirmed the Aurora/Shroud mappings are accurate, so original weather IDs remain unchanged. See `docs/mutations.md` for the fit and exact observations.

## Models and search

The description model is 128-wide with two layers/four heads. The strategic model is 768-wide with four layers/eight heads and 192-wide attention. It has exactly 16 border embeddings, separate support embeddings, fixed ordered team slots, and whole-slot hidden-information masking. Checkpointing and detached embedding precomputation are available in `card_engine.model`.

`card_engine.selfplay` holds the prototype of gradient-based candidate generation against an explicit legal inventory. Its scores are model estimates; the training pipeline is described in `docs/training.md`.
