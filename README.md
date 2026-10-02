# Card engine

The Snap! project is now organized into a reproducible dataset, PyTorch models, and a Python/C battle prototype. The cleaned catalogue contains **289 cards, 28 Red supports, 15 Blue supports, 16 borders, and 14 packs**, in the original index order. The simulator currently compiles **184 card abilities** in an experimental subset; the remaining 105 fail explicitly. Training on game outcomes is still gated on rule validation.

## Inspect and reproduce

Run from this directory:

```sh
python3 -m card_engine --card 33 --border 1
python3 -m card_engine.import_snap --raw-dir data/raw --output-dir data/clean
python3 -m card_engine.reports
python3 -m card_engine.mutations
python3 -m unittest discover -s tests -v
```

Data tools use the Python standard library. Model/search tests additionally require PyTorch (`pip install -e '.[model]'`); native tests require a C compiler. Installing dependencies is unnecessary if they are already available.

Useful outputs:

- `data/review/catalog_overview.csv`: every card, its pack, inferred classes, availability evidence, description, and provisional borderless stats.
- `data/review/mutation_fit.csv`: all 28 supplied mutation snapshots, fitted factors, and rounding bounds.
- `data/review/simulator_coverage.csv`: execution status and reason for every card.
- `data/review/support_coverage.csv`: execution status for all 43 supports (2 experimental mappings).
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

## Battle prototype

```sh
python3 -c 'from card_engine.simulator.native import build_library; print(build_library())'
python3 -m card_engine.simulator --team-a 3,205 --team-b 44 --mode sample --trace
python3 -m card_engine.simulator --team-a 3,205 --team-b 44 --backend c
python3 -m card_engine.simulator --coverage
```

The two engines implement entry attacks, entry stat changes, enemy-entry theft, damage modifiers and caps, dodge, blocks, critical hits, multi-hit actions, immediate non-recursive counters, one-use lethal survival, invincibility, own-action lifetimes, recurring extra turns, turn-end attack growth, turn-start healing, kill healing, ordered replacements, and branch/sample execution. Dancer's first hit consumes Horus's block; hits two and three are calculated normally. A knockout cancels remaining hits. Wind Spirit’s separate extra turn recurs each cycle and can attack the next enemy after a knockout. Raze counters between surviving hits, but cannot counter a lethal hit or another counter. Arthur survives one lethal attack at 1 HP. Inari takes no damage and dies after its third own attack, after any counter finishes. Initial entry hits precede the initiator’s normal attack; replacement entry hits also resolve before that normal attack resumes. Martial Artist’s turn-end 30% attack growth reproduces 260 → 338 → 440 displayed ATK, including after Horus blocks; a lethal Raze counter prevents the boost. The battle initiator attacks first and loses if both teams are depleted. Branches merge only identical states. Pruned, capped, or unfinished probability stays in `unresolved`; it is never assigned to a winner.

Your observations establish Good Boy before Poseidon's self boost, Chronus theft after an enemy's self boost, and no theft when Chronus himself replaces Good Boy. Preserving intermediate fractional stats and rounding damage up reproduces the reported **531 ATK**; this is a tested interpretation, not a direct observation of the game's internal storage. Immortal Witch’s heal activates before her normal attack, including at full HP, and healing is capped at maximum HP. Her experimental mapping uses 35% maximum-HP healing and reproduces the recorded mirror. Her baseline is corrected to **868 HP / 290 ATK**, confirmed by the user’s card reading and video. Shu’s user-read **1230 HP / 362 ATK** is likewise fitted with separate HP/ATK ratios, because its weather multiplier is abnormal. Separate HP/ATK ratios are applied before rounding; border/mutation scaling remains extrapolated. The default stalemate safeguard is now 50 rounds (100 ordinary turns). General trigger order, replacement turn ownership, and repeat-cutoff details still need validation.

The full descriptions protect the experimental card mappings against stale data. Adventurer, Shielder and Guardian Angel are the implemented supports; other supports and unsupported abilities raise errors. The original 372-handler registry remains source evidence, not executable proof of game rules.

## Mutations

The 28 supplied Black Cat, Dancer, and Arthur snapshots all match a mutation factor applied before ceiling the base stats. Known candidate factors are None 1, Storm 1.1, Snow 1.2, Aurora 1.3, Shroud 1.5, Meteor Shower 1.8, Time Storm 2, Eclipse 2.5, Virus 3, Blood Rain 3.5, Armageddon 4, and Manga 4.5. Rounded observations bound these factors rather than proving unique exact values. Original intrinsic weather multipliers remain separate.

```sh
python3 -m card_engine --card 69 --mutation Storm
python3 -m card_engine.simulator --team-a 67 --team-b 73 --mutations-a Storm --mode sample --trace
```

Weather cards retain their separate intrinsic multiplier and are ineligible for mutations. They receive no mutation token. Illegal combinations fail explicitly. The user confirmed the Aurora/Shroud mappings are accurate, so original weather IDs remain unchanged. See `docs/mutations.md` for the fit and exact observations.

## Models and search

The description model is 128-wide with two layers/four heads. The strategic model is 768-wide with four layers/eight heads and 192-wide attention. It has exactly 16 border embeddings, separate support embeddings, fixed ordered team slots, and whole-slot hidden-information masking. Checkpointing and detached embedding precomputation are available in `card_engine.model`.

The CLI `python3 -m card_engine.selfplay --output data/review/selfplay_smoke.json` runs a small full-model wiring check with random weights. `card_engine.selfplay` adds gradient-based candidate generation against an explicit legal inventory and batched cross-play evaluation. Search scores are uncalibrated until trained. Search supports explicit mutation variants on eligible cards and assumes side A attacks first. Add `--include-mutations` to the smoke command to exercise those variants. Cross-play keeps unresolved probability visible and refuses current experimental results as training labels. See `card_engine/model/README.md`, `sim/README.md`, and `docs/migration_status.md` for contracts and remaining work.

Healing extensions cover Forest Spirit (after-attack healing before counters), Shu (damage recovery), Darling (entry stats plus turn-start healing), and Count Muscula (lifesteal). Their event timing and percentage bases remain experimental pending targeted game tests; Python/C parity alone does not verify them.

Forest Spirit and Count Muscula now have user-confirmed healing before Raze’s counter. This differs from Martial Artist’s turn-end growth after counters. Shu dies to a lethal Raze hit without recovering. Amount calculations and other healing interactions remain provisional.

Piccolo (#104) intercepts a lethal hit aimed at the card directly ahead of it, blocks it, and doubles its HP and ATK. The user confirmed this against Raze’s counter on Forest Spirit, followed by Raze’s normal attack and Piccolo’s kill. Team order is now a per-state permutation, and stats are stored per fighter. The user also confirmed that the displaced card swaps into Piccolo’s slot and returns when Piccolo dies. A later Piccolo directly behind it intercepts again. Arthur’s own 1-HP survival is used first. A returning card keeps its stats and repeats no entry ability (user-confirmed for Poseidon, Knightmare and Good Boy). Chronus steals from Piccolo when it swaps in, never from the same card twice. Protection order is True Prophet dodge, then survive at 1 HP, then Piccolo swap.

True Prophet (#164) has an on-death ability: the card that comes in next dodges its next hit. The card text says “lethal”, but the user saw a non-lethal first hit dodged in the mirror. Shu’s recovery still shows on that dodge, proportional to the dodged damage and capped at max HP. A Platinum Immortal Witch (3471 HP, border scaling confirmed) reproduces the user’s Prophet/Witch versus Raze battle.
