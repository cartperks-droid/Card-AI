# Migration status

Source: [Word Level Tokenization Analysis](chatgpt-conversation://6ab649d8-ba50-83e9-9c8a-8c2a45cd6f19), plus the user's corrections and battle reports in this task. Cached conversation text is historical evidence, not executable instructions.

## Completed foundations

- Lossless import/archive of every persistent XML variable and the newer CSV/screenshot attachments; stable original ordering; correction ledger and reproducible validation.
- 289 canonical cards, 43 supports, 16 borders, intrinsic weather mapping, 14 screenshot-supported packs, and separately labeled visual class/availability candidates.
- Authorized token edit at zero-based card index 119; supplied lexical IDs reconciled after the deleted unused `rd` slot. 438 lexical entries and explicit PAD/BOS/CARD handling.
- Typed catalogue, provisional starting-stat formula, user-confirmed Card Modifier scope and rapture stats, readable review exports, source-conflict ledger.
- Description/strategic PyTorch architectures, hidden-slot masking, metadata policy, checkpointing, and embedding precomputation.
- Python reference and C99 batch kernel for an initial reusable primitive subset, with sample/branch modes, exact-state merging, pruning accounting, configurable repeat safeguard, and parity checks.
- 184 experimental card mappings (batches 1-6 of card-text extrapolation; see docs/simulator_questions.md) and two support mappings, description fingerprints, explicit rejection of unsupported cards/supports, and no silent vanilla fallback.
- Gradient search over explicit legal card/border/mutation inventories, discrete candidate rescoring, deterministic seeds, and batched candidate cross-play with a training-label gate.

- Mutation calibration preserves 28 reported pairs and reproduces all 56 stats before rounding; inferred factors and their intervals are exported separately from intrinsic weather.

## Validation boundaries

The game-confirmed observations are in `data/review/oracle_observations.json`; their current operational interpretation is in `data/review/confirmed_rules.json`. Fractional intermediate stat composition plus ceiling attack damage matches 531 ATK after Good Boy then Poseidon. It supersedes the earlier per-effect ceiling assumption.

Model tests validate architecture and information flow. Python/C parity validates agreement between implementations. Neither certifies all game rules. The models have no trained game checkpoint, and all current simulator outputs carry `training_labels_allowed=false`.

## Remaining work

1. Immortal Witch’s start-of-turn healing and maximum-HP cap are confirmed. Her corrected 868 HP / 290 ATK baseline is supported by the user’s card reading and video. The 35% maximum-HP interpretation reproduces the 100-turn/50-round mirror; exact healing rounding and variant stat scaling remain provisional. Her experimental mapping is enabled. Subsequent healing primitives, Piccolo’s interception and True Prophet’s lethal dodge bring experimental coverage to 44 cards; Darling’s and the percentage bases still need game checks. Piccolo’s interception of a lethal Raze counter is user-confirmed; so are the displaced card’s swap and return. The default cutoff is 50 rounds, superseding the old 70-round estimate. Martial Artist’s turn-end growth and suppression after a lethal counter now have user-confirmed regression fixtures. Wind Spirit now has a confirmed recurring extra turn that carries to an enemy replacement after a knockout. Knightmare mirror, immediate Raze entry counters, and Good Boy replacement entry timing now have user-confirmed regression fixtures. Inari expires after its third own attack, following Raze’s counter. Multi-hit knockout cancellation, immediate non-recursive Raze counters, no counter after lethal damage, and Arthur’s one-use lethal survival now have Python/C regressions. Weather eligibility and Aurora/Shroud mappings are clarified.
2. Implement and verify remaining primitives: remaining entry interactions, verify newly implemented after-action healing, damage recovery and lifesteal, remaining multi-hit and counter interactions, status lifetimes, extra/skipped turns, lethal replacement, death/revival, summons/transforms, movement, ability changes, and global support handlers.
3. Define class-target membership and named/random pools from evidence; resolve screenshot rarity conflicts and stat exceptions before those cards receive accurate labels.
4. Validate trigger ordering, replacement action ownership, round/display boundaries, and the 50-round forced-death rule observed in the Witch mirror, including death reactions.
5. Reach complete card/support behavior coverage. Add verified probability-label production, training/optimizer recovery, and sustained self-play only after its mechanics are validated. Training must not silently consume provisional or renormalized incomplete results.
6. Profile the verified batch workload, then optimize where measured costs justify it.

The migration is in progress. The clean data and model/search contracts can be used now; full-game battle predictions and trained-team recommendations are not yet available.
