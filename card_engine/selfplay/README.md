# Candidate search and diagnostic cross-play

`generate_candidates` differentiates with respect to four card-slot vectors,
then samples nearest legal card/border/mutation variants and rescores each discrete team.
The model's weights, gradients, modes, and global random state are preserved.
Independent local seeds make candidate prefixes reproducible. Returned scores
are uncalibrated model estimates, including when using a saved checkpoint.

The caller supplies a `LegalInventory` containing one choice pool per slot and
an explicit duplicate policy (`allow`, `unique_card`, or `unique_variant`). This
does not infer ownership from screenshots or enforce quantities, support
ownership, or game restrictions beyond those pools and that policy. Selection
checks that each choice leaves a feasible completion of the four-slot lineup.

The relaxation includes description, border, mutation, pack, and any explicitly selected
class embeddings. Supports and the opposing team stay fixed. Hidden-information
search and support optimization are not yet exposed. Mutation selections require
an explicit intrinsic-weather eligibility table; weather cards cannot be mutated. Current
search diagnostics assume side A acts first; the model has no initiative input.
No mutation is silently assigned to a searched card.

`run_diagnostic_cycle` uses the same model for both sides with distinct seeds,
then sends the Cartesian product to `evaluate_crossplay`. C performs the batch
in one native call. Unsupported inventory abilities fail before search begins.
The default search API generates 32 candidates per side; the CLI uses two for
a quick wiring check with the full intended model architectures:

```sh
python3 -m card_engine.selfplay --candidates 2 --steps 2 --include-mutations --output data/review/selfplay_smoke.json
```

Build the C kernel first as described in `sim/README.md`, or pass
`--backend python`. The CLI initializes random weights and uses an explicit
fixture inventory; its output is an integration check, not trained team advice.

Cross-play returns A-win, B-win, tie, and unresolved mass for every ordered
pair. Branch-mode win bounds range from known win mass to known win mass plus
unresolved mass. They describe uncertainty within the provisional simulator,
not confidence that its rules match the game. A single sample trajectory cannot
provide probability bounds, so the bounds API rejects sample mode. Current
experimental results cannot become training rows. Future validated labels must
also account for all probability without dropping or renormalizing unresolved
mass.
