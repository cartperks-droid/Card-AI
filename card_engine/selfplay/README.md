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
