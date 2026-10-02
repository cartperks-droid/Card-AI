# Mutation stat observations

The user's 28 mutation snapshots cover Black Cat (#69, eight pairs), Dancer (#67, nine pairs), and Arthur (#10, eleven pairs). All **56 reported HP and attack values** agree with applying these candidate factors to the unrounded rarity formula, then rounding up once:

| Mutation | Candidate factor |
| --- | ---: |
| None | 1 |
| Storm | 1.1 |
| Snow | 1.2 |
| Aurora | 1.3 |
| Shroud | 1.5 |
| Meteor Shower | 1.8 |
| Time Storm | 2 |
| Eclipse | 2.5 |
| Virus | 3 |
| Blood Rain | 3.5 |
| Armageddon | 4 |
| Manga | 4.5 |

For this baseline-card fit, unrounded HP is `10 × 2^log10(rarity)` and unrounded attack is `5 × 2^log10(rarity)`. The mutation factor multiplies these values before the ceiling operation. Black Cat illustrates why the order matters: the formula predicts 156 Storm attack, as reported; multiplying its already-rounded 142 base attack by 1.1 and rounding up would give 157.

These are simple exact factors consistent with the rounded observations, not uniquely proven internal values. If `ceil(base × multiplier) = observed`, then `(observed − 1) / base < multiplier ≤ observed / base`. Intersecting those bounds across every reported HP and attack gives an allowed interval for each mutation. The generated CSV includes both individual and joint intervals, calculated with floating-point arithmetic.

The [annotation file](../data/annotations/mutations.json) preserves all pairs in the user's order, with attack and HP labeled explicitly. In particular, Dancer's Shroud entry precedes Aurora, while Arthur's Aurora entry precedes Shroud. It also retains the user's note about a manual Shroud/Aurora weather-order mistake. No weather indices or original values have been changed by this fit.

All three cards currently map to intrinsic Base weather and Card Modifier 1. The mutation report does not restate border or support conditions. The user subsequently clarified that weather cards retain their separate weather multiplier and are ineligible for mutations. Calling them mutated was only an analogy for that restriction, not a shared category. The stat API rejects all non-None mutation selections on weather cards, including a selection with the same weather name. The user also confirmed the Shroud/Aurora mappings are accurate; their original IDs remain unchanged. **No Rapture mutation was reported**, so the mutation API rejects that name rather than borrowing the intrinsic Rapture multiplier.

`mutation_multiplier(name)` accepts exact display names from the table, including the string `"None"`. `generate_fit(catalog)` verifies every recorded pair before returning ordered fit rows. To regenerate [mutation_fit.csv](../data/review/mutation_fit.csv), run from the project directory:

```sh
python3 -m card_engine.mutations
```

This module records and validates starting-stat observations. It does not certify the affected card abilities or enable simulator training labels.
