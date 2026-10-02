# Limited probability kernel

This C99 kernel implements an explicit provisional subset, not verified game
logic. It supports fixed initial HP/attack, multiplicative outgoing/incoming
damage, dodge, first-hit or first-nonlethal-hit block, multiplicative entry self
stats and entry enemy attack, enemy-entry stat theft, bounded multi-hit actions,
critical hits, max-HP reductions/caps, threshold defenses, immediate non-recursive
counters, limited lethal survival, invincibility, own-action lifetimes, entry attacks, kill healing, alternating actions, and a
configurable repeated pair limit. It contains no card names, rarity/stat formula,
status effects, revival, ability stealing, or training. Python compiles supported
card/support descriptions into these primitives.

Initial entry applies side A then side B, regardless of first attacker. Dodge is
resolved before a block. Attack damage defaults to ceiling. Intermediate entry stats retain fractions by
default (`stat_rounding=unrounded`), matching the reported Good Boy/Poseidon result of 531
ATK. This does not uniquely establish internal game storage. Separate explicit
rounding policies exist for synthetic comparisons. Every hit resolution consumes one step; natural replacement resets the pair-action
counter. The default `repeat_cycles=50` follows the user-counted Witch mirror.
After twice `repeat_cycles` actions without a replacement, both active
fighters die. Replacement entries apply A then B. Depletion is terminal before
further entry effects. These scheduling choices require real-game validation.

After a fighter applies its own entry modifiers, the opposing active fighter's
`enemy_entry_steal_fraction` takes that fraction of the entering fighter's current HP/attack,
subtracting those amounts from the entrant and adding them to the thief under
the selected stat-rounding policy.
Both reactions are registered initially; entering does not trigger the entering
fighter's own enemy-entry reaction. Reactions do not recurse. Theft that kills
on entry raises an explicit error because that transition is not implemented.

`CE_SAMPLE` runs one trajectory per input battle using xorshift64star, with
independent seed `(seed + match_index) modulo 2^64`. Seed zero maps to the fixed
nonzero constant in the implementation. It draws only for probabilities strictly
between zero and one. `CE_BRANCH` expands dodge, critical hit, and normal hit, merges byte-identical
complete states, prunes merged live states below the configured probability
threshold, then caps the frontier by probability with stable insertion-order
ties. Pruned/capped/step-limited probability is returned as `unresolved`; it is
never reassigned to a winner. Results sum to one within floating-point error.

Compile explicitly from the project directory:

```sh
python3 -c 'from card_engine.simulator.native import build_library; print(build_library())'
```

Use `simulate_batch` from `card_engine.simulator.native`. A battle is
`{"teams": [[fighter], [fighter]], "first_side": 0}` and a fighter requires
`hp` and `attack`. Optional fields and C layouts are in `card_sim.h`. Unknown
Python fields, invalid numerics, arithmetic overflow, allocation failure, and
missing libraries raise errors. The checked-in source is portable; a compiled
library is a local build product.

A multi-hit action retains its attacking side until all hits resolve. Dodge and
block are evaluated per hit; per-hit dodge independence remains an experimental
interpretation. The pending hit index is part of canonical state identity. The
repeat counter advances once per complete action. A knockout cancels follow-up hits. A surviving defender can counter immediately
after each damaging hit; counters do not trigger counters or consume a normal
turn. Pending counters and used lethal-survival charges participate in state
identity. Simultaneous team depletion is a loss for the initiator. Dancer/Horus
has a regression for one blocked hit followed by two normal hits. ABI version
is 0.10-provisional.

Mutation scaling is applied by Python when compiling initial HP/ATK, before the
C boundary. The kernel sees the resolved stats and does not guess mutation or
weather factors. Weather cards retain their intrinsic factor and are ineligible for mutations.
Illegal mutation selections on weather cards are rejected before compilation.

Entry attacks use the shared hit/defense/counter resolver and a separate two-side
queue. Initially the initiator’s entry hit precedes the defender’s, then the
initiator takes its normal action. Entry hits and their counters consume steps
but do not advance normal-action or lifetime counters. Dead cards lose queued
entry hits; replacements queue their own entry hits before normal combat resumes.
The pending queue, counter origin, and each card’s completed lifetime actions
participate in state identity. Inari expiration occurs after the final normal
action’s counters, bypasses lethal-hit survival, and grants no enemy kill heal.
Generic stat-entry ordering versus entry damage, simultaneous replacement order,
and interactions with unimplemented death/revival effects remain provisional.

`actions_per_turn` grants separate complete actions before the other side acts.
It is independent of `attacks_per_action`: Wind Spirit receives two normal
actions every cycle, while Dancer has three hits in one normal action. Remaining
hits stop at a knockout; unused extra actions can attack a replacement enemy.
Each complete action advances own-action lifetimes and the repeat safeguard
count, while entry hits and counters do not. Unused allowances are part of state
identity and are discarded if their owner dies; inheritance after a replacement
entry kill is a provisional scheduling interpretation.

`action_end_attack_multiplier` scales the surviving actor’s current ATK once
after its normal action and reactions finish, before the opponent’s normal
action. A block still completes an action; individual multi-hits, entry hits,
and counters do not independently trigger this effect. The default stat policy
retains fractions; the trace includes both internal and ceiling-displayed ATK.
Martial Artist’s reported 260 → 338 → 440 agrees with this policy but does not
uniquely prove intermediate rounding. A lethal counter prevents the boost.

`action_start_heal_max_hp_fraction` heals once before the first hit of each
normal action. It also activates at full HP (zero effective gain), as observed
for Immortal Witch. Healing is capped at current maximum HP, does not increase
maximum HP, and cannot revive a dead card. Entry hits, counters, and resumed
follow-up hits do not separately trigger it. Healing currently rounds its
amount with the damage-rounding option; exact heal amounts remain unverified.
Immortal Witch’s experimental mapping uses 35% maximum HP. With her confirmed
baseline of 868 HP / 290 ATK, it reproduces the 100-turn mirror. A current-HP
interpretation instead predicts a turn-9 finish for those starting stats. The
video does not establish extra-turn/multi-hit cutoff counting or death triggers.

### Healing reaction contracts pending game validation

`heal_damage_dealt_fraction` heals the attacker immediately after a damaging
hit, before counter reactions. `heal_damage_taken_fraction` heals a surviving
defender at the same damage boundary. The current interpretation uses actual
HP removed after defenses and lethal survival, capped to pre-hit HP to exclude
overkill. Damage recovery cannot rescue a lethal hit or revive a dead fighter.
Prevented and zero-damage hits trigger neither effect. These effects apply per
hit, including entry and counter hits; healing itself cannot queue a counter.

`action_end_heal_max_hp_fraction` activates once when a living fighter completes
its normal action, after its follow-up hits and counter reactions. A blocked or
dodged action still completes; an entry hit or counter does not independently
trigger this effect. An extra action gets its own completion heal. A lethal
counter suppresses the original attacker's heal. The current order places this
heal before end-of-action ATK changes and lifetime expiration.

All healing shares the user-confirmed maximum-HP cap. Amounts currently use the
configured damage rounding. The new timing, percentage basis, overkill behavior,
and specific card applicability are provisional; synthetic regression tests and
Python/C parity test the implementation contracts, not the game's hidden logic.

`after_attack_heal_max_hp_fraction` is separate from action-end healing:
Forest Spirit’s user-tested heal resolves after its normal attack and before
Raze’s counter, even when that counter then kills it. The current composition
runs this once on the final normal hit (or a knockout), before counter queuing;
its multi-hit frequency and blocked/dodged-hit behavior are not yet verified.
Entry/counter attacks do not independently invoke it. Count Muscula’s normal
lifesteal-before-counter timing is now also user-confirmed. Neither observation
settles the healing percentage basis, overkill calculation, or exact rounding.

The kernel is compiled with `-ffp-contract=off`. Fused multiply-add skips an intermediate rounding, which broke bit-for-bit parity with the Python reference (for example `160 - 320 * 0.35` rounding up to 49 instead of 48).

Monte Carlo playouts (user suggestion): with `rollouts=N` in branch mode, every state pruned for probability or frontier size is continued by N sampled playouts. Each playout's outcome receives the state's probability divided by N, so totals still sum to one. The playout-assigned probability is reported as `estimated`, separately from exact branching. Unfinished playouts stay in `unresolved`. Playout seeds depend only on `seed` and a running playout counter, so the Python reference and the C kernel agree exactly.
