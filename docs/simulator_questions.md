# Simulator observations and next tests

IDs are one-based original Snap! positions. Border 1 means no border. The user can play lit cards; the Training Dummy can play any opponent. No-support battles are confirmed possible. Source descriptions and proposed registry handlers help design tests but do not prove event semantics.

## Observed

| Setup | User report | Consequence |
| --- | --- | --- |
| Good Boy (#3) versus Good Boy (#3) | Both reduced the other's damage; the user's card activated first, attacked first, and won. | Reproduce this initial side and result. The later general rule confirms the battle initiator attacks first. |
| Chronus The Hoarder (#205) versus Poseidon (#44); no borders/supports | Poseidon boosts stats, then Chronus steals 10%; Poseidon wins. | Enemy-entry reaction follows Poseidon's own stat boost in this matchup. |
| Good Boy (#3), then Chronus (#205) versus Poseidon (#44) | Good Boy reduces attack before Poseidon's boost. Chronus's replacement entry causes no theft. Poseidon ends at 531 displayed ATK; HP cannot be read exactly. | Chronus listens to enemy entry, not his own entry. Immediate rounding up after both percentage changes predicts 532 and does not fit the observation. |
| General rounding clarification | “Always round up.” | Use ceiling for attack damage. Intermediate stat fractions are retained in the current interpretation to fit 531; equivalent modifier composition is still possible. |
| Hell's Army (#33) and Heaven's Armor (#38) | Each has 2560 HP and 1280 damage; Card Modifier affects both stats. | Replace the old attack-only interpretation. |

Weather scaling is intrinsic to a card. Borders increase stats via rarity. The base formula remains provisional for special ratios and screenshot/source rarity conflicts. See `source_conflicts.md`.

Exact user quotations and scope limits are preserved in `data/review/oracle_observations.json`; `data/review/confirmed_rules.json` records the current working interpretation. Do not replace unknown HP with an estimate derived from the red bar.

## Latest multi-hit result

The user confirmed Dancer's first individual attack is blocked by Horus and all follow-ups are calculated normally. A block is consumed per hit in this matchup. The subsequent Good Boy/Poseidon versus Dancer test confirms that a knockout ends Dancer’s turn, cancelling follow-up hits. Raze counters immediately between hits, never chains counters, and cannot counter after a lethal hit. Arthur survives one lethal attack at 1 HP; the next kills him. The initiator must finish with a surviving card or loses.

## Mutation clarification

The user supplied 28 attack/HP pairs for Black Cat, Dancer, and Arthur. All 56 numbers match simple mutation multipliers applied before ceiling the base stats. See `mutations.md` for the complete evidence. Weather cards retain their separate intrinsic multiplier and are ineligible for mutations. The user corrected the earlier analogy that they were mutated cards. Aurora/Shroud mappings are confirmed accurate; keep source IDs unchanged.

## Subsequent tests, only as needed

- Immortal Witch (#94): baseline 868 HP / 290 ATK is now supported by the user and video. Exact heal amount/rounding and border/mutation scaling remain unverified; the mirror does not expose the nominal heal because the cap binds.

- Repeat safeguard: define what a cycle counts, whether active-card replacement resets it, and whether forced deaths invoke on-death/revival rules. A long-lived mirror by itself cannot prove death-trigger semantics.

Other unresolved definitions include status chance/duration/stacking, defense versus ability damage, summon/transform/copy pools, named awakenings, class-target membership, and support composition. Request targeted tests as each primitive is implemented rather than asking for a complete battle manual at once.

`data/review/oracle_battle_cases.json` retains earlier proposed fixtures and conditional variants; it is not a list of currently outstanding requests. The generated 372-handler registry remains archived with its original HIGH/MEDIUM/REVIEW flags. None of those flags alone permits training labels. In particular, Chronus's enemy-entry handler must not be replaced with a generic own-entry handler merely because his name also appears in the supplied entry-name list.

## Entry attacks and lifetime observations

Knightmare mirror opens with the initiator’s entry hit, then the enemy’s entry
hit, then the initiator’s normal attack. Raze counters an entry hit immediately.
When enemy Knightmare’s entry hit kills Good Boy, the player’s replacement
Knightmare makes an entry hit followed by its normal attack.

Inari takes no damage and dies after its third own attack. Against Raze, the
third counter resolves before Inari disappears. These observations are distinct
from the still-provisional scheduling of other turn effects and death reactions.

Wind Spirit attacks twice before Horus and twice again on the next cycle. Only
the first hit is blocked. When Wind Spirit kills Good Boy on its first attack,
its extra turn immediately attacks the replacement Poseidon. This differs from
Dancer’s follow-up-hit cancellation after a knockout.

Martial Artist shows 260 → 338 → 440 ATK, with each boost before Horus’s
reply. Its first blocked hit still grants growth. Raze’s lethal counter kills
Martial Artist before it can receive the boost. The implementation scales
current ATK by 1.3 at own action end after reactions; intermediate fractions
remain a working interpretation rather than a new rounding observation.

Immortal Witch’s healing activates at the start of her normal turn, before
attacking, even when she starts at full health. Healing is capped at maximum
HP. The supplied mirror recording instead reaches a stalemate cutoff: the user
counted 100 turns/50 rounds before both cards fell out. Both cards visibly show
290 ATK. The user subsequently confirmed 868 HP on the card outside battle
and no borders. The wiki-sourced HP 1.2 / ATK 0.8 ratios, applied before
rounding, resolve the generic-formula mismatch without changing raw data.
With those stats, 35% maximum-HP healing reproduces the mirror; 35% current-HP
healing predicts a turn-9 finish. This supports the maximum-HP interpretation
but does not directly measure the uncapped heal amount or rounding.
See `data/review/video_0289/metadata.json` and `data/review/stat_conflicts.json`.

## Next mechanics checks (stats use the corrected formula and mappings)

- Confirmed: Forest Spirit (#7) heals before Raze’s counter. The implementation now uses a separate after-attack phase; it does not reuse Martial Artist’s later turn-end phase.
- Confirmed: Count Muscula (#174) lifesteals before Raze’s counter.
- Confirmed: Shu (#74) dies to a lethal Raze hit; recovery does not prevent death. Damage-before/after-defense basis remains a later check.
- Confirmed: [Forest Spirit #7, Piccolo #104] versus Raze. Piccolo intercepts the lethal counter and blocks it. Raze then makes a normal attack, and Piccolo kills Raze. The implementation doubles Piccolo’s HP/ATK.
- Confirmed: [Arthur #10, Piccolo, Piccolo] versus GaRuCrPl (border 16) Poseidon. The user-read 81,736,447 ATK after Poseidon’s boost matches the border formula exactly. Arthur first survives at 1 HP, then Piccolo swaps in and plays until it dies. Arthur returns, and the third-slot Piccolo swaps in again. So the displaced card swaps into Piccolo’s slot, and own lethal survival precedes interception.
- Confirmed: Poseidon #44 ahead of Piccolo does not re-boost when it returns. The implementation extends this to all own-stat entry multipliers.
- Confirmed: returning Knightmare #14 and Good Boy #3 repeat no entry ability.
- Confirmed: Chronus #205 steals from Piccolo when it swaps in and never steals from the same card twice. Piccolo's 2x is applied before the theft (provisional).
- Confirmed: protection order is True Prophet dodge > survive at 1 HP > Piccolo swap.
- Confirmed: [True Prophet, Platinum Immortal Witch] versus Raze. The Witch (3471 HP, matching the formula with a border) dodges the first hit and survives the counter. Raze's next normal attack kills her.
- Confirmed: True Prophet #164 has an on-death ability. The next card dodges a lethal hit. Against Raze, Shu dodges, then its recovery shows as a healing effect proportional to the dodged damage, never above max HP.
- Confirmed: in the [True Prophet, Shu] mirror, both Shus dodged their first hit (362 of 1230 HP, not lethal) and showed recovery at full HP. So the dodge granted on the Prophet's death applies to the next hit, lethal or not, despite the card text.
- Confirmed stats: Shu #74 is 1230 HP / 362 ATK, an abnormality in its weather multiplier. It is fitted as weather 1.7 on HP and no weather on ATK; see `data/review/stat_conflicts.json`.
- Later: distinguish lifesteal's overkill basis and whether percentage healing uses maximum or current HP on each relevant card.

Do not mark these predictions as observations. Stat verification is deferred;
continue targeted mechanics testing with the user.

## Batch 1 (card-text extrapolation)

37 cards were mapped from their descriptions onto 26 new generic primitives. They include kill rewards, on-hit stat changes, stat theft, retaliation damage, entry debuffs, enemy turn-start drains, per-turn stat decay, next-card gifts and counter damage. See `tests/test_batch1.py`. These readings need checking in batched game tests:

User answers (2026-09-28): 171 (Infected Maw) loses 25% of its **own** max HP each turn; the game does not hurt the enemy. Retaliation fires on a lethal hit and can kill the attacker. 91's stolen HP can exceed max HP. 282 heals before each attack. 222 gifts half of the stats the card had when it died. Kill rewards resolve before the next enemy card enters. All are implemented.

Still extrapolated: kill-steal basis (victim's max HP/ATK), 11/115 triggering on counters, 219/238 decay on both HP values, 58/241 applied before the enemy's own turn-start heal, and 163/46 applied before the enemy's entry boost.

## Batch 2 (card-text extrapolation)

36 cards covering rests and recharge, first-turn and growing turn counts, alternating and own lethal dodges, revives, bypassing defenses, conditional and hit-growth damage, executes, fallen-ally and party scaling, and limited or periodic growth. See `tests/test_batch2.py`. Readings to check:

User answers (2026-09-28), all implemented:

- Recharge cards (24 Leviathan, 42 Thor, 96 Failed Mage) attack normally on recharge turns, without the charged bonus. Enemy abilities still apply.
- 89 has 2 baseline attacks and gains 1 each turn.
- Deus Ex (54) dodges odd incoming hits. Amaterasu (118) nullifies even hits and heals. Her displayed HP is bugged (alive on an empty bar), so the heal amount is unverified; the card text's 80% of the nullified damage is kept.
- 169 revives at full HP.
- Bypass (23, 42, 102, 134, 149) defeats both blocks and chance dodges.
- 161 Resolute Blade loses the converted 15% of its max HP with each attack, so its HP decays exponentially.
- 243 Community counts every party card, dead or alive.
- 248 kills both cards, even when the other is Parallax (65, not yet implemented), so mutual destruction overrides death prevention.

Still extrapolated: whether bypass also defeats alternating, True Prophet and own lethal dodges (it does not), 190's bonus per attack, fallen allies counted only at entry, and execute thresholds checked only after attack hits.

## Batch 3 (statuses)

Implemented: burn and bleed (tick at every turn end, 10% and 16% of max HP), freeze (blocks all actions including counters; durations tick at every turn end) and slow (skips the card's next own turn). Refresh and stacking follow the user's rules. Cards: 57, 81, 92, 135, 147, 153, 167, 198, 257. Durations missing from the card text default to 3 turns. 100 Men's (213) 100x HP is now part of its baseline stats.

Replays: Cr Ankylosaurus kills Ru 100 Men on the 6th bleed, and Ru 100 Men beats PlRu Volcano Spirit after 7-8 burns with 16-27% HP left. Both match the user's reports.

Next: frostbite (video 0290: about 50% chance of 20% max HP damage plus a lost turn), chance-based procs (Ice Queen, Guardian Angel), confusion, and poison (needs precise tests).

## Batch 4 (chance mechanics)

Chance rolls inside a step now branch exactly: branch mode enumerates every roll outcome depth first, and sample mode draws lazily. Initial entries can also roll. The C frontier grows as needed, so capped searches no longer lose probability.

Implemented cards: 15, 21 (Ice Queen), 22, 29, 31, 32, 50, 63 (Yeti), 64 (Achyls), 100, 148, 201, 216, 220, 232, 245, 251 (Frosty), 252 (Old Man Winter), 260 and 263. The Guardian Angel support (blue 15) gives 10% once per ally to survive lethal damage at 1 HP.

Frostbite: each tick rolls 50%. On success the card takes 20% max HP and loses its next turn. Frosty's lasts 1 turn (video 0290: one roll per round), and multi-turn frostbite can proc more than once (Old Man Winter).

User answers (2026-09-29), all matching the implementation: confusion self-hits use the card's own full attack. Old Man Winter's three debuffs are equally likely. 216 uses whole multiples. Arthur and Guardian Angel both survive 50's "infinite" damage.

Still open: chance revive (22/31/32) versus Piccolo order, and 29's miss semantics.

## Batch 5 (card-text extrapolation)

36 cards: fades (206, 217, 285, 181), alternating rests and shields (145, 258, 142), 191's black box, next-card and entry effects (84, 259, 197, 40, 279, 41, 235), turn-start effects (280, 136, 150), 132, 170, 208, 187, 202's extra attack on kill, field damage (8, 168), ally buffs (128, 51), 45, stored and shared damage (227, 231), and rarity/pack/class matchups (20, 68, 223, 12, 124, 55). The class matchups use inferred, unverified art classes; Raze is inferred a dragon.

Readings to check:

1. Fading (206 Malik, 217, 285): once below the threshold, do the bonuses stay gone even after healing (implemented), and does 285 lose its 4x stats?
2. 145 and 258: do they act on the first turn and rest on the second (implemented)?
3. 142: is the shield present from the start (implemented), and is it regained every other turn?
4. 202: does the extra attack after a kill hit the next enemy card immediately (implemented)?
5. 20 "deal 2 x damage, take half if border is rarer": are both conditional on having the rarer border (implemented)?
6. 223 "younger cards": are cards from later packs younger (implemented)?
7. 8/168 "boost all cards' damage": both teams' cards, while the field card is alive anywhere in the party (implemented)?
8. 51: does it dodge every normal attack by paying 20% of current HP (implemented)?

## Batch 6 (waiting cards; Academy Student and Seraphim)

User: descriptions are accurate apart from rare exceptions. (A shared clarification embedding for those cards was removed on 2026-09-30 at the user's request: several apparent deviations were display artifacts, and the game logic may be consistent with the text.)

- Academy Student (88): the other enemies lose 30% of the damage dealt as direct HP loss (the card text says 40%). The basis is the HP the hit removed (implemented).
- Seraphim (172, PG ATK 41,943,040 confirmed): the entry attack hits the whole team for 25% (dodgeable); see "Seraphim entry rules" below.
- General rule now in both engines: a card dies only when a hit or effect actually lowers its HP to 0 or below this step.
- A card with its own chance dodge (Nao Presence) cannot use True Prophet's grant; unused grants pass on at death (reproduces both Result 1 branches).
- A Piccolo swap-and-block deals no damage, so there is no splash, and both cards survive that hit (user).

Videos 0291-0293 were reviewed; the on-screen attack of Piccolo (3,072), Inari (352) and Infected Maw (510) suggests mutations the lineup notes did not mention.

## Seraphim entry rules

From user tests (RuCrPl to PG borders; IMG_0291-0296; mid-battle entries behind Stone Scientist or Mrs. Claus; Deus Ex, Noveau and Judgement Day runs). Implemented in both engines; no strength threshold.

1. The entry hit hits the front card and 25% of it hits every waiting card. Waiting cards can dodge it, use Prophet grants, invincibility, lethal survivals (Noveau restores to full) or Parallax's reflect.
2. Waiting cards reduced to 0 HP die in the deck; their on-death effects go to the card still playing (True Prophet's grant goes to Arthur). A front card's on-death effects go to the next card in the lineup, dead or alive, so Prophet, Stone Scientist or Mrs. Claus can be wasted on a card that died in the deck (user).
3. **Display bug (confirmed; see "Known game display bugs"):** the game drops one card from the front of the shown lineup per deck death. So the card on screen is usually an earlier, surviving one. For example, "Forest Spirit attacks once with an empty bar" is Arthur at 1 HP, and "Maw has Inari's ability" is Inari. The earlier entry-kill, arrival-death and ability-passing rules described this display, not the game, and were removed.
4. The entry hit fires even while Seraphim is frozen (Mrs. Claus); the freeze only skips its normal turns.
5. Deus Ex always dodges the entry hit while waiting (user: the 2nd and 3rd shown cards, really Arthur and Deus Ex, always survive; without Maw, Arthur is shown surviving at 1 HP). Deck hits count toward its alternation, so its next hit lands.
6. Noveau Riche has three revives (the entry used one and two remained).

This reproduces every recorded Seraphim run once the display shift is applied.

Open:

- Resolved: Mrs. Claus killed in the deck doubled Arthur, the card playing (shown 66 ATK = 2 x 33).
- Other cards: a PG Knightmare's entry behaves normally (user); it hits only the front card.
- Parallax (65): once, a lethal hit is negated and the attacking card dies, also when hit in the deck. Against Seraphim this kills Seraphim during its entry (user: all four Parallax lineups won; Inari + Parallax defeats PG Seraphim). #248's mutual destruction still kills it. Animation order (Seraphim falling after Inari arrives) is not modelled.


## Batch 7 (extrapolated from card text, provisional)

16 cards: 4 (no battle effect), 16, 26, 27, 80, 95, 109, 110, 111, 116, 184, 189, 192, 194, 218, 269. New primitives in both engines: on-death blasts (a share of the dead card's max HP taken directly off the enemy front card, optionally by chance) and on-death ally boosts; periodic specials every Nth own turn (damage multiplier, hit count, heal, enemy loses a turn); follow-up hits (optionally bypassing defenses or only after a critical); statuses applied to every enemy card met while active; extra actions below an HP threshold; the friendship bonus (computed when the battle is compiled).

Questions for the user (each reading is a guess):

1. Savior (26) / Frankenstein (27) / Valentine's Specter (218): does the death damage hit the card that killed it, is it the dying card's full max HP, and can it be dodged? Does it also fire when the card dies in the deck?
2. Hollow (184): is the boosted attack on turns 3, 6, 9 (implemented) or 4, 8, 12?
3. Susanoo (116): on turns 3, 6, 9 its attack is doubled and the enemy loses its next turn. Correct?
4. Melanin (189): every third attack is three hits of 50% (implemented), with 25% lifesteal on all hits?
5. Sorceror (80): an extra hit at 50% after every attack, ignoring dodges and defenses? Durante (194): the follow-up only after a critical? Tornado (95): 2x hit followed by a separate 4x hit (implemented), or 2x then 4x on alternate turns?
6. Ice King (192): does it freeze every new enemy card for its first turn, including the one already in front when Ice King enters?
7. Tartarus (16): below 50% HP, 3 actions per turn (1 + 2)?
8. Shuten-doji (111): is 2x damage always on, or only part of the kill bonus?
9. A0-ON1 / AK4-ON1: +40% per unique friendship card including itself (alone x1.4, together x1.8)?
10. Scarecrow (269): which cards count as "bird cards"? It currently uses the art-inferred avian class.

## Poison, fossils, stun (user, 2026-09-29)

- Poison (IMG_0297, Black Plague vs Platinum Jamiy): each tick takes the poisoner's ATK x effectiveness off the target directly, ignoring damage reduction, at every turn end; poisoned waiting cards tick too. Dilophosaurus ticks for 100% (1,559 into Heaven's Armor, which kept 221 HP as the video bar shows); Black Plague for 50% of its ATK on every enemy for 2 turns. Implemented for 52, 146, 237. Open: Dilophosaurus's duration (2 assumed), and whether re-poisoning refreshes or stacks (refresh assumed, keeping the stronger tick).
- Fossils are counted per team (user). Not yet implemented.
- Stun disables attacking but not abilities (user). Not yet implemented.

## Batch 9 (videos IMG_0299-0304 and user answers)

Implemented in both engines: Ghoul (poison + non-stacking 30% weakness), Zombie Nurse (bleed, one 50% heal below half HP), Zombie Dragon (survives lethal once, untouchable for 2 turns then dies; poison ticks for 15% of the target's max HP), Kira (every enemy it meets dies after 2 turns, even after Kira's death), Onmyoji (3x its ATK to the enemy front card 5 turn ends after entering, even after death), Rudolph (blind: the target's attacks always miss, IMG_0299), Banshee (the attacker that drops it below half cannot attack next turn), Gambler (ATK -10% / +15% / +30% after attacking), Loch Ness (x1.1 stats per allied turn while waiting, up to 3x base, IMG_0303), Steve (each attack adds a block absorbing 40% of max HP; at most 3).

User answers (2026-09-30), implemented:

- Onmyoji and Kira count the normal attacks of the ally card in play, just before each attack, and only when that card can attack; they keep counting after they die and hit the enemy card in play (IMG_0303: the strike landed just before Zombie Dragon's first attack). If the count kills the enemy card, that attack is spent (provisional).
- Steve's blocks share one damage pool (a block disappears once a block's worth is used up).
- Banshee stuns only once, for 1 turn. Dilophosaurus's poison lasts the whole battle (IMG_0303).

Open: Ghoul's poison duration and Gambler's draw rates (user videos to come); Tsukuyomi, Buddha.

## Known game display bugs

These are how the game *shows* battles, not how it resolves them. Translate every battle report and video through them before inferring a rule.

1. **Deck-death shift (confirmed across every Seraphim run, IMG_0291-0296 and user tests).** When waiting cards die in the deck, the display drops one card from the front of the shown lineup per death, so the card name on screen is usually an earlier, surviving card. The stats shown belong to the real card: "Mrs. Claus" showed 66 ATK, which is Arthur's 33 doubled by her death. The display realigns once the real order catches up (e.g. Noveau shown correctly at the end). Signs of it: a card dying "instantly" at entry, cards appearing to inherit another card's ability, or a card with an empty HP bar still fighting.
2. **Amaterasu (118):** alive on an empty HP bar; heal amounts cannot be read.

## Batch 12: the last 56 cards (2026-10-01)

Every card is now mapped (289/289) in both engines, kernel 0.42. Teams hold up to 16 cards: the starting lineup (1-4) plus spare slots for summons. The ability pool holds every card's ability plus the forms; random draws are uniform. Readings, user answers and provisional choices:

- **Summons.** Nuwa: on entry, one random card (not Limited, not Glamour) with Nuwa's stats goes to the back (user; IMG_0336). Fresno: two copies with 35% stats and no ability (user). Dr. Frankenstein: a Frankenstein with its stats per kill (at most 3). The Broken One: Joy then Sorrow, no abilities, with 50% max HP / 60% ATK of its stats at death (user). Control Freak: a defeated enemy joins the back with its base stats and ability (user: as described).
- **Revives.** Anubis: half HP, at the back, whenever an ally dies (user). Bloody Mary: 50%, full HP, at the back. Anubis & Hades: per fallen enemy, revives itself first (else the first dead ally) with 75% of its stats, at the back (IMG_0311). Buddha: revives the first dead ally at the back with full HP (user) or heals the weakest card by 50% instead of attacking; moves to the back when every living card is full (IMG_0303/0304). Time Lord Stryx: once, the party returns in its initial order at full HP (IMG_0310). Walking Dead: survives 2 turn ends at 0 HP; each kill adds 2.
- **Transforms.** Sun Wukong x2 stats below 50% HP; Naga heals to full below 65% and its form curses the enemy card for 4 turns on death; Demon / Immortal Cultivator +30% of base stats for 2 turns, awakening on the 3rd turn (Dark Awakening: 2x damage, 30% lifesteal; Immortal: half damage taken; both +50% base stats per kill, IMG_0327); Ultimate Brawler awakens instead of its first death (1.5x stats, full HP, dodges every other attack).
- **Random abilities.** Pandora: two different random abilities (user: any but its own). Loki: takes the enemy's ability, gives it a random one. Glamour (IMG_0330): a new random card at the start of each of its turns; the form's entry effects apply to its current stats, an entry attack comes first, stats carry over. Astraeus: one of 6 arts (docs/card_forms.md). The Awakened One (IMG_0342): one of 6 weapons; the Great Nirvana Sword - Zero gives 2 of the other 5. Summoned and drawn abilities work in full (IMG_0336: a Nuwa-summoned Anubis returned after Savior died, at 3,246 = 2,705 + 20%; IMG_0330: Glamour's The Broken One split), so sides that can draw random abilities get every free spare slot.
- **Lineup.** Jersey Devil shuffles the living enemy lineup on entry (IMG_0341). Marionette swaps 2 random living enemies after attacking, -15% stats each (IMG_0336). Flying Dutchman swaps with a random ally after attacking (at most twice); the ally takes the next attack and has one turn, then the Dutchman swaps back, or when the ally dies (user). Cat Lady: once, evades a lethal hit and swaps with the next ally. Heavenly Demon: 50% to pull a random waiting enemy forward and hit it for 3x. Cosmic Pop Star: 75% of the stats of the card behind; that card is banished (no death); counterattacks.
- **Deck hits.** Chaos: the card in play and up to 3 waiting enemies are each hit independently (50%) for 50% damage (user). Kraken: entry hit on the whole team, 1-8 times at 10%. Brachiosaurus: its attack hits a random living waiting enemy. Tricerotops: entry attack; overkill passes to the next card (user). Bei Fang Xuan Wu takes 50% of the damage its allies take, from the deck too (user). Fate Seamstress links the first 2 enemies (the damage is dealt to both). The Rake adds 25% of its ATK when its active ally attacks; the Hanged Man's curse does not reach it (user).
- **Statuses.** Mist Spirit: cards with any status cannot hit it (5 dodges). Serket: its side is immune to statuses while it lives. Kuchisake-onna: confusion until a self-hit, which costs 20% ATK. Cthulu: every enemy it meets is confused for good; +50% stats per self-hit. The Composer: 10%, +10% per turn, to confuse the enemy at each allied turn start. Tsukuyomi: the opponent's even turns are reflected onto the attacker, with the effects attacks carry (user). The Hanged Man: its killer loses 25% max HP at the start of each turn (user). Sleep Paralysis: the enemy is stunned for 1 turn (it skips its attack; abilities still work), and both active cards die at the start of its 4th turn (user; IMG_0345).
- **Other.** The Curse: while in play, enemy dodges, blocks, damage reduction and survivals are off (not Parallax); attacks are 2-5 slashes at 35%. Marrowclaw: 50% to negate the enemy's blue support; ability damage cannot kill it. Juggernoid reflects 25% of ability damage (at most 8x its ATK). Poison Witch resets the enemy's stats on entry and drains 20% of its ATK as healing after each enemy turn. The Sack: independent 1-50% damage, 1-50% max HP and 1-30% damage reduction (user). Santa Claws: living allies +50%, then -10% stats per turn in play. Eonus: 5% of every other card's stats; they decay after 3 of its turns. Eclipseborn Luminant: allies get 40% evasion (-10% per evade, 20% floor, 2 per card); it gains 10% of the prevented damage, up to 3x its ATK. Divine Doctor: every 3 allied turns, all living allies heal to full. Gingerbread Man: dodges every other attack, one extra attack next turn per dodge. Uncle Sam: the next card deals 5x and dies after attacking. Kid Gohan: Piccolo gains 1.5x instead of 2x. Milk: gives its stats to Dad and dies at its turn start; without Dad it fights (user). Longmu: first in the deck it blocks the first attack and dragons in its deck gain 20% each of its turns; otherwise dragons in its deck are shielded from 2 attacks.

Resolved (user): Astraeus is seven separate cards, one per art (five in the videos, Taurus, and a sign that nullifies damage below 100% of max HP, -15% per nullification); Aquarius (the blue-haired art) heals at or below 50% HP and otherwise grows max HP by 25%; Slum Dweller was nerfed to 10% (description and effect, IMG_0330).

Resolved (IMG_0345, test designed for it: Platinum Yamato no Orochi vs a borderless Sleep Paralysis, alone and behind Good Boy): both battles ran identically - Sleep Paralysis hits, the stunned Yamato (yellow) skips its turn, then they trade hits (Yamato's ATK fell 1/8 per hit: 1,515 -> 1,326 -> 1,160 -> 1,015 and 1,782 -> 1,560 -> 1,365 -> 1,194), and both active cards die at the start of Sleep Paralysis's 4th turn. IMG_0341's readings were confounded by Jersey Devil's shuffles.

## Supports (IMG_0350-0354, 2026-10-01)

Each support card has exactly one border tier: I base, II Platinum, III Crystal, IIII Ruby, IIIII Galaxy. Pass a support as an `id` (base tier) or as `(id, tier)`. The per-tier values are in `RED_SUPPORTS` and `BLUE_SUPPORTS` in `catalog_rules.py`. A cell that couldn't be read in the videos is `None`, and using it raises `UnsupportedCardError`.

- **Red:** a stat percentage applies to every card, plus a larger percentage for the matching pack, weather, or class. The game's description pairs were swapped for 4/5, 13/15, 20/21, 22/23 and 26/27, and those are corrected on import. General Sun Tzu boosts HP only. The One Ring has no matching bonus.
- **Blue mechanics:**

| Support | Engine effect |
|---|---|
| Shielder | incoming damage reduction |
| Flame Wizard | chance to burn on hit |
| Berserker | chance to counter |
| Phantom | chance to stun on hit |
| Vampire Matron | lifesteal |
| Synth Human | dodge while below max HP |
| Executioner | bonus damage against targets under 30% HP |
| Fate | rerolls failed chances |
| Mirror Knight | reflects damage |
| Final Testament | the next card gets stats |
| Storm Spirit | chance of a 50% follow-up |
| Jurassic World | prehistoric stats × (1 + v × prehistoric count) |
| Guardian Angel | guardian chance |

  - Magical Elf (IMG_0355): if the deck holds enough unique Toys (base 4, Platinum 3, Ruby 4), they awaken:
    - Toy Car (Speedy Progression) gets one attack per other Toy, at full damage.
    - Jack-in-the-Box (Pop-Up Impression) confuses each entering enemy for a turn per unique Toy.
    - Toy Nutcracker (Shelter Obsession) has damage capped at 1/4 max HP, and every living awakened Toy gains 10% stats each time it is damaged.
    - "Toy" means the four cards named Toy (261-264).
  - End Times (user: "a transient roll for an ability to fail"):
    - Every step, the enemy front card's ability fails for that step with the support's chance.
    - Chance dodges and criticals are scaled by (1 - chance).
    - Marrowclaw's negation stops End Times.
- Ruby (IIII) blue supports really use their base values (user). The tier values the user supplied on 2026-10-01 are filled in.
- Classes (developer lists via the user, 2026-10-01):
  - The developer's demon, dragon and avian lists were added. Earlier members stay until the user verifies every class.
  - Swordsmen are 2, 10, 14, 18, 20, 23, 29, 33, 55, 87, 106, 113 and 116.
  - Army's Reinforcements boosts Swordsmen; its Galaxy tier is 117/234.
- Magical Elf needs one Toy fewer per tier (4/3/2/4/1; Ruby keeps 4).
- Awakened Toy Bear gets ×(1 + fallen awakened Toys) HP and ATK on entry. User: 3,886 / 5,829 / 7,772 ATK, and 15,544 HP at 3 fallen = Platinum stats.
- Phantom Galaxy is 20%.
- Santa Claus has a 2× stat multiplier, not Aurora's 1.75×. Toy Bear's 2/3 card modifier cancels Snow's 1.5×.
- Hades behind Hades (user, 2026-10-05): the second Hades copies the first's ability, which is useless on its own. His engine agrees (battle-v2.ts: Hades copies the latest fallen ally's ability other than The Underworld), so no change; the classifier overrated such teams at floor 105.
- Achlys's Divine Mist resets the facing enemy to its card's base borderless stats, fixed tower stats included (his engine, battle-v2.ts). Floor 105's Sable the Envious (Jealousy) turns it back onto Achlys, so Achlys teams lose there (engine 0.0).
- Ruler of Humans (Supreme Ozzy) adds 1.25 × the HP Ozzy actually loses, not the whole hit (user, 2026-10-06). Heaven's Armor@RuPl (163,840 ATK) hit Ozzy with Dinosaur King Crystal (×2.28, so 3,918 × 2.28 = 8,933 HP) and Guardian Angel Crystal; every ally's ATK went to 2.28 × deck ATK + 11,167 = 1.25 × 8,933 (Dancer 65 → 11,315, Arthur of Excalibur 1,447 → 14,466, Parallax 10,240 → 34,514). His engine adds 1.25 × the uncapped hit (battle-v2.ts: `damage` reaches targetRetro before the HP cap), 204,800 here. Corrected in `codemod.mjs` and `sim_c/engine.c` (the HP lost to the latest hit, 0 when another card takes it); reported to DaddyDrago. Open: the user saw Ozzy's HP as 4,467 on entry, half of 8,933.
- **Open:**
  - The user will verify every class list.

- **Shuten-dōji, Decapitate** (user, 2026-10-06): nerfed from 2x to 1.5x damage. DaddyDrago's engine already deals 1.5x (`battle-v2.ts` `Decapitate`, mirrored in `sim_c/engine.c`), so labels are unchanged; the model's card text said 2x and now says 1.5x (`import_snap.USER_DESCRIPTION_EDITS`).
- **Eclipse x5** (user, 2026-10-06, the game's combat changes): the Eclipse *weather* (user), not the mutation, which stays 2.5x. DaddyDrago's data already gives every Eclipse-weather card a stat multiplier of 5 (Set, The Grinch, Pandora, Hecate, Chaos, Baba Yaga, Umbrasaur), so nothing changes.
- **Jurassic World: no cap** (user, 2026-10-06, tested in-game: "it does go to 2x"). The game's combat changes said "Jurassic +80% max", but four Prehistoric cards at Galaxy (4 x 25%) reach 2x, as in DaddyDrago's engine, so nothing changes. A cap was briefly added and reverted the same day; a run that already declared `--supports blue13:5` for it loses only those rows until they are relabelled. The other combat changes (Spotlight 75%, Sacred Judgment 25%, Frozen Solitude, Eclipse x5 for the weather) already match his engine (user: changelog entries).
- **Hades, The Underworld** (user, 2026-10-07): Hades cannot copy Parallax's Paradox, and a fallen Hades offers only its own The Underworld, never the ability it copied ("two Hades consecutively is wrong"). DaddyDrago's engine copied both (`battle-v2.ts` on entry, and `resolvePandoraGainedAbility` when Pandora's Box gains The Underworld), so the model search's Parallax / Hades / Hades / Robin Hood beat floor 105 Impossible every time, each Hades a second Parallax. Corrected in `codemod.mjs` (`underworldCopy`) and `sim_c/engine.c`: newest fallen ally first, Paradox passed over, a fallen Hades stops the search with nothing copied. That deck now loses floor 105 (0.0); his Parallax / Judgement Day / Judgement Day / Robin Hood still wins 0.316. Hades passes over a fallen Parallax to the ally before it (user: it does, and it makes no difference, Parallax's one activation is global). Hades fires The Underworld again each time it comes back to the front (user: a second Hades with Piccolo behind it copies Piccolo once Piccolo has swapped in with Aura Farm and died; only Hades, other entry abilities fire once): his engine ran a card's entry once (`onEntry` returns when `entered`). Corrected in `codemod.mjs` (`underworldReturn`, before each turn's entries) and `sim_c/engine.c` (`underworld_return`): a Hades that left the front drops its copied ability when it returns, The Underworld copies again, and the copy gets its own entry. Checked: in Hades / Hades / Piccolo / Robin Hood against floor-35 Hard Archers, the second Hades holds nothing until Piccolo falls, then Aura Farm.
