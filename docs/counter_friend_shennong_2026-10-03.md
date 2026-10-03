# Counter teams for a friend: Shennong / Volcano Spirit / Ra / Bad Boys (6M HP, 3M ATK each)

**Enemy:** borderless Shennong, Volcano Spirit, Ra and Bad Boys, each fixed at 6M HP / 3M ATK, in that order (user, 2026-10-03). Their supports aren't known; the search assumes none.

**The friend's supports:** read from his binder by matching the user's binder (same art and shown rarity, `docs/my_collection.md`).
- **Red:** Khione, Neko, General Sun Tzu, Adventurer (all Galaxy).
- **Blue:** Storm Spirit, Final Testament, Mirror Knight, Fate, Berserker, Flame Wizard (Galaxy); Guardian Angel and Jurassic World (Platinum).
- **Left out:** Executioner, Synth Human, Vampire Matron, Phantom and Shielder (tier unknown), and the 1/120M teal blue support (not identified).

**His cards:** any card in the restricted deck up to his rarest deck card, 30qd (`--max-rolls 3e16`). His four deck cards aren't identified, so none counts as owned.

**Search:** `python -m card_engine.training.counter data/scenarios/2026-10-03_friend_shennong.json --mode restricted --max-rolls 3e16`. It evaluated 9,221 teams, each simulated attacking first and defending; every team is in `data/scenarios/2026-10-03_friend_shennong.rows.jsonl`.

## Result

Every winning team (2,035 at 100%) is built on **Vampire Lord at GaPl** (1 in 2.5e16 rolls; a Blood Rain card, Anime pack) with **Neko (Galaxy)**, which doubles Anime cards' stats. Without Vampire Lord, the best team wins 6% (Hathor, Useless Seer, Pandora, Samurai).

Many teams win 100%, so they're ranked by **headroom**: the mean win chance with the enemy's HP and ATK multiplied by 1.5, 2 and 3. That shows how much margin the other three cards add. "Ability never matters" marks a card whose ability can be removed (stats kept) without losing anything; it's there only as a body.

| # | Lineup (in order, all borderless except Vampire Lord) | Red | Blue | Win, attacking / defending | Headroom | Ability never matters |
|---|---|---|---|---|---|---|
| 1 | Vicious, True Prophet, **Vampire Lord (GaPl)**, Good Boy | Neko | Storm Spirit | 100% / 100% | 77% | Good Boy |
| 2 | Useless Seer, Hard Claws, **Vampire Lord (GaPl)**, Pandora | Neko | Storm Spirit | 100% / 100% | 42% | Useless Seer |
| 3 | Jersey Devil, Set, **Vampire Lord (GaPl)**, Wizard | Neko | Storm Spirit | 100% / 100% | 19% | Wizard |
| 4 | Samurai, Sable The Envious, **Vampire Lord (GaPl)**, Archer | Neko | Guardian Angel | 100% / 100% | 3% | Archer |

- **Pick team 1.** It wins even against an enemy twice as strong most of the time. Team 4 wins today but has almost no margin.
- **Why it wins:** each hit steals about 1M ATK and HP from the enemy, so Vampire Lord snowballs while the enemy shrinks.
- **The other three cards hold the front** until Vampire Lord arrives. Alone with random base cards, it wins attacking first but loses defending.
- **Border:** below GaPl Vampire Lord loses (Galaxy, RuPl, Ruby and Platinum all fail).
- **Teams share at most one card** (Vampire Lord), so these are real alternatives.

## Caveats

- **Vampire Lord's steal is confirmed in game** (`data/review/oracle_observations.json`). Videos IMG_0336 and IMG_0338 show the ATK steal: the target loses ATK equal to the damage dealt, capped at what it has left, and Vampire Lord gains exactly that. The user confirmed the HP steal: the target's max HP drops by the HP removed, and Vampire Lord gains it. The simulator implements both.
- **He may not own Vampire Lord at GaPl.** It sits right at his range limit.
- **The enemy's supports and the left-out blue supports** could change the outcome.
