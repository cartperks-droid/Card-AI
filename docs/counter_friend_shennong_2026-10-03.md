# Counter teams for a friend: Shennong / Volcano Spirit / Ra / Bad Boys (6M HP, 3M ATK each)

**Enemy:** borderless Shennong, Volcano Spirit, Ra and Bad Boys, each fixed at 6M HP / 3M ATK, in that order (user, 2026-10-03). Their supports aren't known; the search assumes none.

**The friend's supports:** read from his binder by matching the user's binder (same art and shown rarity, `docs/my_collection.md`).
- **Red:** Khione, Neko, General Sun Tzu, Adventurer (all Galaxy).
- **Blue:** Storm Spirit, Final Testament, Mirror Knight, Fate, Berserker, Flame Wizard (Galaxy); Guardian Angel and Jurassic World (Platinum).
- **Left out:** Executioner, Synth Human, Vampire Matron, Phantom and Shielder (tier unknown), and the 1/120M teal blue support (not identified).

**His cards:** any card in the restricted deck up to his rarest deck card, 30qd (`--max-rolls 3e16`). His four deck cards aren't identified, so none counts as owned.

**Search:** `python -m card_engine.training.counter data/scenarios/2026-10-03_friend_shennong.json --mode restricted --max-rolls 3e16`. It evaluated 9,221 teams, each simulated attacking first and defending; every team is in `data/scenarios/2026-10-03_friend_shennong.rows.jsonl`.

## Result

Every winning team (2,034 at 100%) is built on **Vampire Lord at GaPl** (1 in 2.5e16 rolls; a Blood Rain card, Anime pack) with **Neko (Galaxy)**, which doubles Anime cards' stats.

| # | Lineup (in order) | Red | Blue | Win, attacking | Win, defending |
|---|---|---|---|---|---|
| 1 | Useless Seer, True Prophet, **Vampire Lord (GaPl)**, Archer (all others no border) | Neko | Storm Spirit | 100% | 100% |
| 2 | Jersey Devil, Set, Baby Skeleton (no border), **Vampire Lord (GaPl)** | Neko | Storm Spirit | 100% | 100% |
| 3 | Good Boy, True Prophet, **Vampire Lord (GaPl)**, Useless Seer | Neko | Storm Spirit | 100% | 100% |
| 4 | Shining Armor, Set (RuPl), **Vampire Lord (GaPl)**, Useless Seer | Neko | Berserker | 100% | 100% |

- **Why it wins:** each hit steals about 1M ATK and HP from the enemy, so Vampire Lord snowballs while the enemy shrinks.
- **The other three cards matter only when defending:** alone with random base cards, Vampire Lord (GaPl) wins attacking first but loses defending. The listed partners hold the front until it arrives.
- **Border:** below GaPl it loses (Galaxy, RuPl, Ruby and Platinum all fail).
- **Without Vampire Lord** the best team wins 6% (Hathor, Useless Seer, Pandora, Samurai).

## Caveats

- **Vampire Lord's ability is unverified.** "Steal ATK and HP equal to damage dealt" is the simulator's reading of the card text: it heals by the damage dealt and lowers the target's max HP and ATK by the same amount. It has never been checked in game. If the steal works differently, these teams may fail.
- **He may not own Vampire Lord at GaPl.** It sits right at his range limit.
- **The enemy's supports and the left-out blue supports** could change the outcome.
