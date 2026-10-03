# Cross-check with DaddyDrago's engine

DaddyDrago's Card RNG Expansion Depths calculator (github.com/daddydrag0/CardRngExpansionDepths, read 2026-10-03) simulates the same game and is used by hundreds of players (user). The user rates his stats and abilities as more accurate than ours (2026-10-03). His rarities, card stats and Final Testament values are adopted (`card_engine/data_corrections.py` `DRAGO_STATS`, `stats.STAT_RATIO_PROFILES`, `catalog_rules.BLUE_SUPPORTS`); the rule differences below stay open as checks (`docs/daily_checks/queue.md`, item 7).

## Where we agree

- **Stat formula:** power = 10 × 2^log10(rarity) × card multiplier × mutation; ATK = power / 2.
- **Border rarities:** Platinum 1e2, Crystal 1e4, Ruby 1e5, Galaxy 1e6, multiplied when combined (RuPl 1e7, GaCrPl 1e12).
- **Mutation factors:** Storm 1.1, Snow 1.2, Aurora 1.3, Shroud 1.5, Meteor Shower 1.8, Time Storm 2, Eclipse 2.5, Virus 3, Blood Rain 3.5, Armageddon 4, Manga 4.5.
- **Supports:** a red stat aura and a blue ability aura; support cards come only in base, Platinum, Crystal and Galaxy (our tier 4, Ruby, repeats base and doesn't exist). Galaxy values match (Storm Spirit 30, End Times 25).
- **Vampire Lord:** steals ATK and HP equal to damage dealt.

## Rules that differ

| Rule | Ours | His |
|---|---|---|
| Both sides wiped out | attacker (A) loses | draw |
| Endless battle | 100-turn stalemate per pairing (user-confirmed 2026-10-02) | draw after 2,000 turns |

## Rarities adopted from his data (2026-10-03)

| Card | Was | Now (his) |
|---|---|---|
| Azure Witch | 3,100,000,000 | 3,118,418,147 |
| Buddha | 10,000,000,000 | 10,800,000,000 |
| Xīfāng Báihü | 400,000,000 | 450,000,000 |
| The Curse | 6,660,000 | 6,660,666 |

## Borderless stats adopted from his data (36 cards)

Adopted 2026-10-03. His data folds weather into each card's one stat multiplier: several weather cards (Santa Claws, Durante, Dionysus, Loch Ness, Old Man Winter, Wandering Snowman, Fresno Nightcrawler) don't get the weather multiplier ours applied, and four cards gain an HP multiplier (Sciron and Gorilla x1.1, Yeti and Vampire Lord x1.5). Final Testament's Platinum and Galaxy values also follow his: 7.5 and 12.5.

| Card | HP was | HP now | ATK was | ATK now | HP ratio | ATK ratio |
|---|---|---|---|---|---|---|
| Abomination | 4,156 | 18,701 | 2,078 | 9,351 | 0.22 | 0.22 |
| Academy Student | 446 | 669 | 223 | 335 | 0.67 | 0.67 |
| Amaterasu | 3,564 | 4,277 | 1,782 | 2,139 | 0.83 | 0.83 |
| Bakunawa | 12,616 | 18,924 | 6,308 | 9,462 | 0.67 | 0.67 |
| Cave Goblin God | 4,531 | 5,437 | 2,266 | 2,719 | 0.83 | 0.83 |
| Community | 7,127 | 8,553 | 3,564 | 4,277 | 0.83 | 0.83 |
| Cthulu | 5,270 | 10,539 | 2,635 | 5,270 | 0.5 | 0.5 |
| Dionysus | 1,920 | 1,280 | 960 | 640 | 1.5 | 1.5 |
| Frank | 2,705 | 4,057 | 1,353 | 2,029 | 0.67 | 0.67 |
| Fresno Nightcrawler | 640 | 320 | 320 | 160 | 2.0 | 2.0 |
| Gorilla | 1,280 | 1,549 | 640 | 704 | 0.83 | 0.91 |
| Kid Gohan | 2,348 | 3,522 | 1,174 | 1,761 | 0.67 | 0.67 |
| Kraken | 9,198 | 11,957 | 4,599 | 5,979 | 0.77 | 0.77 |
| Loch Ness | 1,687 | 1,097 | 844 | 549 | 1.54 | 1.54 |
| Longmu | 4,629 | 9,257 | 2,315 | 4,629 | 0.5 | 0.5 |
| Loveland Frog | 1,039 | 2,078 | 520 | 1,039 | 0.5 | 0.5 |
| Memories | 8,781 | 10,537 | 4,391 | 5,269 | 0.83 | 0.83 |
| Nüwa | 4,733 | 5,409 | 2,367 | 2,705 | 0.88 | 0.88 |
| Old Man Winter | 240 | 160 | 120 | 80 | 1.5 | 1.5 |
| Santa Claws | 6,234 | 4,156 | 3,117 | 2,078 | 1.5 | 1.5 |
| Sarimanok | 12,616 | 18,924 | 6,308 | 9,462 | 0.67 | 0.67 |
| Sciron | 2,078 | 2,058 | 1,039 | 936 | 1.01 | 1.11 |
| Star Eater | 4,788 | 5,745 | 2,394 | 2,873 | 0.83 | 0.83 |
| The Rake | 6,747 | 8,771 | 3,374 | 4,386 | 0.77 | 0.77 |
| Tornado | 844 | 1,687 | 422 | 844 | 0.5 | 0.5 |
| Vampire Lord | 4,156 | 6,234 | 2,078 | 2,078 | 0.67 | 1.0 |
| Wandering Snowman | 1,559 | 1,039 | 780 | 520 | 1.5 | 1.5 |
| Wendigo | 2,530 | 2,868 | 1,265 | 1,434 | 0.88 | 0.88 |
| Xīfāng Báihü | 3,886 | 4,027 | 1,943 | 2,014 | 0.96 | 0.96 |
| Yeti | 6,234 | 9,351 | 3,117 | 3,117 | 0.67 | 1.0 |
| Fate Seamstress | 6,308 | 7,570 | 3,154 | 3,785 | 0.83 | 0.83 |
| Eclipseborn Luminant | 3,563 | 4,276 | 1,782 | 2,138 | 0.83 | 0.83 |
| Eonus | 4,942 | 5,930 | 2,471 | 2,965 | 0.83 | 0.83 |
| The Broken One | 3,886 | 4,663 | 1,943 | 2,332 | 0.83 | 0.83 |
| Durante | 7,121 | 4,747 | 3,561 | 2,374 | 1.5 | 1.5 |
| Ice King | 3,208 | 3,422 | 1,604 | 1,711 | 0.94 | 0.94 |

## Cards in his data but not ours (16)

Probably a newer update. Confirm each exists in game before adding it to the catalogue.

Accuser, Armin The Humble, Baba Yaga, Cave Goblin, Chronal, Epitome, Lyra, Mutant, Ouroboros, Sea Turtle, Something Funny, Supreme Frank, The Circus, The Huntsman, Three Wise Men, True Incarnation
