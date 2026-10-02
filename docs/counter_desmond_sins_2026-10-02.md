# Counter teams: Sable / Noveau Riche / Malik / Parallax (Galaxy Desmond + Vampire Matron)

**Enemy:** solved from the HP in your screenshot.
- Sable (Galaxy), Noveau Riche (GaCrPl), Malik (GaCrPl), Parallax (Platinum).
- Their Vampire Matron's tier isn't known; Galaxy is assumed.

**Availability model** (your rules, 2026-10-02):
- A card costs its expected rolls: card rarity × border rarity ÷ the chance of rolling during its weather.
- **Base cards:** 0.9, your usual rolling phase.
- **Storm and Snow:** 0.1.
- **Shroud, Aurora and Rapture:** 0.05 (an assumption).
- **Rare weathers:** 0.1 × the chance in your weather screenshot (Meteor Shower 16.42%, Time Storm 66.29%, Eclipse 9.07%, Virus 3.74%, Blood Rain 2.63%, Armageddon 1.33%, Manga 0.67%) × 0.75, because they last about 1.5 min rather than 2.
- **Owned cards** (your deck screenshots) cost nothing.
- **Cap:** any other card is limited to 7.8e10 rolls, the rarity of the lowest card shown (Sekhmet 78B), since rarer cards not shown probably aren't in your deck.

**Search:** `python -m card_engine.training.counter data/scenarios/2026-10-02_desmond_sins.json --max-rolls 7.8e10`.
- It first searches your owned cards alone, then everything within the cap: proposals drawn by obtainability, plus stat-targeted cheapest-border proposals, followed by local improvement. About 21,000 teams were evaluated; every one is in `data/scenarios/2026-10-02_desmond_sins.rows.jsonl`.
- **Scoring:** each team is simulated with you attacking first and with the enemy first.
- **Rules:** kernel 0.55 with today's fixes. Sable takes *and disables* the ability. Fate's retry is a second roll at the same chance; it no longer guarantees success.

| # | Lineup (in order) | Red | Blue | Win, you first | Win, enemy first | Not owned |
|---|---|---|---|---|---|---|
| 1 | AK4-ON1 (GaCrPl), Yeti (no border), **Odin** (no border), Stegosaurus (RuCr) | Neko (Galaxy) | Flame Wizard (Galaxy) | 100% | 100% | Yeti 1/500M, Odin 1/7.8M |
| 2 | AK4-ON1 (GaCrPl), Kira (Ga), **Odin** (no border), Ultimate Brawler (no border) | Neko (Galaxy) | Flame Wizard (Galaxy) | 100% | 100% | Odin, Ultimate Brawler |
| 3 | AK4-ON1 (GaCrPl), Hathor (Ga), **Odin** (no border), Divine Doctor (Cr) | Neko (Galaxy) | Flame Wizard (Galaxy) | 100% | 100% | Odin, Divine Doctor |
| 4 | AK4-ON1 (GaCrPl), Infected Maw (no border), **Odin** (no border), Malik (RuCr) | Neko (Galaxy) | Flame Wizard (Galaxy) | 99.6% | 99.2% | Infected Maw (no border), Odin |
| 5 | Malik (RuCrPl), Hathor (Ga), Kira (Ga), Noveau Riche (GaPl) | Disease (Crystal) | Flame Wizard (Galaxy) | 38% | 28% | all owned |
| 6 | AK4-ON1 (GaCrPl), Hathor (Ga), Kira (Ga), Deus Ex (RuPl) | Shrinemaiden (Galaxy) | Flame Wizard (Galaxy) | 25% | 25% | all owned |
| 7 | AK4-ON1 (GaCrPl), Yeti (no border), Hades (no border), Loki (no border) | Yggdrasil (Crystal) | Mirror Knight (Crystal) | 17% | 14% | Yeti, Hades, Loki |
| 8 | AK4-ON1 (GaCrPl), Kira (Ga), Priest (RuPl), Gideon (RuPl) | Myths (Crystal) | Guardian Angel (base) | 10% | 10% | all owned |

**Teams 1–4** rest on one unverified rule: Odin's dodge costs 20% of *current* HP, so normal attacks never kill him. If the cost is 20% of *max* HP, these teams collapse. It's the most valuable thing to test (`daily_checks/queue.md`).

**Teams 5–8** don't depend on Odin. Kira's doom does the heavy lifting: it's the only way through Malik's 13.5M HP. Hathor's shield and Deus Ex's dodges buy Kira's 2 turns, and Flame Wizard's burns finish Parallax. Kira's doom against Noveau Riche's 3 lives is also unverified.
