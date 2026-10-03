# Forms, weapons and random abilities (user videos, 2026-09-30)

Read from IMG_0327, IMG_0329, IMG_0330, IMG_0333 and IMG_0334. "?" marks text that was not fully legible.

## Transforms and awakenings (IMG_0327)

| Card | Before | After |
|---|---|---|
| Sun Wukong | 3,154 ATK, "Monkey King's Rage: transform when HP falls below 50%" | 6,308 ATK (x2 stats), same text, new art |
| Demonic Cultivator | "Dark Qi Manipulation: stats +30% for 2 turns, awaken on the 3rd turn" (Platinum: 4,853 after the first +30%) | "Dark Awakening: deal 2x damage, 30% lifesteal, stats +50% after a kill"; ATK 5,973 then 7,840 = base x 1.6 then x 2.1 (boosts add onto the base) |
| Immortal Cultivator | "Immortal Ascension: gain 30% stats for 2 turns and awaken on the third" (2,619) | User: takes half damage; stats +50% per kill |
| Naga | "Transform and heal when HP drops below 65%" | User: the transformed form leaves a curse on death dealing 25% max HP for 4 turns |
| Ultimate Brawler | "Divine Ascension: enemy's hit chance is squared ratio of damage to your max HP; on first death awaken instead" (6,201) | "Mastered Ascension: gain 1.5x stats, dodge every other attack" (9,302) |

## The Awakened One's weapons (IMG_0329, IMG_0342)

Base text: "Six Realms Staff: on entry, manifest 1 of 6 weapons, gaining its unique ability". In IMG_0342 the user's Malik (1,494,221 ATK) one-shots a deck of Awakened Ones (7,509), each rolling its own weapon.

| Weapon | Text |
|---|---|
| Twelve Devas Axe | Deals 150% increased damage |
| Shield of Ahimsa | Gain a shield every other turn and take 35% less damage |
| Vajra Short Sword | 40% chance to parry an attack, taking no damage and reflecting 75% of its damage, capped at 75% of current HP |
| War Scythe | On entry, attack the first 2 enemies, bypassing defensive and on-death abilities |
| Staff of Perfect Enlightenment | Deals 25% increased damage with a 25% chance to stun. Ignores defensive abilities |
| Great Nirvana Sword - Zero | On entry, gain the effects of 2 random Six Realms weapons (seen: Staff + Vajra, War Scythe + Vajra) |

## Astraeus (IMG_0333, IMG_0334, user)

User: each art is a separate card, all named Astraeus ("Constellar: gain an ability based on the card's art"). Every battle fixes each Astraeus's art (`arts` 1-7). DaddyDrago's engine would draw one on entry, but the user ruled that each art is a deterministic card, so `sim_js/search.ts` presets its Constellar ability. In the ability pool (Nuwa, Pandora, Glamour, Loki) each art is its own card.

| Art | Ability |
|---|---|
| Scorpio (scorpion on a purple sky) | Inflict poison on attack, dealing damage equal to your ATK each turn |
| Aquarius (blue-haired girl under a starry blue sky) | At turn start increase Max HP by 25%, or heal 30% if HP below half (user: heals at or below 50% HP) |
| Virgo (golden-haired woman holding wheat) | Gain 2 shields on entry |
| Gemini (twins) | Boost stats by 50% for every Astraeus in deck (2,560 -> 7,680 with four) |
| Sagittarius (archer) | Deal 2x damage to cards that would have evaded |
| Taurus (bull on a red background, user) | Boost damage and reduce damage taken as HP lowers, up to 2.5x |
| Cancer (crab, user) | Nullify damage below 100% of Max HP, reducing the threshold by 15% each time (each nullification) |

## Glamour (IMG_0330)

User: it changes at the start of its own turn. Read from the video (the user's Malik, 655,360 ATK, against a deck of Glamours, 7,680 each):

- At the start of each of its turns Glamour becomes a random card (Limited and seasonal cards included) and the new form's entry effects apply to its current stats: Serpent Mist 7,680 -> 23,040 (+7,680 per fallen ally), Toy Bear x2 (15,360), Darling +20% (9,216), Titan +25% (9,600), Terra's Aria x1.9 (3 fallen), The Sack +16% ATK, Hard Claws cut Malik to 393,216 (-40%), Eonus took 5% from Malik, The Awakened One manifested a weapon, The Broken One split into Joy and Sorrow (4,608 = 60%).
- A form with an entry attack attacks first (Slum Dweller reduced Malik twice: entry attack, then its normal attack).
- Stats carry over from form to form (Cthulu's +50% after Malik's self-hit stayed at 11,520 as Sabertooth Tiger).
- Once its ability was removed it kept showing new forms with "No Ability".

## Card texts confirmed in IMG_0330

- The Sack: "On entry, roll the D6 to randomly gain 1-50% damage, 1-50% max HP, and 1-30% damage reduction for the battle" - the three rolls are independent (shown "ATK +16% HP +9% DR ...").
- The Broken One / Sorrow: Joy and Sorrow each inherit 50% of max HP and 60% of damage, entering one after another.
- Cosmic Pop Star: steals 75% of the stats from the card behind it and banishes it; counterattacks.
- Walking Dead: survive 1 turn when dead; while dead, gain a turn of lifespan per kill.

## Nüwa summons (IMG_0336)

"Creation and Restoration: create a random card with your stats and place it at the end of your deck." Over 20 rounds of 4 Nüwa vs 4 Nüwa, summons included Prehistoric, Halloween, seasonal (Gingerbread Man) and boss cards; no Limited or Glamour cards appeared. Summons enter at Nüwa's 2,705 ATK, and their own entry and stat abilities apply on top (Abomination ×4 = 10,820; Titan ×1.25; Demonic Cultivator ×1.6). The full list is in oracle_observations.json (obs_20261001_nuwa_img0336).
