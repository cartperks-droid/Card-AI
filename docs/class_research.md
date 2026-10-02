# Gameplay class research

Research date: **2026-09-27**. These are positive statements found in indexed Card RNG Wiki text, not an exhaustive or current game-engine class table. Direct Fandom retrieval was blocked by robots.txt; search results exposed page text and categories. The wiki is community-maintained, and several pages retain older stats or incomplete tables. **The user's current data and battle results take precedence.** No source stats, card order, or descriptions were overwritten.

`data/annotations/sourced_classes.json` contains **24 explicit positive annotations across 22 local cards** using the requested schema. Every row has status `wiki_supported_unverified_current`. Card IDs retain the original **1-based** local ordering. Missing annotations mean **unknown**, not false. In particular, do not fill a complete class bitmask by treating these incomplete lists as exhaustive.

## Explicit positive memberships

| Class | Local cards supported by the retrieved text | Evidence |
| --- | --- | --- |
| dragon | **17 Greedy Belly; 60 Fafnir; 61 Ragon; 112 River Dragon; 134 Dōng Fāng Qīng Lóng; 210 Raze The Destroyer; 238 Ragon & Fafnir** | Dragon King's trivia identifies these as affected dragons. It introduces them as examples, so this is not an exhaustive list. “Raze” is mapped to the local Raze The Destroyer name. [Dragon King](https://card-rng.fandom.com/wiki/Dragon_King) |
| undead | **49 Hades; 66 Mummy; 76 Anubis** | Each card's own page carries an Undead category. This is page-category evidence, weaker than a direct statement about Michael's damage calculation. [Hades](https://card-rng.fandom.com/wiki/Hades), [Mummy](https://card-rng.fandom.com/wiki/Mummy), [Anubis](https://card-rng.fandom.com/wiki/Anubis) |
| demon | **122 Heavenly Demon; 211 Sable The Envious** | Each card's own page carries a Demons category. [Heavenly Demon](https://card-rng.fandom.com/wiki/Heavenly_Demon), [Sable](https://card-rng.fandom.com/wiki/Sable%2C_The_Envious) |
| avian | **31 Phoenix** | Its own page carries an Avians category. [Phoenix](https://card-rng.fandom.com/wiki/Phoenix) |
| friendship | **109 A0-ON1; 110 AK4-ON1** | A0-ON1's trivia identifies the pair and says the bonus includes itself, with an additive combined increase. This supports those two positives, not the claim that no other Friendship card can exist. [A0-ON1](https://card-rng.fandom.com/wiki/A0-ON1) |
| rng | **1 Archer; 15 Arcane Avian; 22 Three-Legged Golden Crow; 27 Frankenstein; 83 Gambler** | The ability table explicitly marks their ability types as RNG. A percentage in a card description alone was not used to assign this class. [Abilities](https://card-rng.fandom.com/wiki/Abilities) |
| rng | **31 Phoenix; 122 Heavenly Demon; 136 Shennong** | Each card's own page carries an RNG Cards category. [Phoenix](https://card-rng.fandom.com/wiki/Phoenix), [Heavenly Demon](https://card-rng.fandom.com/wiki/Heavenly_Demon), [Shennong](https://card-rng.fandom.com/wiki/Shennong) |
| rng | **37 Pandora** | Its trivia specifically links its RNG classification to Black Cat's increased damage. This is direct gameplay-target evidence. [Pandora](https://card-rng.fandom.com/wiki/Pandora) |

Dōng Fāng Qīng Lóng's own page independently describes it as eligible for Dragon King and Taoist, supporting the distinction between a gameplay class and a pack. [Dōng Fāng Qīng Lóng](https://card-rng.fandom.com/wiki/D%C5%8Dng_F%C4%81ng_Q%C4%ABng_L%C3%B3ng)

## Useful leads that are not asserted memberships

**Toy:** the Christmas card table documents base/awakened forms for Toy Bear, Toy Car, Toy Jack-in-the-Box, and Toy Nutcracker, with toy-dependent awakened abilities. These correspond to local **261 Toy Bear, 262 Toy Car, 263 The Jack-in-the-Box, 264 Toy Nutcracker**; the Jack-in-the-Box mapping is an alias supported by matching ability text. This strongly suggests the expected set, but the retrieved material does not supply a formal membership list. Those four remain contextual candidates, not explicit positive rows in `sourced_classes.json`. [Cards: Christmas section](https://card-rng.fandom.com/wiki/Cards#Christmas_Event)

Magical Elf's description requires four distinct toys for awakening. The support list also establishes that Avian King targets birds, Imp targets demons, and Dragon King targets dragons. It does not enumerate all matching cards. [Auras (Support Cards)](https://card-rng.fandom.com/wiki/Auras_%28Support_Cards%29)

**Dev:** the support overview mentions Dev cards but does not provide a membership list. The Binder page discusses excluding dev cards from its display; that does not identify which imported cards belong to the class. No `dev` positives are added. A card referring to a developer or using an avatar is not sufficient gameplay evidence. [Auras (Support Cards)](https://card-rng.fandom.com/wiki/Auras_%28Support_Cards%29), [Binder](https://card-rng.fandom.com/wiki/Binder)

**Other plausible members:** no class was assigned merely because a name, mythology, artwork, or navigation list suggested it. For example, a page's global navigation lists many unrelated cards; its Undead or Demons category applies to that page's subject only. Phoenix's category does not automatically classify every other phoenix, and a missing bird category does not establish that Arcane Avian is excluded.

## Integration and next verification

Keep three separate evidence types: user-confirmed membership, wiki-supported membership awaiting current-game confirmation, and visual/name/context candidates. An art tag is not a simulator class. The JSON supplies only the second type and must not promote a candidate to a confirmed rule.

Use the Training Dummy's unrestricted enemy selection to test uncertain target interactions once a suitable lit card is available on the user's side. Michael versus a proposed undead, Siegfried versus a proposed dragon, and Black Cat versus a proposed RNG card can establish gameplay applicability with a recorded baseline. A bonus that affects all allies needs a controlled support comparison; a single screenshot with unknown supports is insufficient. The exact matchups should be chosen from the user's confirmed available inventory rather than requesting unowned cards.

No authoritative full membership list was recovered for all eight requested classes. The explicit positives above reduce the unknown set while preserving that limitation.
