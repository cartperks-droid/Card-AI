# Verification queue

These are open rule and class questions, most important first. The daily task picks from here and adds new ones as they come up. Once a check is answered, move it to "Answered" with the date and result.

## Open
1. **Class spot-checks.** Next is toy (7 reviewed), then avian (3), dragon (5) and demon (7). Only check the *Reviewed* rows in `docs/classes.md` (`python -m card_engine.class_report` regenerates it).

2. **Odin's dodge cost.** Is it 20% of current HP (it can never die to normal attacks) or 20% of max HP (dead after 5 dodges)?
3. **Skipped turns and the stalemate.** Do skipped turns (e.g. Arcane Avian) count toward the 100-turn stalemate?
4. **Kira vs Noveau Riche.** Does Kira's doom kill Noveau Riche through its 3 lives?
5. **Parallax and status damage.** Does burn or poison kill Parallax without triggering "the enemy dies instead"?
6. **Border rarities.** RuPl was corrected to 1e7 and GaCrPl to 1e12, from the deck screenshots. Confirm with a stat readout of one RuPl or GaCrPl card.

## Answered
- 2026-10-02 **Buddha:**
  - If the enemy can't kill it, Buddha does nothing but revive the dead, one per turn in setup order, while staying in front.
  - With none dead, it heals; a heal that fully heals a card sends Buddha to the end of the deck.
  - Before the fix, revived cards jumped ahead of Buddha because of the "all full" move rule, which caused the endless loops.
  - The user's example matches: RuCrPl Malik beats Platinum Buddha + 3 borderless Inari.
  - Fixed in both engines (kernel 0.51).
- 2026-10-02 **Mother of Beasts** uses a stolen ability at once: the Priest's extra action right after the kill. Kernel 0.51.
- 2026-10-02 **Magical Elf Galaxy = 1 Toy:** correct.
- 2026-10-02 **End Times** model (one roll per step; chance dodges and criticals scaled): correct.
- 2026-10-02 **Toy Bear HP** at 1 and 2 fallen Toys: interpolated like ATK, ×(1 + fallen).
- 2026-10-02 **Buddha never heals itself.** It moves to the back once its heal fully heals another card, or every other card is full. Kernel 0.52.
- 2026-10-02 **Vampire Matron** lifesteal applies on direct attacks only. A card without its own drain (Black Plague) gets the granted %.
- 2026-10-02 **The 100-turn stalemate** counts within one interaction (pairing), not the whole battle. This matches the implementation.
- 2026-10-02 **Undead** (user scan): War and Bloody Mary are undead, Marrowclaw is not. All 25 are now confirmed.
- 2026-10-02 **Border rarities:** combined borders multiply (RuPl 1e7, GaCrPl 1e12). Evidence: the user's deck screenshots and the stats-sorted order.
- 2026-10-02 **Sable** takes the enemy card in play's ability *and disables it* on that card. Kernel 0.55.
