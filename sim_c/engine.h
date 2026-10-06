// C battle engine: DaddyDrago's battle rules (sim_js/vendor/.../battle-v2.label.ts, patched by sim_js/codemod.mjs),
// re-expressed with integer ids and flat state so the label search can copy a battle with one memcpy (user, 2026-10-03:
// "build it in C ... don't leave out any abilities"). Every function mirrors his function of the same name, in the same
// order of checks, so both engines meet the same chance points with the same probabilities; tests/test_kernel.py
// checks that they give identical results.
#pragma once
#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>
#include "build/tables.h"

#ifndef MAXC
#define MAXC 4096  // card slots per battle (team members, fallen and every card created in battle; Pandora battles
                   // have reached 836)
#endif
#ifndef MAXT
#define MAXT MAXC  // cards per team list
#endif
#define NOCARD (-1)
#define AO_UNDEFINED (-2)  // abilityOverride never set (his `undefined`)
#define AO_NULL (-1)       // abilityOverride set to null: no ability

#define FLAG_LIST(X) X(appearedOnField) X(armageddonLethalThisHit) X(awakened) X(banquetStolen) X(beyondGraveRevived) \
  X(blackBoxUsed) X(bloodlustFirstTurn) X(chaosTriple) X(chimericFaded) X(constellarGeminiApplied) X(dangerSense) \
  X(diesAfterAttack) X(divinationFired) X(divineBarrier) X(dodge) X(dodgeLethal) X(double_) X(eternalConfusion) \
  X(eternalDevotion) X(evadedThisHit) X(extraTurn) X(farmed) X(frozenSolitudeFirstTurnUsed) X(guardianAngelUsed) \
  X(hanged) X(insatiableAttack) X(kitchenDomain) X(laserCharged) X(lastStand) X(lightsWay) X(limitless) X(loser) \
  X(lotusReviveUsed) X(manaShield) X(mirrorImageReturned) X(moonlightUsed) X(naughtyListDrain) X(noRng) X(onBonusTurn) \
  X(ouroborosActive) X(paired) X(pandoraRolled) X(paradox) X(perseveranceBoosted) X(revealed) X(revived) X(sealed) \
  X(shapeshifterActive) X(sixRealmsRolled) X(slowed) X(stealChristmasUsed) X(suppressArmageddonOnce) X(suppressOnDeath) \
  X(transformed) X(undeadPractitioner) X(undyingActive) X(unholyActive) X(voidReady) X(wail) X(warScytheEntry) \
  X(warScytheEntryUsed) X(witchCurseStolen) X(worldCooldown) X(zeroWeaponsRolled)
#define COUNTER_LIST(X) X(ahimsaTurns) X(ascension) X(attacks) X(bindFatePair) X(bleed) X(blockHp) X(bloodlustBase) \
  X(cancerThreshold) X(cosmicRivalryDR) X(d8Reduction) X(damageTaken) X(death) X(defensiveManeuver) X(divinationMoves) \
  X(drop) X(extraTurns) X(finalTail) X(fireWorld) X(frostbite) X(fullMoon) X(gloryBaseDamage) X(gloryKills) X(grind) \
  X(healingMiracle) X(heardHits) X(heavenly) X(hiddenCurse) X(hiddenDepthsBonusDamage) X(hiddenDepthsBonusHp) \
  X(hpShield) X(interns) X(luminescentEvades) X(luminescentVeilGain) X(maelstrom) X(martialHits) X(masteredAscension) \
  X(normalDamage) X(normalMaxHp) X(ouroborosBonusDamage) X(ouroborosBonusHp) X(ouroborosBonusMaxHp) X(ouroborosTurns) \
  X(passion) X(perishTurns) X(persistence) X(poisonFlat) X(poisonPercent) X(poisonTurns) X(runFast) X(slowTurns) \
  X(slowed) X(snowbound) X(tail) X(telekinesis) X(thunder) X(toyCount) X(transcend) X(turnsPerTurn) X(undyingTurns) \
  X(unholyActivatedTurn) X(unholyLastTick) X(unholyTurns) X(upheaval) X(videoFrozen) X(void_) X(waterfowl) \
  X(weaknessTurns) X(worldCreation)
#define ENUM_ITEM(name) F_##name,
enum { FLAG_LIST(ENUM_ITEM) NFLAG };
#undef ENUM_ITEM
#define ENUM_ITEM(name) C_##name,
enum { COUNTER_LIST(ENUM_ITEM) NCOUNTER };
#undef ENUM_ITEM

typedef struct {
  uint64_t id;                // FNV-1a hash of his card id string (Hidden Blade keys its flag on it)
  int def;                    // definition index (tables.h)
  int team;                   // 0 Allies (side A), 1 Enemies (side B)
  double index;
  int border;                 // border id 1-16 (1 = none); his border list order
  double power, hp, maxHp, damage;
  double hpTaken;  // HP the card lost to the latest hit (Ruler of Humans; codemod.mjs)
  unsigned char entered, dead, boss;
  int identity;               // identityOverride: definition index, or NOCARD
  int abilityOverride;        // ability id, AO_NULL or AO_UNDEFINED
  int nbonus;                 // bonusAbilities (his undefined == empty)
  short bonus[6];
  double stunned, confused, burn, shield;
  unsigned char weakness, blind;
  unsigned char flag[NFLAG];
  double counter[NCOUNTER];
} Card;

typedef struct {
  double shielder, fate, flameWizard, phantom, berserker, synthHuman, endTimes, vampireMatron, stormSpirit,
    guardianAngel, executioner, mirrorKnight, finalTestament, fossils, composerCount, composerThreshold, noAbilities;
  unsigned char composerThresholdSet;  // `?? 1` tells undefined from 0
  int skillAura;                       // skillAuraName present (Erosion clears it); his debug text only
} Boosts;

#define MAXSTRUCK MAXC
typedef struct { int attacker; uint64_t target; } Struck;  // Hidden Blade: his flags['hiddenBlade:' + target id]

typedef struct {
  // scalars first: a battle's copy is these, then the used part of each list and of the cards (state_pack)
  int ncard, nteam[2], nfallen[2], nstruck;
  Boosts boosts[2];
  double turn;
  int moving;
  int team[2][MAXT], fallen[2][MAXT];
  Struck struck[MAXSTRUCK];
  Card card[MAXC];
} State;

// A battle's compact copy (only the cards and list entries in use), for the search's saved turns.
size_t state_packed_size(const State *s);
void state_pack(const State *s, void *out);
void state_unpack(const void *in, State *s);

// The loop's own counters at the start of a turn (his TurnCounters), saved with the state.
typedef struct { double turnsWithoutDeaths, lastDeathEpoch, deathEpoch; } Counters;

// Chance points, provided by the search (search.c): the same four calls as sim_js/search.ts.
typedef struct Runtime Runtime;
int chance_roll(Runtime *rt, int team, int cmp, double t);  // cmp: CMP_LT, CMP_LE, CMP_GT, CMP_GE
int chance_pickroll(Runtime *rt, int team, int n);
int chance_raw(Runtime *rt, int cmp, double t);
int chance_pick(Runtime *rt, int n);
enum { CMP_LT, CMP_LE, CMP_GT, CMP_GE };
// FNV-1a over his card id strings; it streams, so `parent + suffix` continues the parent's hash.
#define FNV_START 1469598103934665603ull
static inline uint64_t fnv(uint64_t h, const char *s) { for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; } return h; }
// More cards than MAXC / MAXT, recursion past MAXDEPTH, or more than MAXOPS engine operations in one playthrough: the
// search abandons the battle (an error, never a wrong answer).
_Noreturn void engine_overflow(Runtime *rt);

struct Runtime {
  State s;
  double deathEpoch;
  int depth;     // nesting of on_entry / deal_damage / resolve_deaths (engine.c MAXDEPTH)
  long ops;      // on_entry / deal_damage / death checks in this playthrough (engine.c MAXOPS)
  void *search;  // the search's tape (search.c)
};

// One battle from `start` (or resumed from a saved turn); returns 0 A wins, 1 B wins, 2 draw. on_turn is called at the
// start of every turn with the loop counters (the search saves the battle there).
int simulate(Runtime *rt, int max_turns, const Counters *resume, void (*on_turn)(Runtime *, const Counters *));
int rollContext_zero(Runtime *rt, int team, double *fate);
