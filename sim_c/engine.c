// C battle engine: see engine.h. Function names and the order of every check follow his battle-v2.label.ts; comments
// name the few places where C needs explicit code for JavaScript semantics (NaN in Math.max/min, `|| 0`, `??`).
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "engine.h"

#define S (&rt->s)
#define CARD(i) (&rt->s.card[i])
#define FL(i, f) (rt->s.card[i].flag[F_##f])
#define CT(i, c) (rt->s.card[i].counter[C_##c])
#define OTHER(team) (1 - (team))

// ---- JavaScript number semantics ----
static double jmax(double a, double b) { return (isnan(a) || isnan(b)) ? NAN : (a > b ? a : (b > a ? b : (a == 0 && signbit(a) ? b : a))); }
static double jmin(double a, double b) { return (isnan(a) || isnan(b)) ? NAN : (a < b ? a : (b < a ? b : (a == 0 && !signbit(a) ? b : a))); }
static int truthy(double x) { return x != 0 && !isnan(x); }
static double or0(double x) { return truthy(x) ? x : 0; }           // `x || 0`
static double orv(double x, double y) { return truthy(x) ? x : y; } // `x || y`
static double jmod(double a, double b) { return fmod(a, b); }       // `%` (sign of the dividend, like fmod)

// ---- team lists ----
static int team_len(Runtime *rt, int t) { return S->nteam[t]; }
static int at(Runtime *rt, int t, int i) { return i < S->nteam[t] ? S->team[t][i] : NOCARD; }
static int active(Runtime *rt, int t) { return at(rt, t, 0); }
static int index_of(const int *list, int n, int card) { for (int i = 0; i < n; i++) if (list[i] == card) return i; return -1; }
static void list_remove_at(int *list, int *n, int i) { memmove(list + i, list + i + 1, sizeof(int) * (*n - i - 1)); (*n)--; }
static void list_push(int *list, int *n, int card) { if (*n >= MAXT) engine_overflow(NULL); list[(*n)++] = card; }
static void list_unshift(int *list, int *n, int card) { if (*n >= MAXT) engine_overflow(NULL); memmove(list + 1, list, sizeof(int) * *n); list[0] = card; (*n)++; }
static int new_card(Runtime *rt) { if (S->ncard >= MAXC) engine_overflow(rt); return S->ncard++; }

// ---- identity and abilities (his definition / effectiveCardName / ability / abilityNames) ----
static int effective_name(Runtime *rt, int c) { return c == NOCARD ? NOCARD : (CARD(c)->identity != NOCARD ? CARD(c)->identity : CARD(c)->def); }

static int ability(Runtime *rt, int c) {
  if (c == NOCARD) return AO_NULL;
  Card *card = CARD(c);
  if (card->abilityOverride != AO_UNDEFINED) return card->abilityOverride;
  int name = effective_name(rt, c);
  if (card->identity != NOCARD && name == D_LONGMU) return AO_NULL;
  int a = DEF_ABILITY[name];
  if (a < 0) a = DEF_ABILITY[card->def];
  return a < 0 ? AO_NULL : a;
}

// abilityNames: [ability, ...bonusAbilities], falsy dropped, duplicates removed, first occurrence kept.
static int ability_names(Runtime *rt, int c, int *out) {
  int n = 0;
  if (c == NOCARD) return 0;
  int own = ability(rt, c);
  if (own >= 0) out[n++] = own;
  Card *card = CARD(c);
  for (int i = 0; i < card->nbonus; i++) {
    int a = card->bonus[i], seen = 0;
    if (a < 0) continue;
    for (int j = 0; j < n; j++) if (out[j] == a) seen = 1;
    if (!seen) out[n++] = a;
  }
  return n;
}

static int names_include(Runtime *rt, int c, int a) {
  int names[8], n = ability_names(rt, c, names);
  for (int i = 0; i < n; i++) if (names[i] == a) return 1;
  return 0;
}

// activeBonusAbilities: Pandora's Box and Heroes hold their gained abilities; Six Realms Staff's Zero holds two weapons.
static int active_bonus(Runtime *rt, int c, int *out) {
  Card *card = CARD(c);
  int root = DEF_ABILITY[card->def], own = ability(rt, c);
  if ((root == A_PANDORA_S_BOX || root == A_HEROES) && own == root) { memcpy(out, card->bonus, sizeof(int) * 0); for (int i = 0; i < card->nbonus; i++) out[i] = card->bonus[i]; return card->nbonus; }
  if (root == A_SIX_REALMS_STAFF && own == A_GREAT_NIRVANA_SWORD_ZERO) { for (int i = 0; i < card->nbonus; i++) out[i] = card->bonus[i]; return card->nbonus; }
  return 0;
}

static int has_ability(Runtime *rt, int c, int name) {
  if (c == NOCARD) return 0;
  Card *card = CARD(c);
  if (card->dead || card->flag[F_sealed]) return 0;
  int locked = or0(S->boosts[card->team].noAbilities) > 0;
  int friendship = name == A_FRIENDSHIP && DEF_ABILITY[card->def] == A_FRIENDSHIP;
  if (locked && !friendship) return 0;
  int opposing = active(rt, OTHER(card->team));
  int matched = names_include(rt, c, name);
  if (!matched && ability(rt, c) == A_JEALOUSY && opposing != NOCARD && effective_name(rt, opposing) != D_AMENHOTEP)
    matched = names_include(rt, opposing, name);
  if (!matched) return 0;
  if (opposing != NOCARD && ability(rt, opposing) == A_JEALOUSY && effective_name(rt, c) != D_AMENHOTEP) return 0;
  for (int t = 0; t < 2; t++) {
    int a = active(rt, t);
    if (a != NOCARD && !CARD(a)->dead && !CARD(a)->flag[F_sealed] && names_include(rt, a, A_HONOR) && name != A_HONOR) return 0;
  }
  double endTimes = S->boosts[OTHER(card->team)].endTimes;
  if (truthy(endTimes) && chance_raw(rt, CMP_LT, endTimes / 100)) return 0;
  return 1;
}

static int resolved_ability(Runtime *rt, int c) {
  int raw = ability(rt, c);
  if (c == NOCARD || raw < 0) return raw;
  int opposing = active(rt, OTHER(CARD(c)->team));
  if (raw == A_JEALOUSY && opposing != NOCARD && effective_name(rt, opposing) != D_AMENHOTEP) return ability(rt, opposing);
  if (opposing != NOCARD && ability(rt, opposing) == A_JEALOUSY && effective_name(rt, c) != D_AMENHOTEP) return AO_NULL;
  return raw;
}

int rollContext_zero(Runtime *rt, int team, double *fate) {
  int a = active(rt, 0), e = active(rt, 1);
  int zero = has_ability(rt, a, A_UNLUCKY) || has_ability(rt, e, A_UNLUCKY);
  if (!zero) { int f = active(rt, team); zero = f != NOCARD && CARD(f)->flag[F_noRng]; }
  *fate = or0(S->boosts[team].fate) / 100;
  return zero;
}

// ---- small helpers ----
static int alive(Runtime *rt, int c) { return c != NOCARD && CARD(c)->hp > 0 && !CARD(c)->dead; }
static void boost_stats(Runtime *rt, int c, double mult) { Card *k = CARD(c); k->damage *= mult; k->maxHp *= mult; k->hp *= mult; }
static void steal_stats(Runtime *rt, int from, int to, double fraction) {
  Card *f = CARD(from), *t = CARD(to);
  double d = jmax(0, f->damage * fraction), m = jmax(0, f->maxHp * fraction), h = jmax(0, f->hp * fraction);
  f->damage = jmax(0, f->damage - d);
  f->maxHp = jmax(1, f->maxHp - m);
  f->hp = jmax(0, jmin(f->maxHp, f->hp - h));
  t->damage += d; t->maxHp += m; t->hp += h;
}
static int status_protected(Runtime *rt, int team) {
  for (int i = 0; i < S->nteam[team]; i++) if (has_ability(rt, S->team[team][i], A_PROTECTION_OF_GODS)) return 1;
  return 0;
}
static int luminescent_holder(Runtime *rt, int team) {
  for (int i = 0; i < S->nteam[team]; i++) { int c = S->team[team][i]; if (alive(rt, c) && has_ability(rt, c, A_LUMINESCENT_VEIL)) return c; }
  return NOCARD;
}
static int can_receive_protection(Runtime *rt, int c) { return effective_name(rt, c) != D_JUDGMENT_DAY; }
static void clear_skill_aura(Runtime *rt, int team) {
  Boosts *b = &S->boosts[team], keep = *b;
  memset(b, 0, sizeof *b);
  b->fossils = or0(keep.fossils);
  b->composerCount = keep.composerCount;
  b->composerThreshold = keep.composerThreshold;
  b->composerThresholdSet = keep.composerThresholdSet;
  b->noAbilities = keep.noAbilities;
}
static int water_shield(Runtime *rt, int team, int target) {
  for (int i = 0; i < S->nteam[team]; i++) { int c = S->team[team][i]; if (c != target && has_ability(rt, c, A_WATER_SHIELD_OF_XUANWU)) return c; }
  return NOCARD;
}
static void reset_combat_stats(Runtime *rt, int c) {
  Card *k = CARD(c);
  double nd = k->counter[C_normalDamage], nm = k->counter[C_normalMaxHp];
  if (nd > 0) k->damage = nd;
  if (nm > 0) { k->maxHp = nm; k->hp = jmin(k->hp, k->maxHp); }
}
static void clear_statuses(Runtime *rt, int c) {
  Card *k = CARD(c);
  k->stunned = 0; k->confused = 0; k->burn = 0; k->weakness = 0; k->blind = 0;
  k->counter[C_bleed] = 0; k->counter[C_frostbite] = 0; k->counter[C_videoFrozen] = 0;
  k->counter[C_poisonFlat] = 0; k->counter[C_poisonPercent] = 0; k->counter[C_weaknessTurns] = 0;
}
static void clear_card_state(Card *k) {
  k->stunned = k->confused = k->burn = k->shield = 0; k->weakness = k->blind = 0;
  memset(k->flag, 0, sizeof k->flag); memset(k->counter, 0, sizeof k->counter);
}
static int primary_border(int border) {  // 4 Galaxy, 3 Ruby, 2 Crystal, 1 Platinum, 0 none
  const char *code = BORDER_CODES[border];
  if (strstr(code, "Ga")) return 4;
  if (strstr(code, "Ru")) return 3;
  if (strstr(code, "Cr")) return 2;
  if (strstr(code, "Pl")) return 1;
  return 0;
}
static double border_tier(Runtime *rt, int c) { static const double t[5] = {0, 10, 20, 25, 30}; return t[primary_border(CARD(c)->border)]; }
static double toy_bear_awakened_multiplier(Runtime *rt, int c, int fallenToys) {
  static const double current_of[5] = {1, 4, 16, 32, 64}, ladder[5] = {1, 4, 16, 32, 64};
  double current = current_of[primary_border(CARD(c)->border)];
  int start = 0;
  for (int i = 0; i < 5; i++) if (ladder[i] == current) { start = i; break; }
  int k = start + fallenToys; if (k > 4) k = 4;
  return ladder[k] / current;
}
static double constellar_taurus_factor(Runtime *rt, int c) {
  Card *k = CARD(c);
  if (k->maxHp <= 0) return 2.5;
  return jmin(2.5, 1 + (1 - jmax(0, k->hp) / k->maxHp) * 1.5);
}
#define ID_FORMAT(k, ...) do { char text_[256]; snprintf(text_, sizeof text_, __VA_ARGS__); (k)->id = fnv(FNV_START, text_); } while (0)

// ---- forward declarations (his functions call each other recursively) ----
static double deal_damage(Runtime *rt, int attacker, int target, double mult, int bypass);
static void resolve_deaths(Runtime *rt);
static void on_entry(Runtime *rt, int c);
// His engine recurses through these three (deaths trigger entries trigger damage...); where JS would throw "Maximum
// call stack size exceeded", the C engine abandons the battle before its own stack runs out.
#ifndef MAXDEPTH
#define MAXDEPTH 2000
#endif
static void enter(Runtime *rt) { if (++rt->depth > MAXDEPTH) engine_overflow(rt); }

// withAbility: run with the card's ability temporarily replaced.
#define WITH_ABILITY(c, name, stmt) do { int prev_ = CARD(c)->abilityOverride; CARD(c)->abilityOverride = (name); stmt; CARD(c)->abilityOverride = prev_; } while (0)

static int random_battle_card(Runtime *rt) { return RANDOM_CARD_POOL[chance_pick(rt, NRANDOM_CARD_POOL)]; }
static int random_constellar(Runtime *rt) { return CONSTELLAR_POOL[chance_pick(rt, NCONSTELLAR_POOL)]; }
static int random_creatable(Runtime *rt) { return NUWA_POOL[chance_pick(rt, NNUWA_POOL)]; }

static int resolve_pandora_gained(Runtime *rt, int c, int name) {
  if (name == A_CONSTELLAR) return random_constellar(rt);
  if (name == A_THE_UNDERWORLD) {
    int team = CARD(c)->team;
    for (int i = S->nfallen[team] - 1; i >= 0; i--) {  // [...fallen].reverse().flatMap(abilityNames).find(...)
      int names[8], n = ability_names(rt, S->fallen[team][i], names);
      for (int j = 0; j < n; j++) if (names[j] != A_THE_UNDERWORLD && names[j] != A_PANDORA_S_BOX) return names[j];
    }
  }
  return name;
}

// cloneAtFraction: {...source} with scaled stats, fresh status, flags {paired}, counters {normalDamage}.
static int clone_at_fraction(Runtime *rt, int source, double fraction, double serial) {
  int n = new_card(rt);
  Card *k = CARD(n), *src = CARD(source);
  *k = *src;
  ID_FORMAT(k, "%s:copy:%g:%s", src->team ? "Enemies" : "Allies", serial, DEF_NAME[src->def]);
  k->hp = src->hp * fraction; k->maxHp = src->maxHp * fraction; k->damage = src->damage * fraction; k->power = src->power * fraction;
  k->entered = 0; k->dead = 0; k->identity = NOCARD;
  clear_card_state(k);
  k->flag[F_paired] = 1;
  k->counter[C_normalDamage] = src->damage * fraction;
  return n;
}

static void perform_entry_attack(Runtime *rt, int c, double mult, int all_enemies) {
  int enemy_team = OTHER(CARD(c)->team), first = active(rt, enemy_team);
  if (first == NOCARD || !alive(rt, c)) return;
  double dealt = deal_damage(rt, c, first, mult, 0);
  if (all_enemies && dealt > 0)
    for (int i = 1; i < S->nteam[enemy_team]; i++) { Card *t = CARD(S->team[enemy_team][i]); t->hp -= jmin(t->hp, dealt); }
  resolve_deaths(rt);
}

static void on_entry_(Runtime *rt, int c) {
  Card *card = CARD(c);
  if (card->entered || !alive(rt, c)) return;
  card->entered = 1;
  card->flag[F_appearedOnField] = 1;
  int team = card->team, enemy_team = OTHER(team);
  int enemy = active(rt, enemy_team);
  if (enemy == NOCARD) return;
  Card *e = CARD(enemy);

  if (enemy != c && has_ability(rt, enemy, A_DESIRE)) steal_stats(rt, c, enemy, 0.1);
  if (enemy != c && has_ability(rt, enemy, A_COSMIC_MAW)) steal_stats(rt, c, enemy, 0.2);
  if (enemy != c && e->flag[F_awakened] && has_ability(rt, enemy, A_POP_UP_IMPRESSION) && !status_protected(rt, team))
    card->confused = jmax(card->confused, orv(e->counter[C_toyCount], 1));

  int name = resolved_ability(rt, c);
  if (name < 0 || !has_ability(rt, c, name)) return;

  if (name == A_THE_D8) {
    card->damage *= 1 + (1 + chance_pickroll(rt, team, 50)) / 100.0;
    double hpFactor = 1 + (1 + chance_pickroll(rt, team, 50)) / 100.0;
    card->maxHp *= hpFactor;
    card->hp *= hpFactor;
    card->counter[C_d8Reduction] = (1 + chance_pickroll(rt, team, 30)) / 100.0;
  }

  if (name == A_PANDORA_S_BOX && !card->flag[F_pandoraRolled]) {
    card->flag[F_pandoraRolled] = 1;
    int chosen[2], n = 0, attempts = 0;
    while (n < 2 && attempts++ < 100) {
      int raw = PANDORA_POOL[chance_pick(rt, NPANDORA_POOL)];
      int gained = resolve_pandora_gained(rt, c, raw);
      if (gained != A_PANDORA_S_BOX && !(n > 0 && chosen[0] == gained)) chosen[n++] = gained;
    }
    card->nbonus = n;
    for (int i = 0; i < n; i++) card->bonus[i] = chosen[i];
    for (int i = 0; i < n; i++) WITH_ABILITY(c, chosen[i], { CARD(c)->entered = 0; on_entry(rt, c); });
    CARD(c)->entered = 1;
    return;
  }

  if (name == A_HEROES) {
    int chosen[2], n = 0;
    for (int i = 0; i < S->nfallen[team] && i < 2; i++) {
      Card *f = CARD(S->fallen[team][i]);
      if (f->def == card->def || f->def == D_LEGENDS) continue;
      int a = DEF_ABILITY[f->def];
      if (a < 0) continue;
      if (!(n > 0 && chosen[0] == a)) chosen[n++] = a;  // [...new Set(chosen)]
    }
    if (n) { card->nbonus = n; for (int i = 0; i < n; i++) card->bonus[i] = chosen[i]; }
    int gained[6], g = active_bonus(rt, c, gained);
    for (int i = 0; i < g; i++) WITH_ABILITY(c, gained[i], { CARD(c)->entered = 0; on_entry(rt, c); });
    CARD(c)->entered = 1;
    return;
  }

  if (name == A_CONSTELLAR) {
    card->abilityOverride = random_constellar(rt);
    name = ability(rt, c);
    if (name == A_CONSTELLARGEMINI && !card->flag[F_constellarGeminiApplied]) {
      card->flag[F_constellarGeminiApplied] = 1;
      int count = 0;
      for (int i = 0; i < S->nteam[team]; i++) if (CARD(S->team[team][i])->def == D_ASTRAEUS) count++;
      boost_stats(rt, c, 1 + count * 0.5);
    }
    card->entered = 0;
    on_entry(rt, c);
    return;
  }

  if (name == A_THE_UNDERWORLD) {
    int copied = AO_NULL;
    for (int i = S->nfallen[team] - 1; i >= 0 && copied < 0; i--) {
      int a = ability(rt, S->fallen[team][i]);
      if (a >= 0 && a != A_THE_UNDERWORLD) copied = a;
    }
    if (copied >= 0) {
      card->abilityOverride = copied;
      card->entered = 0;
      on_entry(rt, c);
      return;
    }
  }

  if (name == A_SIX_REALMS_STAFF && !card->flag[F_sixRealmsRolled]) {
    static const int weapons[6] = {A_TWELVE_DEVAS_AXE, A_VAJRA_SHORT_SWORD, A_STAFF_OF_PERFECT_ENLIGHTENMENT,
                                   A_SHIELD_OF_AHIMSA, A_WAR_SCYTHE, A_GREAT_NIRVANA_SWORD_ZERO};
    card->flag[F_sixRealmsRolled] = 1;
    card->abilityOverride = weapons[chance_pick(rt, 6)];
    card->entered = 0;
    on_entry(rt, c);
    return;
  }

  if (name == A_GREAT_NIRVANA_SWORD_ZERO && !card->flag[F_zeroWeaponsRolled]) {
    static const int pool[5] = {A_TWELVE_DEVAS_AXE, A_VAJRA_SHORT_SWORD, A_STAFF_OF_PERFECT_ENLIGHTENMENT, A_SHIELD_OF_AHIMSA, A_WAR_SCYTHE};
    card->flag[F_zeroWeaponsRolled] = 1;
    int first_index = chance_pick(rt, 5), remaining[4], r = 0;
    for (int i = 0; i < 5; i++) if (i != first_index) remaining[r++] = pool[i];
    int first = pool[first_index], second = remaining[chance_pick(rt, 4)];
    card->nbonus = 2; card->bonus[0] = first; card->bonus[1] = second;
    int gained[2] = {first, second};
    for (int i = 0; i < 2; i++) WITH_ABILITY(c, gained[i], { CARD(c)->entered = 0; on_entry(rt, c); });
    CARD(c)->entered = 1;
    return;
  }

  switch (name) {
    case A_BIND_FATE: {
      int first_two[2], n = 0;
      for (int i = 0; i < S->nteam[enemy_team] && n < 2; i++) if (alive(rt, S->team[enemy_team][i])) first_two[n++] = S->team[enemy_team][i];
      if (n == 2) {
        double pair = S->turn * 1000 + jmax(1, card->index);
        CT(first_two[0], bindFatePair) = pair;
        CT(first_two[1], bindFatePair) = pair;
      }
      break;
    }
    case A_OUROBOROS:
      if (!card->flag[F_ouroborosActive]) {
        double sd = 0, sm = 0, sh = 0;
        for (int t = 0; t < 2; t++) {
          for (int i = 0; i < S->nteam[t]; i++) {
            int o = S->team[t][i];
            if (o == c || !alive(rt, o)) continue;
            Card *k = CARD(o);
            double od = k->damage, om = k->maxHp, oh = k->hp;
            k->damage = jmax(0, od * 0.95);
            k->maxHp = jmax(1, om * 0.95);
            k->hp = jmax(0, jmin(k->maxHp, oh * 0.95));
            sd += od - k->damage; sm += om - k->maxHp; sh += oh - k->hp;
          }
        }
        card->damage += sd; card->maxHp += sm; card->hp += sh;
        card->counter[C_ouroborosBonusDamage] = sd;
        card->counter[C_ouroborosBonusMaxHp] = sm;
        card->counter[C_ouroborosBonusHp] = sh;
        card->counter[C_ouroborosTurns] = 3;
        card->flag[F_ouroborosActive] = 1;
      }
      break;
    case A_PERSEVERANCE:
      if (!card->flag[F_perseveranceBoosted]) { card->flag[F_perseveranceBoosted] = 1; card->maxHp *= 100; card->hp *= 100; }
      break;
    case A_CONSTELLARVIRGO: card->counter[C_hpShield] = or0(card->counter[C_hpShield]) + card->maxHp * 2; break;
    case A_CONSTELLARGEMINI:
      if (!card->flag[F_constellarGeminiApplied]) {
        card->flag[F_constellarGeminiApplied] = 1;
        int count = 0;
        for (int i = 0; i < S->nteam[team]; i++) if (CARD(S->team[team][i])->def == D_ASTRAEUS) count++;
        boost_stats(rt, c, 1 + count * 0.5);
      }
      break;
    case A_GATHERING: card->damage *= pow(1.5, S->nteam[team] + S->nfallen[team]); break;
    case A_REMEMBRANCE: { int count = S->nfallen[team]; if (count) boost_stats(rt, c, pow(1.5, count)); break; }
    case A_FRIENDSHIP: {
      int seen[S->nteam[team] + S->nfallen[team] + 1], unique = 0;
      for (int pass = 0; pass < 2; pass++) {
        const int *list = pass ? S->fallen[team] : S->team[team];
        int n = pass ? S->nfallen[team] : S->nteam[team];
        for (int i = 0; i < n; i++) {
          if (ability(rt, list[i]) != A_FRIENDSHIP) continue;
          int d = CARD(list[i])->def, dup = 0;
          for (int j = 0; j < unique; j++) if (seen[j] == d) dup = 1;
          if (!dup) seen[unique++] = d;
        }
      }
      if (unique > 0) boost_stats(rt, c, 1 + unique * 0.4);
      break;
    }
    case A_HUMANITY_S_SPIRIT: { int count = S->nfallen[team]; if (count) boost_stats(rt, c, 1 + count * 0.3); break; }
    case A_PERFORATING_MIST: {
      double sum = 0;
      for (int i = 0; i < S->nfallen[team]; i++) sum += CARD(S->fallen[team][i])->damage;
      if (sum > 0) card->damage += sum;
      break;
    }
    case A_BEYOND_COMPREHENSION:
      if (!status_protected(rt, e->team)) { e->flag[F_eternalConfusion] = 1; e->confused = jmax(e->confused, 1); }
      break;
    case A_DANCE_OF_DISCORD: {
      int *deck = S->team[enemy_team], len = S->nteam[enemy_team];
      if (len >= 2) {
        int fi = chance_pick(rt, len), si = chance_pick(rt, len - 1);
        if (si >= fi) si += 1;
        Card *f = CARD(deck[fi]), *s2 = CARD(deck[si]);
        double t;
        t = f->damage; f->damage = s2->damage; s2->damage = t;
        t = f->maxHp; f->maxHp = s2->maxHp; s2->maxHp = t;
        double fh = jmin(s2->hp, s2->maxHp), sh = jmin(f->hp, f->maxHp);
        f->hp = fh; s2->hp = sh;
        boost_stats(rt, deck[fi], 0.85);
        boost_stats(rt, deck[si], 0.85);
        int tmp = deck[fi]; deck[fi] = deck[si]; deck[si] = tmp;
      }
      break;
    }
    case A_SNOWSCAPE: {
      if (status_protected(rt, e->team)) break;
      int roll = chance_pickroll(rt, team, 3);
      if (roll <= 0) e->counter[C_frostbite] = jmax(or0(e->counter[C_frostbite]), 3);
      else if (roll == 1) { e->flag[F_slowed] = 1; e->counter[C_slowTurns] = jmax(or0(e->counter[C_slowTurns]), 3); e->counter[C_slowed] = 0; }
      else e->stunned = jmax(e->stunned, 3);
      break;
    }
    case A_SPOOK:
      if ((DEF_SETS[e->def] & SET_AVIAN) && !status_protected(rt, e->team)) e->confused = jmax(e->confused, 3);
      break;
    case A_PERISH:
      if (!status_protected(rt, e->team)) e->stunned = jmax(e->stunned, 1);
      card->counter[C_perishTurns] = 3;
      break;
    case A_DESIRE: break;
    case A_COSMIC_MAW: steal_stats(rt, enemy, c, 0.2); break;
    case A_SAP:
      if (chance_roll(rt, team, CMP_GT, 1 - card->damage / e->damage)) { card->damage += e->damage * 0.5; e->damage *= 0.5; }
      break;
    case A_HAUNT: {
      double dl = card->damage * 0.35, hl = card->maxHp * 0.35;
      e->damage = jmax(0, e->damage - dl);
      e->maxHp = jmax(1, e->maxHp - hl);
      e->hp = jmax(0, jmin(e->maxHp, e->hp - hl));
      break;
    }
    case A_HEX: e->flag[F_noRng] = 1; break;
    case A_ORDER_OF_THE_COSMOS: S->boosts[enemy_team].noAbilities = 3; break;
    case A_MIND_RIFT: if (card->damage > e->damage / 4) e->confused = 3; break;
    case A_AM_I_BEAUTIFUL: e->confused = 2; break;
    case A_GOD_OF_TRICKERY: {
      int random_card = random_battle_card(rt);
      e->identity = random_card;
      e->abilityOverride = AO_UNDEFINED;
      card->identity = e->def;
      card->abilityOverride = AO_UNDEFINED;
      break;
    }
    case A_FIRE_WORLD: for (int i = 0; i < S->nteam[enemy_team]; i++) CARD(S->team[enemy_team][i])->burn = 3; break;
    case A_BOOK_OF_DEATH: e->counter[C_death] = 2; break;
    case A_EROSION: if (chance_roll(rt, team, CMP_LT, 0.5)) clear_skill_aura(rt, enemy_team); break;
    case A_DIVINATION: card->counter[C_divinationMoves] = 5; break;
    case A_CREATION_AND_RESTORATION: {
      int def = random_creatable(rt);
      int n = new_card(rt);
      Card *k = CARD(n);
      *k = *CARD(c);
      card = CARD(c);
      ID_FORMAT(k, "%s:created:%g:%s", team ? "Enemies" : "Allies", S->turn, DEF_NAME[def]);
      k->def = def;
      k->index = S->nteam[team] + 1;
      k->border = 1;
      k->hp = k->maxHp = ceil(card->power);
      k->damage = ceil(card->power / 2);
      k->power = card->power;
      k->entered = 0; k->dead = 0; k->identity = NOCARD; k->abilityOverride = AO_UNDEFINED; k->nbonus = 0;
      clear_card_state(k);
      k->counter[C_normalDamage] = ceil(card->power / 2);
      k->counter[C_normalMaxHp] = ceil(card->power);
      list_push(S->team[team], &S->nteam[team], n);
      break;
    }
    case A_DISPEL: reset_combat_stats(rt, enemy); break;
    case A_PANDEMIC:
      for (int i = 0; i < S->nteam[enemy_team]; i++) {
        Card *t = CARD(S->team[enemy_team][i]);
        if (status_protected(rt, t->team)) continue;
        t->counter[C_poisonPercent] = jmin(or0(t->counter[C_poisonPercent]), -0.075);
        t->counter[C_poisonTurns] = jmax(or0(t->counter[C_poisonTurns]), 2);
      }
      break;
    case A_DIVINE_MIST:
      if (chance_roll(rt, team, CMP_LT, 0.7)) {
        double hp = DEF_HEALTH[e->def][1];
        e->power = DEF_POWER[e->def][1];
        e->damage = DEF_ATTACK[e->def][1];
        e->maxHp = hp;
        e->hp = hp;
      }
      break;
    case A_CHIMERIC: boost_stats(rt, c, 4); break;
    case A_PUPPY_EYES: e->damage *= 0.85; break;
    case A_CATASTROPHE: e->damage *= 0.6; break;
    case A_CLAWLESS: e->hp -= e->maxHp * 0.15; break;
    case A_CERBERUS: e->damage *= 0.7; break;
    case A_INFECTIOUS: e->damage *= e->boss ? 0.85 : 0.5; e->hp *= e->boss ? 0.85 : 0.5; break;
    case A_DRAGON_SLAYER: card->damage *= 1.75; break;
    case A_GREATER_MIGHT: boost_stats(rt, c, 1.4); break;
    case A_HEAVENLY_MIGHT: boost_stats(rt, c, 1.65); break;
    case A_COMBATANT: boost_stats(rt, c, 1.2); break;
    case A_SACRIFICE: card->damage *= 2; card->hp /= 2; break;
    case A_SUPER_STRENGTH: card->damage *= 1.25; card->maxHp *= 1.25; card->hp = card->maxHp; break;
    case A_IMMORTAL: card->maxHp *= 3; card->hp *= 3; break;
    case A_FURY_OF_THE_WHITE_TIGER: card->damage *= 3; break;
    case A_TYRANNOSPIRIT: { double f = or0(S->boosts[team].fossils); if (f > 0) card->damage *= pow(1.5, f); break; }
    case A_TURTLE_SHELL: card->maxHp = 30000; card->hp = 30000; break;
    case A_HAPPY_FAMILY: {
      int dads[S->nteam[team] + 1], n = 0;
      for (int i = 0; i < S->nteam[team]; i++) { int a = S->team[team][i]; if (a != c && CARD(a)->def == D_DAD && alive(rt, a)) dads[n++] = a; }
      for (int i = 0; i < n; i++) { Card *d = CARD(dads[i]); d->damage += card->damage; d->maxHp += card->maxHp; d->hp += jmax(0, card->hp); }
      card->hp = 0;
      resolve_deaths(rt);
      break;
    }
    case A_POP_UP_IMPRESSION:
      if (!status_protected(rt, e->team)) {
        double turns = card->flag[F_awakened] ? orv(card->counter[C_toyCount], 1) : 2;
        e->confused = jmax(e->confused, turns);
      }
      break;
    case A_NAUGHTY_LIST:
      for (int i = 0; i < S->nteam[team]; i++) {
        int a = S->team[team][i];
        if (!alive(rt, a)) continue;
        boost_stats(rt, a, 1.5);
        CARD(a)->flag[F_naughtyListDrain] = 1;
      }
      break;
    case A_TOIL: boost_stats(rt, c, 2); break;
    case A_BLOODLUST: card->counter[C_bloodlustBase] = card->damage; card->damage += card->damage; card->flag[F_bloodlustFirstTurn] = 1; break;
    case A_FLUFFY_AGGRESSION:
      if (card->flag[F_awakened]) {
        int seen[S->nfallen[team] + 1], toys = 0;
        for (int i = 0; i < S->nfallen[team]; i++) {
          int d = CARD(S->fallen[team][i])->def, dup = 0;
          if (!(DEF_SETS[d] & SET_TOY)) continue;
          for (int j = 0; j < toys; j++) if (seen[j] == d) dup = 1;
          if (!dup) seen[toys++] = d;
        }
        card->damage *= toy_bear_awakened_multiplier(rt, c, toys);
      } else card->damage *= 2;
      break;
    case A_SPEEDY_PROGRESSION:
      card->counter[C_attacks] = or0(card->counter[C_attacks]) + (card->flag[F_awakened] ? orv(card->counter[C_toyCount], 1) : 3);
      break;
    case A_RED_NOSED_REINDEER: if (!status_protected(rt, e->team)) e->blind = 1; break;
    case A_BEHAVIORAL_THERAPY: e->flag[F_slowed] = 1; e->counter[C_slowed] = 0; break;
    case A_STAMPEDE: card->counter[C_attacks] = or0(card->counter[C_attacks]) + 1; e->stunned = jmax(1, e->stunned); break;
    case A_ICE_AGE: e->flag[F_slowed] = 1; e->counter[C_slowed] = 0; break;
    case A_HELL_S_CURSE: e->flag[F_sealed] = 1; e->hp /= 2; break;
    case A_NORTHERN_WINDS:
      deal_damage(rt, c, enemy, 1, 0);
      card->damage += e->damage * 0.25;
      e->damage *= 0.75;
      resolve_deaths(rt);
      if (alive(rt, enemy) && has_ability(rt, enemy, A_HATRED) && alive(rt, c)) { deal_damage(rt, enemy, c, 0.5, 0); resolve_deaths(rt); }
      break;
    case A_AZURE_DRAGON_WRATH:
      deal_damage(rt, c, enemy, 1.5, 1);
      resolve_deaths(rt);
      if (alive(rt, enemy) && has_ability(rt, enemy, A_HATRED) && alive(rt, c)) { deal_damage(rt, enemy, c, 0.5, 0); resolve_deaths(rt); }
      break;
    case A_REVENGE: if (S->nfallen[team] > 0) { deal_damage(rt, c, enemy, 2, 0); resolve_deaths(rt); } break;
    case A_STOLEN_SPOTLIGHT: {
      int behind = at(rt, team, 1);
      if (behind != NOCARD && behind != c) {
        Card *b = CARD(behind);
        card->damage += b->damage * 0.75;
        card->maxHp += b->maxHp * 0.75;
        card->hp += jmax(0, b->hp) * 0.75;
        list_remove_at(S->team[team], &S->nteam[team], 1);
        b->dead = 1;
      }
      break;
    }
    case A_A_PAIR_OF_TWO:
      if (!card->flag[F_paired]) {
        card->flag[F_paired] = 1;
        int one = clone_at_fraction(rt, c, 0.35, S->nteam[team] + 1);
        list_push(S->team[team], &S->nteam[team], one);
        int two = clone_at_fraction(rt, c, 0.35, S->nteam[team] + 1);
        list_push(S->team[team], &S->nteam[team], two);
      }
      break;
    case A_TERROR_FROM_ABOVE: {
      int *deck = S->team[enemy_team];
      for (int i = S->nteam[enemy_team] - 1; i > 0; i--) { int j = chance_pick(rt, i + 1); int t = deck[i]; deck[i] = deck[j]; deck[j] = t; }
      break;
    }
    case A_SUDDEN_DEMISE: {
      int hits = 1 + chance_pick(rt, 8);
      double hit_damage = card->damage * 0.1;
      for (int h = 0; h < hits; h++)
        for (int i = 0; i < S->nteam[enemy_team]; i++) { Card *t = CARD(S->team[enemy_team][i]); t->hp -= jmin(t->hp, hit_damage); }
      resolve_deaths(rt);
      break;
    }
    case A_COSMIC_RIVALRY: perform_entry_attack(rt, c, 3, 0); break;
    case A_KITCHEN: {
      card->flag[F_kitchenDomain] = 1;
      int hits = 2 + chance_pick(rt, 4);
      for (int h = 0; h < hits; h++) {
        int target = active(rt, enemy_team);
        if (target == NOCARD || !alive(rt, c)) break;
        deal_damage(rt, c, target, 0.35, 1);
        resolve_deaths(rt);
      }
      break;
    }
    case A_WAR_SCYTHE: {
      if (card->flag[F_warScytheEntryUsed]) break;
      card->flag[F_warScytheEntryUsed] = 1;
      int targets[2], n = 0;
      for (int i = 0; i < S->nteam[enemy_team] && n < 2; i++) if (alive(rt, S->team[enemy_team][i])) targets[n++] = S->team[enemy_team][i];
      card->flag[F_warScytheEntry] = 1;
      for (int i = 0; i < n; i++) {
        int t = targets[i];
        if (!alive(rt, c) || !alive(rt, t)) continue;
        CARD(t)->flag[F_suppressOnDeath] = 1;
        deal_damage(rt, c, t, 1, 1);
        if (CARD(t)->hp > 0) CARD(t)->flag[F_suppressOnDeath] = 0;
        int tt = CARD(t)->team, idx = index_of(S->team[tt], S->nteam[tt], t);
        if (CARD(t)->hp <= 0 && idx > 0) {
          list_remove_at(S->team[tt], &S->nteam[tt], idx);
          CARD(t)->dead = 1;
          CARD(t)->hp = 0;
          list_push(S->fallen[tt], &S->nfallen[tt], t);
        }
        resolve_deaths(rt);
      }
      CARD(c)->flag[F_warScytheEntry] = 0;
      break;
    }
    case A_FIRST_BLOOD: perform_entry_attack(rt, c, 0.5, 0); break;
    case A_DEADLY_AMBUSH: {
      int first = active(rt, enemy_team);
      if (first != NOCARD) {
        deal_damage(rt, c, first, 1, 0);
        if (alive(rt, first) && !status_protected(rt, CARD(first)->team)) CT(first, poisonPercent) = -0.15;
        resolve_deaths(rt);
      }
      break;
    }
    case A_HORNED_ATTACK: {
      int first = active(rt, enemy_team);
      if (first != NOCARD) {
        double hp_before = CARD(first)->hp;
        double dealt = deal_damage(rt, c, first, 1, 0);
        resolve_deaths(rt);
        if (dealt > hp_before && CARD(first)->hp <= 0) {
          int next = active(rt, enemy_team);
          if (next != NOCARD) {
            double overflow = jmin(CARD(next)->hp, dealt - hp_before);
            if (overflow >= CARD(next)->hp && has_ability(rt, next, A_PARADOX) && !FL(next, paradox)) FL(next, paradox) = 1;
            CARD(next)->hp -= overflow;
          }
          resolve_deaths(rt);
        }
      }
      break;
    }
    case A_FIGHT_DIRTY:
    case A_QUICK_STRIKE:
    case A_HEART_HUNTER:
      perform_entry_attack(rt, c, 1, 0);
      if (name == A_HEART_HUNTER && active(rt, enemy_team) != NOCARD) CT(active(rt, enemy_team), bleed) = 100;
      break;
    case A_SACRED_JUDGMENT: {
      int n = S->nteam[enemy_team], targets[n + 1];
      memcpy(targets, S->team[enemy_team], sizeof(int) * n);
      for (int i = 0; i < n; i++) {
        if (!alive(rt, targets[i])) continue;
        deal_damage(rt, c, targets[i], 0.25, 0);
        resolve_deaths(rt);
      }
      break;
    }
    case A_STARDUST_DRIVER: perform_entry_attack(rt, c, 2.5, 0); break;
  }
}
static void on_entry(Runtime *rt, int c) { enter(rt); on_entry_(rt, c); rt->depth--; }

// ---- offensive / defensive modifiers ----
typedef struct { double damage; int bypass, special; } Offense;

static int is_special(int name) {
  switch (name) {
    case A_TRUE_STRIKE: case A_MAELSTROM: case A_JUDGMENT: case A_ARMAGEDDON: case A_DRACONIC_HEART: case A_EXPLOSION:
    case A_TELEKINESIS: case A_FAVORABLE_ODDS: case A_VAINGLORY: case A_MODESTY: case A_DECAPITATE: case A_MARTIAL_WILL:
    case A_DOMINATE: case A_DECIMATE: case A_PREHISTORIC_WRATH: case A_BIG_AND_LARGE: case A_BLADE: case A_DEFRAUD:
    case A_ASSASSINATE: case A_SKY_DROP: case A_SHADOW_PREDATOR: case A_APEX_PREDATOR: case A_INFINITE_DAGGER_WORKS:
    case A_EXTINCTION: case A_GOD_OF_THUNDER: case A_FIRE_WORLD: case A_MOONLIGHT_BEAM: case A_DIRTY_CLAW: case A_HEART_HUNTER:
    case A_CHAINSAW: case A_FIREPOWER: case A_RAPID_BLOWS: case A_BEHAVIORAL_THERAPY: case A_HOLY_WRATH: case A_UNLUCKY:
    case A_DRAGON_SLAYER: case A_FROZEN_WRATH: case A_ABSOLUTE_APEX: case A_DARK_QI_MANIPULATION: case A_CHAOS_DESTRUCTION:
    case A_CONSTELLARTAURUS: case A_CONSTELLARSAGITTARIUS: case A_WHOOPING: case A_TWELVE_DEVAS_AXE:
    case A_STAFF_OF_PERFECT_ENLIGHTENMENT: case A_KITCHEN: case A_WAR_SCYTHE:
      return 1;
  }
  return 0;
}

static Offense offensive(Runtime *rt, int attacker, int target, double initial) {
  int gained[6], g = active_bonus(rt, attacker, gained);
  if (g) {
    Offense result = {initial, 0, 0};
    for (int i = 0; i < g; i++) {
      Offense next;
      WITH_ABILITY(attacker, gained[i], { next = offensive(rt, attacker, target, result.damage); });
      result.damage = next.damage; result.bypass = result.bypass || next.bypass; result.special = result.special || next.special;
    }
    return result;
  }
  int name = resolved_ability(rt, attacker);
  Offense o = {initial, 0, 0};
  if (name < 0 || !has_ability(rt, attacker, name)) return o;
  Card *a = CARD(attacker), *t = CARD(target);
  o.special = is_special(name);
  switch (name) {
    case A_SPARTAN_S_RAGE: o.damage *= 1 + jmax(0, 1 - a->hp / a->maxHp); break;
    case A_LIFE_TAP: o.damage *= 1.3; break;
    case A_FROZEN_SOLITUDE: if (or0(t->counter[C_videoFrozen]) > 0) o.damage *= 2; break;
    case A_DIVINE_ARROGANCE: if (a->hp > t->hp) o.damage *= 1.75; break;
    case A_HIDDEN_BLADE: {
      int seen = 0;
      for (int j = 0; j < S->nstruck; j++) if (S->struck[j].attacker == attacker && S->struck[j].target == t->id) seen = 1;
      if (!seen) {
        if (S->nstruck >= MAXSTRUCK) engine_overflow(rt);
        S->struck[S->nstruck++] = (Struck){attacker, t->id};
        o.damage *= 2; o.bypass = 1; o.special = 1;
      }
      break;
    }
    case A_VOID_HEART: if (a->flag[F_voidReady]) { a->flag[F_voidReady] = 0; o.damage *= 2; } break;
    case A_TWELVE_DEVAS_AXE: o.damage *= 2.5; break;
    case A_STAFF_OF_PERFECT_ENLIGHTENMENT:
      o.damage *= 1.25; o.bypass = 1;
      if (!status_protected(rt, t->team) && chance_roll(rt, a->team, CMP_LT, 0.25)) t->stunned = jmax(1, t->stunned);
      break;
    case A_KITCHEN: o.bypass = 1; break;
    case A_WAR_SCYTHE: if (a->flag[F_warScytheEntry]) o.bypass = 1; break;
    case A_TRUE_STRIKE: if (chance_roll(rt, a->team, CMP_GT, 0.5)) o.damage *= 2; break;
    case A_ABSOLUTE_APEX: o.damage *= 1.5; break;
    case A_WHOOPING: if (DEF_AGE[a->def] > DEF_AGE[t->def]) o.damage *= 2; break;
    case A_CONSTELLARTAURUS: o.damage *= constellar_taurus_factor(rt, attacker); break;
    case A_CHAOS_DESTRUCTION: if (a->flag[F_chaosTriple]) { o.damage *= 3; a->flag[F_chaosTriple] = 0; } break;
    case A_DARK_QI_MANIPULATION: if (a->flag[F_awakened]) o.damage *= 2; break;
    case A_MONKEY_KING_S_RAGE:
      if (a->hp / a->maxHp <= 0.5 && !a->flag[F_transformed]) { a->flag[F_transformed] = 1; a->maxHp *= 2; a->hp *= 2; o.damage *= 2; }
      break;
    case A_REAPER_S_LUCK: {
      static const double changes[3] = {-0.1, 0.15, 0.3};
      double change = changes[chance_pickroll(rt, a->team, 3)];
      double ratio = a->maxHp > 0 ? a->hp / a->maxHp : 0;
      a->maxHp *= 1 + change;
      a->hp = ratio * a->maxHp;
      a->damage *= 1 + change;
      break;
    }
    case A_HOLY_WRATH: if (DEF_SETS[t->def] & SET_UNDEAD) o.damage *= 2; break;
    case A_UNLUCKY: if (DEF_SETS[t->def] & SET_RNG_ABILITY) o.damage *= 2; break;
    case A_MAELSTROM:
      a->counter[C_maelstrom] = jmod(or0(a->counter[C_maelstrom]), 2) + 1;
      if (a->counter[C_maelstrom] == 1) o.damage *= 2;
      break;
    case A_JUDGMENT: o.damage += (a->maxHp - a->hp) * 0.7; break;
    case A_ARMAGEDDON: {
      if (a->flag[F_suppressArmageddonOnce]) { a->flag[F_suppressArmageddonOnce] = 0; a->flag[F_armageddonLethalThisHit] = 0; break; }
      int success = chance_roll(rt, a->team, CMP_GT, 0.5);
      a->flag[F_armageddonLethalThisHit] = success;
      if (success) o.damage = INFINITY;
      break;
    }
    case A_DRACONIC_HEART: o.damage *= 3; a->damage *= 0.9; a->hp *= 0.9; a->maxHp *= 0.9; break;
    case A_EXPLOSION: o.damage *= 3; a->stunned = jmax(1, a->stunned); break;
    case A_TELEKINESIS:
      a->counter[C_telekinesis] = jmod(or0(a->counter[C_telekinesis]), 2) + 1;
      o.damage *= a->counter[C_telekinesis] == 1 ? 2 : 4;
      break;
    case A_DRAGON_SLAYER: if (DEF_SETS[t->def] & SET_DRAGON) o.damage *= 2; break;
    case A_FROZEN_WRATH: if (!status_protected(rt, t->team)) t->counter[C_frostbite] = jmax(or0(t->counter[C_frostbite]), 2); break;
    case A_FAVORABLE_ODDS: o.damage *= jmax(1, chance_pickroll(rt, a->team, 5) + 1); break;
    case A_VAINGLORY: if (a->hp / a->maxHp > 0.5) o.damage *= 1.5; break;
    case A_FRAIL: o.damage *= 1.5; break;
    case A_MODESTY: o.damage *= 0.7; break;
    case A_DECAPITATE: o.damage *= 1.5; break;
    case A_MARTIAL_WILL: {
      double ah = or0(a->counter[C_martialHits]), th = or0(t->counter[C_martialHits]);
      if (ah > 0 && th > 0) o.damage *= pow(1.5, ah);
      else if (ah == 0 && th > 0) t->counter[C_martialHits] = 0;
      a->counter[C_martialHits] = ah + 1;
      t->counter[C_martialHits] = or0(t->counter[C_martialHits]) + 1;
      break;
    }
    case A_DECIMATE: o.damage *= 3; a->damage *= 0.7; break;
    case A_PREHISTORIC_WRATH: if (t->hp / t->maxHp <= 0.5) o.damage *= 2; break;
    case A_BIG_AND_LARGE: if (a->hp / a->maxHp > 0.25) o.damage *= 3; break;
    case A_BLADE: o.damage += a->maxHp * 0.15; a->maxHp *= 0.85; a->hp = jmin(a->hp, a->maxHp); break;
    case A_DEFRAUD: o.damage = t->hp * 0.5; break;
    case A_ASSASSINATE: if (t->hp / t->maxHp <= 0.25) o.damage = t->maxHp; break;
    case A_SKY_DROP: o.damage *= 1.5; break;
    case A_SHADOW_PREDATOR: if (a->flag[F_double_]) { o.damage *= 2; a->flag[F_double_] = 0; } break;
    case A_APEX_PREDATOR: o.damage *= 1.5; break;
    case A_INFINITE_DAGGER_WORKS: o.damage *= 2; break;
    case A_EXTINCTION: o.damage *= 10; a->hp = 0; break;
    case A_GOD_OF_THUNDER:
      a->counter[C_thunder] = jmod(or0(a->counter[C_thunder]), 2) + 1;
      if (a->counter[C_thunder] == 1) o.damage *= 2.5;
      o.bypass = 1;
      break;
    case A_FIRE_WORLD:
      a->counter[C_fireWorld] = jmod(or0(a->counter[C_fireWorld]), 2) + 1;
      if (a->counter[C_fireWorld] == 1) o.damage *= 4;
      break;
    case A_MOONLIGHT_BEAM: if (!a->flag[F_moonlightUsed]) { a->flag[F_moonlightUsed] = 1; o.damage *= 5; } break;
    case A_DIRTY_CLAW: t->counter[C_poisonPercent] = -0.15; t->weakness = 1; t->counter[C_weaknessTurns] = 100; break;
    case A_UNDEAD_PRACTITIONER: t->counter[C_bleed] = 100; break;
    case A_HEART_HUNTER: if (or0(t->counter[C_bleed]) > 0) o.damage *= 3; break;
    case A_CHAINSAW: o.damage *= 0.5; break;
    case A_FIREPOWER: o.damage *= 0.25; break;
    case A_RAPID_BLOWS: o.damage *= 0.5; break;
    case A_SPEEDY_PROGRESSION: o.damage /= 3; break;
    case A_BEHAVIORAL_THERAPY: t->counter[C_bleed] = or0(t->counter[C_bleed]) + 1; break;
  }
  if (name == A_DOMINATE && border_tier(rt, attacker) > border_tier(rt, target)) o.damage *= 2;
  if (name == A_LIGHTNING_SLASH) { o.damage *= 1.5; o.bypass = 1; }
  if (name == A_LIMITLESS || name == A_TRUE_FANG) o.bypass = 1;
  return o;
}

static double defensive(Runtime *rt, int attacker, int target, double initial) {
  int gained[6], g = active_bonus(rt, target, gained);
  if (g) {
    double damage = initial;
    for (int i = 0; i < g; i++) WITH_ABILITY(target, gained[i], { damage = defensive(rt, attacker, target, damage); });
    return damage;
  }
  int name = resolved_ability(rt, target);
  double damage = initial;
  if (name < 0 || !has_ability(rt, target, name)) return damage;
  Card *a = CARD(attacker), *t = CARD(target);
  switch (name) {
    case A_DIVINE_ARROGANCE: if (t->hp < a->hp) damage *= 0.55; break;
    case A_THE_D8: damage *= 1 - or0(t->counter[C_d8Reduction]); break;
    case A_FORTIFY: {
      double absorbed = jmin(damage, or0(t->counter[C_blockHp]));
      t->counter[C_blockHp] = jmax(0, or0(t->counter[C_blockHp]) - absorbed);
      damage -= absorbed;
      break;
    }
    case A_CONSTELLARTAURUS: damage /= constellar_taurus_factor(rt, target); break;
    case A_CONSTELLARCANCER: {
      double threshold = orv(t->counter[C_cancerThreshold], 1);
      if (threshold > 0 && damage < t->maxHp * threshold) { damage = 0; t->counter[C_cancerThreshold] = jmax(0, threshold - 0.15); }
      break;
    }
    case A_DANGER_SENSE:
    case A_DEADLY_AMBUSH:
      if (!t->flag[F_dangerSense] && damage > t->hp) {
        t->flag[F_dangerSense] = 1;
        damage = 0;
        int team = t->team, idx = index_of(S->team[team], S->nteam[team], target);
        if (idx >= 0 && idx + 1 < S->nteam[team]) { S->team[team][idx] = S->team[team][idx + 1]; S->team[team][idx + 1] = target; }
      }
      break;
    case A_EVASION: if (chance_roll(rt, t->team, CMP_GT, 0.9)) damage = 0; break;
    case A_DIVINE_ASCENSION: if (chance_roll(rt, t->team, CMP_GT, pow(damage / t->maxHp, 2))) damage = 0; break;
    case A_MASTERED_ASCENSION:
      t->counter[C_masteredAscension] = jmod(or0(t->counter[C_masteredAscension]) + 1, 2);
      if (t->counter[C_masteredAscension] == 1) damage = 0;
      break;
    case A_VAJRA_SHORT_SWORD:
      if (chance_roll(rt, t->team, CMP_LT, 0.4)) {
        double reflected = jmin(jmax(0, a->hp * 0.75), jmax(0, damage * 0.75));
        damage = 0;
        a->hp -= reflected;
      }
      break;
    case A_SHIELD_OF_AHIMSA: damage *= 0.65; break;
    case A_COSMIC_RIVALRY: damage *= jmax(0, 1 - jmin(1, or0(t->counter[C_cosmicRivalryDR]))); break;
    case A_FINESSE: if (damage < t->maxHp * 0.3) damage = 0; break;
    case A_LAST_STAND: if (damage >= t->hp && !t->flag[F_lastStand]) { damage = t->hp - 1; t->flag[F_lastStand] = 1; } break;
    case A_ARMOR: damage = jmax(0, damage - t->maxHp * 0.1); break;
    case A_DRAGON_SLAYER: if (DEF_SETS[a->def] & SET_DRAGON) damage *= 0.5; break;
    case A_OUTRANK: if (DEF_RARITY[a->def][a->border] < DEF_RARITY[t->def][t->border]) damage *= 0.5; break;
    case A_GOLDEN_BELL_SHIELD: if (DEF_SETS[a->def] & (SET_DEMON | SET_IMP)) damage /= 3; break;
    case A_FROZEN_WRATH: if (or0(a->counter[C_frostbite]) > 0) damage *= 0.5; break;
    case A_BRITTLE: damage *= 2; break;
    case A_MANA_SHIELD: if (!t->flag[F_manaShield] && damage < t->hp) { damage = 0; t->flag[F_manaShield] = 1; } break;
    case A_VAINGLORY: if (t->hp / t->maxHp > 0.5) damage *= 0.7; break;
    case A_MODESTY: damage *= 1.3; break;
    case A_SCALE_ARMOR: damage = jmax(0, damage - t->maxHp * 0.15) / 2; break;
    case A_STALWART: if (damage > t->maxHp / 3 && t->hp > t->maxHp / 3) damage = t->maxHp / 3; break;
    case A_DIVINE_BARRIER: if (!t->flag[F_divineBarrier]) { damage = 0; t->flag[F_divineBarrier] = 1; } break;
    case A_UNTOUCHABLE: if (chance_roll(rt, t->team, CMP_GT, pow(damage / t->maxHp, 2))) damage = 0; break;
    case A_GUERILLA_WARFARE: if (chance_roll(rt, t->team, CMP_GT, 0.6)) { damage = 0; t->damage *= 1.2; } break;
    case A_THE_LOSER: if (!t->flag[F_loser] && damage > t->hp) { damage = 0; t->flag[F_loser] = 1; t->damage *= 2; } break;
    case A_INVISIBILITY: if (chance_roll(rt, t->team, CMP_GT, 0.4)) damage = 0; break;
    case A_LIMITLESS: if (!t->flag[F_limitless]) { damage = 0; t->flag[F_limitless] = 1; } break;
    case A_HEAVENLY_RULER:
      t->counter[C_heavenly] = jmod(or0(t->counter[C_heavenly]) + 1, 2);
      if (t->counter[C_heavenly] == 0) damage *= -0.8;
      break;
    case A_ABSOLUTE_SOVEREIGNTY: damage *= 0.65; break;
    case A_DRACONIC_HEART: damage /= 3; break;
    case A_INVINCIBILITY: damage *= 0.25; break;
    case A_HIDDEN_CURSE: {
      double maxes = or0(t->counter[C_hiddenCurse]);
      int afflicted = a->weakness || a->burn > 0 || a->confused > 0 || a->stunned > 0 || a->blind || or0(a->counter[C_bleed]) > 0
                      || truthy(a->counter[C_poisonFlat]) || truthy(a->counter[C_poisonPercent]);
      if (maxes <= 5 && afflicted) { damage = 0; t->counter[C_hiddenCurse] = maxes + 1; }
      break;
    }
    case A_TRANSCEND_TIME:
      t->counter[C_transcend] = jmod(or0(t->counter[C_transcend]) + 1, 2);
      if (t->counter[C_transcend] == 1) damage = 0;
      break;
    case A_SNOWBOUND: if (t->stunned > 0 || t->flag[F_dodge]) { damage = 0; t->flag[F_dodge] = 0; } break;
    case A_SHELTER_OBSESSION: {
      double cap = t->flag[F_awakened] ? t->maxHp / 4 : t->maxHp / 2;
      if (damage > cap && t->hp > cap) damage = cap;
      break;
    }
    case A_BIG_AND_LARGE: if (t->hp / t->maxHp > 0.25) damage *= 0.5; break;
    case A_FRAIL: damage *= 1.5; break;
    case A_HUMANITY_S_SPIRIT: if (t->hp / t->maxHp < 0.25) damage *= 0.5; break;
    case A_PERFORATING_MIST: damage *= 1.5; break;
    case A_REFLECTIVE_SHELL: {
      double ability_damage = damage - a->damage;
      if (ability_damage > 0) {
        double reflected = jmin(t->damage * 8, ability_damage * 0.25);
        damage -= reflected;
        a->hp -= reflected;
      }
      break;
    }
    case A_SKY_DROP: if (!truthy(t->counter[C_drop]) || jmod(t->counter[C_drop], 2) != 0) damage = 0; break;
    case A_SPIKES: damage *= 0.75; a->counter[C_bleed] = 2; break;
    case A_SHADOW_PREDATOR: if (chance_roll(rt, t->team, CMP_GT, 0.6)) { damage = 0; t->flag[F_double_] = 1; } break;
    case A_APEX_PREDATOR: damage *= 0.5; break;
    case A_ABSOLUTE_APEX: damage *= 0.5; break;
    case A_IMMORTAL_ASCENSION: if (t->flag[F_awakened]) damage *= 0.5; break;
    case A_FINAL_TAIL: damage = 0; break;
    case A_PERSISTENT: {
      double persistence = or0(t->counter[C_persistence]);
      if (damage >= t->hp && persistence < 2) { damage = t->hp - 1; t->counter[C_persistence] = persistence + 1; }
      break;
    }
    case A_RUN_AS_FAST_AS_YOU_CAN:
      t->counter[C_runFast] = jmod(or0(t->counter[C_runFast]) + 1, 2);
      if (t->counter[C_runFast] == 0) { damage = 0; t->counter[C_attacks] = or0(t->counter[C_attacks]) + 1; }
      break;
    case A_BIND: a->damage *= 0.9; break;
    case A_AVALON: if (damage < t->damage * 0.75) damage = 0; break;
    case A_HEARD_BUT_NOT_SEEN: {
      double dodge = jmin(0.5, 0.2 + or0(t->counter[C_heardHits]) * 0.1);
      if (chance_roll(rt, t->team, CMP_LT, dodge)) damage = 0;
      else t->counter[C_heardHits] = or0(t->counter[C_heardHits]) + 1;
      break;
    }
    case A_LIGHTS_WAY:
      if (!t->flag[F_lightsWay] && damage >= t->hp) {
        t->flag[F_lightsWay] = 1;
        damage = 0;
        t->hp = jmin(t->maxHp, t->hp + t->maxHp * 0.5);
      }
      break;
  }
  if (name == A_DOMINATE && border_tier(rt, target) > border_tier(rt, attacker)) damage /= 2;
  if (initial > 0 && damage == 0 && ABILITY_DODGE[name]) t->flag[F_evadedThisHit] = 1;
  return damage;
}

static int try_revive(Runtime *rt, int attacker, int target) {
  Card *a = CARD(attacker), *t = CARD(target);
  if (t->hp > 0 || a->flag[F_warScytheEntry]) return 0;
  int gained[6], g = active_bonus(rt, target, gained);
  if (g) {
    for (int i = 0; i < g; i++) {
      int revived;
      WITH_ABILITY(target, gained[i], { revived = try_revive(rt, attacker, target); });
      if (revived) return 1;
    }
    return 0;
  }
  int name = resolved_ability(rt, target);
  if (name < 0) return 0;
  if (!has_ability(rt, target, name)) return 0;
  if (name == A_BLACK_BOX && !t->flag[F_blackBoxUsed]) { t->flag[F_blackBoxUsed] = 1; t->hp = 1; a->hp -= t->maxHp * 0.5; return 1; }
  if (name == A_REVIVE && !t->flag[F_revived] && chance_roll(rt, t->team, CMP_GT, 0.5)) { t->flag[F_revived] = 1; t->hp = t->maxHp * 0.5; return 1; }
  if (name == A_ETERNITY && !t->flag[F_revived] && chance_roll(rt, t->team, CMP_GT, 0.5)) { t->flag[F_revived] = 1; t->hp = t->maxHp; return 1; }
  if (name == A_FROZEN_ASHES && !t->flag[F_revived] && chance_roll(rt, t->team, CMP_GT, 0.5)) {
    t->flag[F_revived] = 1; t->hp = t->maxHp; a->stunned = jmax(1, a->stunned); return 1;
  }
  if (name == A_UNPAID_INTERNS && or0(t->counter[C_interns]) < 2) { t->counter[C_interns] = or0(t->counter[C_interns]) + 1; t->hp = t->maxHp; return 1; }
  if (name == A_FLAMES_OF_REBIRTH && !t->flag[F_revived]) { t->flag[F_revived] = 1; t->hp = t->maxHp * 0.5; t->damage *= 2; a->burn = 2; return 1; }
  return 0;
}

static double marrowclaw_reflect(Card *a, double amount) {
  return a->def == D_MARROWCLAW ? jmin(jmax(0, a->hp - 1), amount) : jmin(a->hp, amount);
}

static void target_retro(Runtime *rt, int attacker, int target, double damage) {
  int gained[6], g = active_bonus(rt, target, gained);
  if (g) {
    for (int i = 0; i < g; i++) WITH_ABILITY(target, gained[i], { target_retro(rt, attacker, target, damage); });
    return;
  }
  int name = resolved_ability(rt, target);
  if (name < 0 || !has_ability(rt, target, name)) return;
  Card *a = CARD(attacker), *t = CARD(target);
  switch (name) {
    case A_RULER_OF_HUMANS:
      for (int i = 0; i < S->nteam[t->team]; i++) { int ally = S->team[t->team][i]; if (alive(rt, ally)) CARD(ally)->damage += damage * 1.25; }
      break;
    case A_RESTORATION: if (t->hp > 0) t->hp += damage * 0.7; break;
    case A_RAGE: if (t->hp > 0) t->damage *= 1.25; break;
    case A_UNDEAD: if (t->hp > 0) t->hp = jmin(t->maxHp, t->hp + t->maxHp * 0.25); break;
    case A_PASSION:
      a->counter[C_passion] = or0(a->counter[C_passion]) + 1;
      if (a->counter[C_passion] <= 3) a->damage *= 0.65;
      break;
    case A_EIGHT_HEADS: t->damage *= 0.875; t->hp *= 0.875; break;
    case A_WAIL: if (t->hp < t->maxHp / 2 && !t->flag[F_wail]) { t->flag[F_wail] = 1; a->stunned = jmax(1, a->stunned); } break;
    case A_FURY_OF_THE_WHITE_TIGER: t->damage = jmax(0, t->damage - damage); break;
    case A_THE_FALL: a->hp -= marrowclaw_reflect(a, damage); break;
    case A_SELF_DESTRUCT:
    case A_DEATH_EMBRACE:
      if (t->hp <= 0 && chance_roll(rt, t->team, CMP_GT, 0.5)) a->hp -= marrowclaw_reflect(a, t->maxHp);
      break;
    case A_UNDEAD_PRACTITIONER:
      if (t->hp > 0 && !t->flag[F_undeadPractitioner] && t->hp <= t->maxHp / 2) { t->hp += t->maxHp * 0.5; t->flag[F_undeadPractitioner] = 1; }
      break;
    case A_GUILT: if (t->hp <= 0) a->flag[F_hanged] = 1; break;
    case A_INTO_THE_SUN: if (t->hp / t->maxHp < 0.33) { t->hp = 0; a->hp = 0; } break;
    case A_FRIGID_TOUCH: if (damage > 0 && chance_roll(rt, a->team, CMP_GE, 0.5)) t->stunned = jmax(1, t->stunned); break;
    case A_BLINDING_FLASH: if (chance_roll(rt, a->team, CMP_GT, 0.7)) a->flag[F_extraTurn] = 1; break;
    case A_GRAPE_JUICE: a->hp -= marrowclaw_reflect(a, t->damage / 2); break;
    case A_PERFECT_SACRIFICE:
      if (t->hp <= 0) {
        a->hp -= marrowclaw_reflect(a, t->maxHp);
        for (int i = 0; i < S->nteam[t->team]; i++) boost_stats(rt, S->team[t->team][i], 1.2);
      }
      break;
    case A_PLAGUE:
      if (damage > 0 && attacker != target && !status_protected(rt, a->team)) {
        a->counter[C_poisonFlat] = jmax(or0(a->counter[C_poisonFlat]), t->damage);
        a->counter[C_poisonTurns] = jmax(or0(a->counter[C_poisonTurns]), 2);
      }
      break;
    case A_STEAL_CHRISTMAS:
      if (damage > 0 && attacker != target && !t->flag[F_stealChristmasUsed]) {
        t->flag[F_stealChristmasUsed] = 1;
        double sh = jmax(0, a->hp * 0.2), sd = jmax(0, a->damage * 0.2);
        t->damage += sd; t->hp += sh;
        a->hp = jmax(0, a->hp - sh); a->damage = jmax(0, a->damage - sd);
      }
      break;
    case A_SHELTER_OBSESSION:
      if (damage > 0 && t->flag[F_awakened]) {
        int team = t->team, seen[S->nteam[team] + S->nfallen[team] + 1], n = 0;
        for (int pass = 0; pass < 2; pass++) {
          const int *list = pass ? S->fallen[team] : S->team[team];
          int len = pass ? S->nfallen[team] : S->nteam[team];
          for (int i = 0; i < len; i++) {
            int d = CARD(list[i])->def, dup = 0;
            if (!(DEF_SETS[d] & SET_TOY)) continue;
            for (int j = 0; j < n; j++) if (seen[j] == d) dup = 1;
            if (dup) continue;
            seen[n++] = d;
            boost_stats(rt, list[i], 1.1);
          }
        }
      }
      break;
    case A_POKE_THE_BEAST: if (damage > 0 && !status_protected(rt, a->team)) a->burn = jmax(a->burn, 2); break;
    case A_LAST_MEAL:
      if (damage > 0) { double fossils = or0(S->boosts[t->team].fossils); a->counter[C_death] = jmax(2, 5 - fossils); }
      break;
    case A_BOILING_BLOOD: if (!status_protected(rt, a->team)) a->burn = 3; break;
    case A_MELT: if (!status_protected(rt, a->team)) a->burn += 5; break;
  }
}

static int vampire_matron_can_heal(Runtime *rt, int c) { int n = effective_name(rt, c); return n != D_ODIN && n != D_GILGAMESH; }
static double lifesteal_fraction(Runtime *rt, int attacker, double base) {
  double vamp = S->boosts[CARD(attacker)->team].vampireMatron;
  return truthy(vamp) && vampire_matron_can_heal(rt, attacker) ? base * (100 + vamp * 5) / 100 : base;
}

static int attacker_retro(Runtime *rt, int attacker, int target, double damage) {
  int gained[6], g = active_bonus(rt, attacker, gained);
  if (g) {
    int did = 0;
    for (int i = 0; i < g; i++) { int r; WITH_ABILITY(attacker, gained[i], { r = attacker_retro(rt, attacker, target, damage); }); did = r || did; }
    return did;
  }
  int name = resolved_ability(rt, attacker), did = 0;
  if (name < 0 || !has_ability(rt, attacker, name)) return 0;
  Card *a = CARD(attacker), *t = CARD(target);
  switch (name) {
    case A_BLADE_OF_MIQUELLA:
    case A_SPARTAN_S_RAGE: {
      double fraction = name == A_BLADE_OF_MIQUELLA ? 0.25 : jmax(0, 1 - a->hp / a->maxHp) * 0.5;
      if (damage > 0 && alive(rt, attacker)) { a->hp = jmin(a->maxHp, a->hp + damage * lifesteal_fraction(rt, attacker, fraction)); did = 1; }
      break;
    }
    case A_CONSTELLARSCORPIO:
      if (damage > 0 && !status_protected(rt, t->team)) t->counter[C_poisonFlat] = jmax(or0(t->counter[C_poisonFlat]), a->damage);
      break;
    case A_PLAGUE:
      if (damage > 0 && !status_protected(rt, t->team)) {
        t->counter[C_poisonFlat] = jmax(or0(t->counter[C_poisonFlat]), a->damage);
        t->counter[C_poisonTurns] = jmax(or0(t->counter[C_poisonTurns]), 2);
      }
      break;
    case A_UNDYING: if (t->hp <= 0 && a->flag[F_undyingActive]) a->counter[C_undyingTurns] = or0(a->counter[C_undyingTurns]) + 1; break;
    case A_WITCH_S_CURSE:
      if (damage > 0 && !a->flag[F_witchCurseStolen]) {
        int stolen = ability(rt, target);
        if (stolen >= 0 && stolen != A_WITCH_S_CURSE) { a->flag[F_witchCurseStolen] = 1; a->abilityOverride = stolen; }
      }
      break;
    case A_FLESH_EATER:
      if (damage > 0) { double gain = damage * 0.25; a->hp = jmin(a->maxHp, a->hp + gain); a->damage += gain; }
      break;
    case A_GOBBLE:
      if (t->hp <= 0 && target != attacker) {
        a->damage += t->damage * 0.5;
        a->maxHp += t->maxHp * 0.5;
        a->hp = jmin(a->maxHp, a->hp + t->maxHp * 0.5 + a->maxHp * 0.3);
      }
      break;
    case A_PLAYING_GOD:
      if (t->hp <= 0 && target != attacker) {
        int team = a->team, n = new_card(rt);
        Card *k = CARD(n);
        a = CARD(attacker); t = CARD(target);
        *k = *a;
        ID_FORMAT(k, "%s:frankenstein:%g:%d", team ? "Enemies" : "Allies", S->turn, S->nteam[team]);
        k->def = D_FRANKENSTEIN;
        k->index = S->nteam[team] + 1;
        k->hp = a->maxHp; k->maxHp = a->maxHp; k->damage = a->damage;
        k->entered = 0; k->dead = 0; k->identity = NOCARD; k->abilityOverride = AO_UNDEFINED; k->nbonus = 0;
        clear_card_state(k);
        k->counter[C_normalDamage] = a->damage;
        k->counter[C_normalMaxHp] = a->maxHp;
        list_push(S->team[team], &S->nteam[team], n);
      }
      break;
    case A_FORBIDDEN_BANQUET:
      if (t->hp <= 0 && !a->flag[F_banquetStolen]) {
        int stolen = ability(rt, target);
        if (stolen >= 0 && stolen != A_FORBIDDEN_BANQUET) { a->flag[F_banquetStolen] = 1; a->abilityOverride = stolen; }
      }
      break;
    case A_REGENERATE: a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.2); did = 1; break;
    case A_PLUNDER:
      if (t->hp <= 0 && target != attacker) { a->damage += t->damage * 0.3; a->maxHp += t->maxHp * 0.3; a->hp += t->maxHp * 0.3; }
      break;
    case A_VORACITY: if (t->hp <= 0) { a->damage *= 1.2; a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.2); } break;
    case A_BLOOD_DRINKER:
    case A_LIFESTEAL: { double heal = damage * lifesteal_fraction(rt, attacker, 0.5); a->hp = jmin(a->maxHp, a->hp + heal); did = 1; break; }
    case A_DRAIN_VITALITY: {
      a->hp = jmin(a->maxHp, a->hp + damage * lifesteal_fraction(rt, attacker, 1)); did = 1;
      double stolen = jmin(t->damage, damage); a->damage += stolen; t->damage -= stolen;
      break;
    }
    case A_UNHOLY_CREATURE: if (!status_protected(rt, t->team)) t->counter[C_poisonPercent] = -0.15; break;
    case A_INSATIABLE: {
      int unholy = has_ability(rt, target, A_UNHOLY_CREATURE) && (!t->flag[F_unholyActive] || or0(t->counter[C_unholyTurns]) > 0);
      int undying = has_ability(rt, target, A_UNDYING) && (!t->flag[F_undyingActive] || or0(t->counter[C_undyingTurns]) > 0);
      int paradox = has_ability(rt, target, A_PARADOX) && !t->flag[F_paradox];
      if (t->hp <= 0 && target != attacker && !unholy && !undying && !paradox) {
        a->damage += t->damage * 0.3; a->maxHp += t->maxHp * 0.3; a->hp += t->maxHp * 0.3;
        a->flag[F_insatiableAttack] = 1;
      }
      break;
    }
    case A_DEVILISH:
      if (t->hp <= 0 && target != attacker) {
        int team = a->team, n = new_card(rt);
        Card *k = CARD(n);
        a = CARD(attacker); t = CARD(target);
        *k = *t;
        ID_FORMAT(k, "%s:devilish:%g:%s", team ? "Enemies" : "Allies", S->turn, DEF_NAME[t->def]);
        k->team = team;
        k->index = S->nteam[team] + 1;
        k->hp = t->maxHp;
        k->entered = 0; k->dead = 0;
        clear_card_state(k);
        k->counter[C_normalDamage] = t->damage;
        list_push(S->team[team], &S->nteam[team], n);
      }
      break;
    case A_ECLIPSE: if (damage > 0) t->flag[F_sealed] = 1; break;
    case A_DARK_QI_MANIPULATION:
      if (a->flag[F_awakened]) {
        a->hp = jmin(a->maxHp, a->hp + damage * 0.3); did = 1;
        if (t->hp <= 0) boost_stats(rt, attacker, 1.5);
      }
      break;
    case A_IMMORTAL_ASCENSION: if (a->flag[F_awakened] && t->hp <= 0) boost_stats(rt, attacker, 1.5); break;
    case A_DOOM:
      if (!has_ability(rt, target, A_EROSION) && t->hp > 0 && chance_roll(rt, a->team, CMP_GT, 1 - damage / t->hp)) { t->hp = 0; t->flag[F_sealed] = 1; }
      break;
    case A_DECAPITATE: {
      int unholy = has_ability(rt, target, A_UNHOLY_CREATURE) && (!t->flag[F_unholyActive] || or0(t->counter[C_unholyTurns]) > 0);
      if (t->hp <= 0 && !unholy) { boost_stats(rt, attacker, 1.2); a->flag[F_extraTurn] = 1; }
      break;
    }
    case A_FURY_OF_THE_WHITE_TIGER: if (t->hp <= 0) { a->damage *= 1.35; a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.35); } break;
    case A_FEEDER: if (t->hp <= 0) a->hp = a->maxHp; break;
    case A_DEFRAUD: a->hp -= a->maxHp / 4; break;
    case A_FIGHT_DIRTY: t->damage = floor(t->damage * 0.7); break;
    case A_UNFORGIVING: t->maxHp = jmax(1, t->maxHp - damage); t->hp = jmin(t->hp, t->maxHp); break;
    case A_EAT_THE_MOON: if (!has_ability(rt, target, A_EROSION) && t->hp / t->maxHp < 0.33) t->hp = 0; break;
    case A_DEATH_EMBRACE:
      if (!has_ability(rt, target, A_EROSION) && t->hp > 0 && chance_roll(rt, a->team, CMP_GT, 1 - damage / t->hp)) t->hp = 0;
      break;
    case A_PREHISTORIC_WRATH: if (t->hp <= 0) a->damage *= 2; break;
  }
  return did;
}

// ---- damage ----
static int resolve_aura_farm(Runtime *rt, int target, double incoming, double *damage) {
  *damage = incoming;
  if (!can_receive_protection(rt, target)) return target;
  if (incoming < CARD(target)->hp) return target;
  int team = CARD(target)->team;
  if (active(rt, team) != target) return target;
  int piccolo = at(rt, team, 1);
  if (piccolo == NOCARD || piccolo == target || !alive(rt, piccolo) || CARD(piccolo)->def != D_PICCOLO || FL(piccolo, farmed)) return target;
  if (or0(S->boosts[CARD(piccolo)->team].noAbilities) > 0 && FL(piccolo, appearedOnField)) return target;
  int fatherhood = CARD(target)->def == D_KID_GOHAN;
  FL(piccolo, farmed) = 1;
  S->team[team][0] = piccolo;
  S->team[team][1] = target;
  boost_stats(rt, piccolo, 2);
  if (fatherhood) boost_stats(rt, piccolo, 1.5);
  *damage = 0;
  return piccolo;
}

static double deal_damage_(Runtime *rt, int attacker, int original_target, double mult, int bypass) {
  Card *a = CARD(attacker);
  a->flag[F_armageddonLethalThisHit] = 0;
  int target = original_target;
  int confused = a->confused > 0 || a->flag[F_eternalConfusion];
  int self_hit = confused && chance_raw(rt, CMP_LT, 0.5);
  if (self_hit) target = attacker;
  if (a->confused > 0 && !a->flag[F_eternalConfusion]) a->confused -= 1;
  if (self_hit) {
    int observer = active(rt, OTHER(a->team));
    if (observer != NOCARD && has_ability(rt, observer, A_BEYOND_COMPREHENSION)) boost_stats(rt, observer, 1.5);
  }
  Card *t = CARD(target);
  int frostbite_active = or0(t->counter[C_frostbite]) > 0 && !status_protected(rt, t->team);

  double damage = a->damage * mult;
  if (has_ability(rt, attacker, A_JAWS)) damage += t->damage;
  if (a->burn > 0) damage *= 0.85;
  Offense off = offensive(rt, attacker, target, damage);
  damage = off.damage;
  bypass = bypass || off.bypass;

  double executioner = S->boosts[a->team].executioner;
  if (truthy(executioner) && t->maxHp > 0 && t->hp / t->maxHp < 0.3) damage *= 1 + executioner / 100;
  if (a->blind && chance_roll(rt, a->team, CMP_GT, 0.4)) damage = 0;
  if (!bypass && !off.special && has_ability(rt, target, A_ALL_FATHER) && damage > 0) { double cost = t->maxHp / 5; damage = 0; t->hp -= cost; }
  if (status_protected(rt, t->team)) clear_statuses(rt, target);
  if (t->weakness) damage *= 1.3;

  t->flag[F_evadedThisHit] = 0;
  double before_defense = damage;
  int external = can_receive_protection(rt, target);
  if (!external) { t->flag[F_eternalDevotion] = 0; t->flag[F_dodgeLethal] = 0; t->shield = 0; t->counter[C_hpShield] = 0; }
  if (!bypass && external && t->flag[F_eternalDevotion]) { t->flag[F_eternalDevotion] = 0; damage = 0; }
  else if (!bypass && external && t->flag[F_dodgeLethal]) { t->flag[F_dodgeLethal] = 0; damage = 0; }
  else if (!bypass) {
    int veil = luminescent_holder(rt, t->team);
    double evades = or0(t->counter[C_luminescentEvades]);
    if (external && veil != NOCARD && evades < 2) {
      double chance = jmax(0.2, 0.4 - evades * 0.1);
      if (chance_roll(rt, t->team, CMP_LT, chance)) {
        t->counter[C_luminescentEvades] = evades + 1;
        t->flag[F_evadedThisHit] = 1;
        Card *v = CARD(veil);
        double base = orv(v->counter[C_normalDamage], v->damage), gain_now = or0(v->counter[C_luminescentVeilGain]);
        double room = jmax(0, base * 2 - gain_now), gain = jmin(room, jmax(0, before_defense) * 0.1);
        if (gain > 0) { v->damage += gain; v->counter[C_luminescentVeilGain] = gain_now + gain; }
        damage = 0;
      } else damage = defensive(rt, attacker, target, damage);
    } else damage = defensive(rt, attacker, target, damage);
  }
  if (!bypass && t->flag[F_evadedThisHit] && has_ability(rt, attacker, A_CONSTELLARSAGITTARIUS)) damage = before_defense * 2;

  double shielder = S->boosts[t->team].shielder;
  if (truthy(shielder)) damage *= (100 - shielder) / 100;
  double threshold = S->boosts[t->team].synthHuman;
  if (truthy(threshold) && DEF_TIME_STORM[t->def]) threshold *= 1.5;
  if (truthy(threshold) && damage < t->maxHp * threshold / 100) damage = 0;

  if (external && t->shield > 0 && damage > 0) { t->shield -= 1; damage = 0; }
  if (damage < 0) damage = jmax(-(t->maxHp - t->hp), damage);
  damage = isfinite(damage) ? ceil(damage) : t->hp;

  if (external && or0(t->counter[C_hpShield]) > 0 && damage > 0) {
    double absorbed = jmin(t->counter[C_hpShield], damage);
    t->counter[C_hpShield] -= absorbed;
    damage -= absorbed;
  }
  int xuanwu = damage > 0 && external ? water_shield(rt, t->team, target) : NOCARD;
  if (xuanwu != NOCARD) {
    double redirected = ceil(damage * 0.5);
    damage -= redirected;
    CARD(xuanwu)->hp -= jmin(CARD(xuanwu)->hp, redirected);
  }

  target = resolve_aura_farm(rt, target, damage, &damage);
  t = CARD(target);
  int deck_team = t->team;
  int long_reach = NOCARD;
  if (has_ability(rt, attacker, A_LONG_REACH) && active(rt, deck_team) == target && S->nteam[deck_team]) {
    int len = S->nteam[deck_team];
    int idx = chance_pickroll(rt, a->team, len);
    if (idx > len - 1) idx = len - 1;
    long_reach = S->team[deck_team][idx];
  }
  int hp_target = long_reach != NOCARD ? long_reach : target;
  Card *h = CARD(hp_target);
  if (has_ability(rt, attacker, A_DEFRAUD)) damage = jmin(damage, h->hp * 0.5);
  double applied = jmin(h->hp, damage);
  h->hp -= applied;

  double mirror_knight = S->boosts[h->team].mirrorKnight;
  if (applied > 0 && attacker != hp_target && truthy(mirror_knight) && alive(rt, attacker)) {
    double reflected = jmin(a->hp, applied * mirror_knight / 100);
    if (reflected > 0) a->hp -= reflected;
  }
  if (applied > 0 && or0(h->counter[C_bindFatePair]) > 0) {
    double pair = h->counter[C_bindFatePair];
    for (int i = 0; i < S->nteam[h->team]; i++) {
      int p = S->team[h->team][i];
      if (p != hp_target && alive(rt, p) && CARD(p)->counter[C_bindFatePair] == pair) { CARD(p)->hp -= jmin(CARD(p)->hp, applied); break; }
    }
  }
  if (long_reach != NOCARD && CARD(long_reach)->hp <= 0) {
    int idx = index_of(S->team[deck_team], S->nteam[deck_team], long_reach);
    if (idx > 0) {
      list_remove_at(S->team[deck_team], &S->nteam[deck_team], idx);
      CARD(long_reach)->dead = 1;
      list_push(S->fallen[CARD(long_reach)->team], &S->nfallen[CARD(long_reach)->team], long_reach);
    }
  }
  if (frostbite_active && t->hp > 0 && chance_raw(rt, CMP_LT, 0.5)) { double fd = jmin(t->hp, t->maxHp * 0.2); t->hp -= fd; }

  int observer = active(rt, OTHER(a->team));
  if (has_ability(rt, observer, A_AM_I_BEAUTIFUL)) {
    if (t->team == a->team) t->damage *= 0.8;
    else t->confused += 1;
  }
  if ((has_ability(rt, target, A_MEOW) || has_ability(rt, target, A_NEVER_FORGOTTEN)) && damage > 0)
    t->counter[C_damageTaken] = jmin(t->maxHp, or0(t->counter[C_damageTaken]) + damage);
  if (has_ability(rt, attacker, A_DISARM) && damage > 0) t->damage = jmax(0, t->damage - damage * 0.4);
  if (has_ability(rt, attacker, A_SHINY_STEAL) && damage > 0 && target != attacker) {
    double sd = t->damage * 0.1, sh = t->maxHp * 0.1;
    t->damage = jmax(0, t->damage - sd);
    t->maxHp = jmax(1, t->maxHp - sh);
    t->hp = jmin(t->hp, t->maxHp);
    a->damage += sd; a->maxHp += sh; a->hp += sh;
  }
  double flame = S->boosts[a->team].flameWizard;
  if (!status_protected(rt, t->team) && truthy(flame) && damage > 0 && chance_raw(rt, CMP_LT, flame / 100)) t->burn = 2;
  double phantom = S->boosts[a->team].phantom;
  if (!status_protected(rt, t->team) && truthy(phantom) && damage > 0 && chance_raw(rt, CMP_LT, phantom / 100)) t->stunned = jmax(1, t->stunned);

  if (has_ability(rt, target, A_CHIMERIC) && t->hp > 0 && t->hp <= t->maxHp / 2 && !t->flag[F_chimericFaded]) {
    t->flag[F_chimericFaded] = 1;
    t->maxHp /= 4; t->hp /= 4; t->damage /= 4;
  }
  target_retro(rt, attacker, target, damage);
  int did_regen = attacker_retro(rt, attacker, target, damage);
  if (has_ability(rt, target, A_REVEAL) && !t->flag[F_revealed] && t->hp > 0 && t->hp / t->maxHp < 0.65) { t->flag[F_revealed] = 1; t->hp = t->maxHp; }
  double vamp = S->boosts[a->team].vampireMatron;
  if (damage > 0 && truthy(vamp) && !did_regen && alive(rt, attacker) && vampire_matron_can_heal(rt, attacker))
    a->hp = jmin(a->maxHp, a->hp + damage * vamp / 100);
  if (t->hp <= 0) try_revive(rt, attacker, target);
  if (has_ability(rt, attacker, A_INFINITE_DAGGER_WORKS) && chance_roll(rt, a->team, CMP_GT, 0.5)) a->flag[F_extraTurn] = 1;
  return damage;
}
static double deal_damage(Runtime *rt, int attacker, int original_target, double mult, int bypass) { enter(rt); double r = deal_damage_(rt, attacker, original_target, mult, bypass); rt->depth--; return r; }

// ---- death ----
static void apply_on_death(Runtime *rt, int dead, int opponent, int skip_opponent_passives) {
  Card *d = CARD(dead);
  int team = d->team;
  int next = active(rt, team);
  int name = resolved_ability(rt, dead);
  if (!skip_opponent_passives && opponent != NOCARD) {
    Card *o = CARD(opponent);
    if (alive(rt, opponent) && has_ability(rt, opponent, A_GLORY_KILL)) {
      o->hp = o->maxHp;
      o->counter[C_gloryKills] = or0(o->counter[C_gloryKills]) + 1;
      double base = orv(o->counter[C_gloryBaseDamage], o->damage);
      o->counter[C_gloryBaseDamage] = base;
      o->damage = base * (1 + 0.5 * o->counter[C_gloryKills]);
    }
    if (alive(rt, opponent) && has_ability(rt, opponent, A_LIFE_TAP)) o->hp = jmin(o->maxHp, o->hp + o->maxHp * 0.2);
    if (alive(rt, opponent) && has_ability(rt, opponent, A_PREHISTORIC_WRATH)) o->damage *= 2;
    if (alive(rt, opponent) && has_ability(rt, opponent, A_ALL_FATHER))
      for (int i = 0; i < S->nteam[o->team]; i++) boost_stats(rt, S->team[o->team][i], 1.25);
  }
  if (d->flag[F_suppressOnDeath]) return;
  Boosts *b = &S->boosts[team];
  if (or0(b->noAbilities) > 0) {
    if (name == A_NIGHTMARE_MELODY && truthy(b->composerCount)) b->composerCount = jmax(0, or0(b->composerCount) - 1);
    return;
  }
  int gained[6], g = active_bonus(rt, dead, gained);
  if (g) {
    for (int i = 0; i < g; i++) WITH_ABILITY(dead, gained[i], { apply_on_death(rt, dead, opponent, 1); });
    return;
  }
  if (name == A_SPLIT_IN_TWO && !d->flag[F_sealed]) {
    static const int children[2] = {D_JOY, D_SORROW};
    int made[2];
    for (int i = 0; i < 2; i++) {
      int n = new_card(rt);
      Card *k = CARD(n);
      d = CARD(dead);
      memset(k, 0, sizeof *k);
      k->def = children[i];
      k->id = fnv(fnv(d->id, ":"), DEF_NAME[children[i]]);
      k->team = team;
      k->index = d->index;
      k->border = d->border;
      k->power = DEF_POWER[children[i]][d->border];
      k->maxHp = k->hp = d->maxHp * 0.5;
      k->damage = d->damage * 0.6;
      k->boss = DEF_BOSS[children[i]];
      k->identity = NOCARD;
      k->abilityOverride = AO_UNDEFINED;
      made[i] = n;
    }
    list_unshift(S->team[team], &S->nteam[team], made[1]);
    list_unshift(S->team[team], &S->nteam[team], made[0]);
  }
  d = CARD(dead);
  if (name == A_NIGHTMARE_MELODY && truthy(b->composerCount)) b->composerCount = jmax(0, or0(b->composerCount) - 1);
  if (name == A_HARD_BOILED) b->fossils = or0(b->fossils) + 3;
  if (name == A_EXTINCTION) b->fossils = or0(b->fossils) + 2;
  if (name == A_IMMINENT_DOOM && opponent != NOCARD && alive(rt, opponent) && !status_protected(rt, CARD(opponent)->team))
    CT(opponent, frostbite) = jmax(or0(CT(opponent, frostbite)), 2);
  if (name == A_GEHENNA) {
    int revive_count = S->nfallen[OTHER(team)], candidates[S->nfallen[team] + 1], n = 0;
    for (int i = S->nfallen[team] - 1; i >= 0 && n < revive_count; i--) if (S->fallen[team][i] != dead) candidates[n++] = S->fallen[team][i];
    double sd = orv(d->counter[C_normalDamage], d->damage) * 0.75, sh = orv(d->counter[C_normalMaxHp], d->maxHp) * 0.75;
    for (int i = 0; i < n; i++) {
      int ally = candidates[i], idx = index_of(S->fallen[team], S->nfallen[team], ally);
      if (idx >= 0) list_remove_at(S->fallen[team], &S->nfallen[team], idx);
      Card *k = CARD(ally);
      k->dead = 0; k->damage = sd; k->maxHp = sh; k->hp = sh; k->entered = 0;
      k->counter[C_normalDamage] = sd; k->counter[C_normalMaxHp] = sh;
      list_push(S->team[team], &S->nteam[team], ally);
    }
  }
  if (next == NOCARD || name < 0) return;
  Card *x = CARD(next);
  if (name == A_BLESSING) { x->damage += d->damage / 2; x->maxHp += d->maxHp / 2; x->hp += d->maxHp / 2; }
  if (name == A_MEOW) x->damage += or0(d->counter[C_damageTaken]) * 1.5;
  if (name == A_NEVER_FORGOTTEN) {
    double gain = or0(d->counter[C_damageTaken]) * 1.25;
    for (int i = 0; i < S->nteam[team]; i++) if (alive(rt, S->team[team][i])) CARD(S->team[team][i])->damage += gain;
  }
  if (name == A_WE_WANT_YOU) { x->damage *= 5; x->flag[F_diesAfterAttack] = 1; }
  if (name == A_BETTER_DAYS) {
    int revive[S->nfallen[team] + 1], n = 0;
    for (int i = 0; i < S->nfallen[team]; i++) if (S->fallen[team][i] != dead) revive[n++] = S->fallen[team][i];
    for (int i = 0; i < n; i++) {
      int ally = revive[i], idx = index_of(S->fallen[team], S->nfallen[team], ally);
      if (idx >= 0) list_remove_at(S->fallen[team], &S->nfallen[team], idx);
      Card *k = CARD(ally);
      k->dead = 0; k->hp = k->maxHp; k->entered = 0;
      list_push(S->team[team], &S->nteam[team], ally);
    }
  }
  if (name == A_HEART_LEGACY) { x->maxHp += d->maxHp; x->hp += d->maxHp; }
  if (name == A_TONIC) boost_stats(rt, next, 1.2);
  if (name == A_FUSION_HA) {
    if (chance_roll(rt, team, CMP_GT, 0.5)) { x->damage += d->damage * 0.5; x->maxHp += d->maxHp * 0.5; x->hp += d->maxHp * 0.5; }
  }
  if (name == A_DESTINY_SIGHT && can_receive_protection(rt, next)) x->flag[F_dodgeLethal] = 1;
  if (name == A_HOUSEWIFE_S_BLESSING) { boost_stats(rt, next, 2); x->stunned = 2; }
  if (name == A_ETERNAL_DEVOTION && can_receive_protection(rt, next)) x->flag[F_eternalDevotion] = 1;
  if (name == A_FINAL_STAND) {
    x->damage += d->damage * 0.25; x->maxHp += d->maxHp * 0.25; x->hp += d->maxHp * 0.25;
    if (can_receive_protection(rt, next)) x->shield += 1;
  }
}

static void resolve_deaths_(Runtime *rt) {
  int changed = 1;
  while (changed) {
    changed = 0;
    for (int team = 0; team < 2; team++) {
      int c = active(rt, team);
      if (c == NOCARD || CARD(c)->hp > 0) continue;
      Card *k = CARD(c);
      double ga = S->boosts[team].guardianAngel;
      if (truthy(ga) && !k->flag[F_guardianAngelUsed] && chance_raw(rt, CMP_LT, ga / 100)) {
        k->flag[F_guardianAngelUsed] = 1; k->hp = 1; changed = 1; continue;
      }
      if (has_ability(rt, c, A_DIVINE_ASCENSION) && !k->flag[F_awakened]) {
        k->flag[F_awakened] = 1;
        k->abilityOverride = A_MASTERED_ASCENSION;
        k->maxHp *= 1.5; k->damage *= 1.5; k->hp = k->maxHp;
        k->counter[C_normalDamage] = orv(k->counter[C_normalDamage], k->damage / 1.5) * 1.5;
        k->counter[C_normalMaxHp] = orv(k->counter[C_normalMaxHp], k->maxHp / 1.5) * 1.5;
        changed = 1; continue;
      }
      if (has_ability(rt, c, A_UNDYING)) {
        if (!k->flag[F_undyingActive]) { k->flag[F_undyingActive] = 1; k->counter[C_undyingTurns] = 1; k->hp = 1; changed = 1; continue; }
        if (or0(k->counter[C_undyingTurns]) > 0) { k->hp = 1; changed = 1; continue; }
      }
      if (has_ability(rt, c, A_UNHOLY_CREATURE)) {
        if (!k->flag[F_unholyActive]) {
          k->flag[F_unholyActive] = 1; k->counter[C_unholyTurns] = 2;
          k->counter[C_unholyActivatedTurn] = S->turn; k->counter[C_unholyLastTick] = S->turn;
          k->hp = 1; changed = 1; continue;
        }
        if (or0(k->counter[C_unholyTurns]) > 0) { k->hp = 1; changed = 1; continue; }
      }
      if (has_ability(rt, c, A_PARADOX) && !k->flag[F_paradox]) {
        k->flag[F_paradox] = 1; k->hp = 1;
        int opp = active(rt, OTHER(team));
        if (opp != NOCARD) CARD(opp)->hp = 0;
        changed = 1; continue;
      }
      int beyond = has_ability(rt, c, A_BEYOND_THE_GRAVE) && !k->flag[F_beyondGraveRevived];
      list_remove_at(S->team[team], &S->nteam[team], 0);
      rt->deathEpoch += 1;
      k->hp = 0; k->dead = 1;
      list_push(S->fallen[team], &S->nfallen[team], c);
      double ft = S->boosts[team].finalTestament;
      int inheritor = active(rt, team);
      if (inheritor != NOCARD && truthy(ft)) {
        Card *i = CARD(inheritor);
        double dmg = k->damage * ft / 100, hp = k->maxHp * ft / 100;
        i->damage += dmg; i->maxHp += hp; i->hp += hp;
      }
      if (!k->flag[F_mirrorImageReturned]) {
        for (int idx = S->nfallen[team] - 1; idx >= 0; idx--) {
          int m = S->fallen[team][idx];
          if (m == c || CARD(m)->flag[F_sealed] || !names_include(rt, m, A_MIRROR_IMAGE)) continue;
          if (chance_roll(rt, team, CMP_LE, 0.5)) continue;
          Card *mk = CARD(m);
          mk->flag[F_mirrorImageReturned] = 1; mk->dead = 0; mk->hp = mk->maxHp; mk->entered = 0;
          list_remove_at(S->fallen[team], &S->nfallen[team], idx);
          list_unshift(S->team[team], &S->nteam[team], m);
        }
      }
      int opponent = active(rt, OTHER(team));
      apply_on_death(rt, c, opponent, 0);
      if (beyond) {
        k = CARD(c);
        double base_max_hp = k->power * DEF_HP_MULTIPLIER[k->def], base_damage = k->power / 2;
        int idx = index_of(S->fallen[team], S->nfallen[team], c);
        if (idx >= 0) list_remove_at(S->fallen[team], &S->nfallen[team], idx);
        int n = new_card(rt);
        Card *r = CARD(n);
        k = CARD(c);
        *r = *k;
        r->id = fnv(k->id, ":btg");
        r->hp = base_max_hp / 2; r->maxHp = base_max_hp; r->damage = base_damage;
        r->entered = 0; r->dead = 0; r->identity = NOCARD; r->abilityOverride = AO_UNDEFINED; r->nbonus = 0;
        clear_card_state(r);
        r->flag[F_beyondGraveRevived] = 1;
        r->counter[C_normalDamage] = base_damage; r->counter[C_normalMaxHp] = base_max_hp;
        list_push(S->team[team], &S->nteam[team], n);
      }
      changed = 1;
    }
  }
}
static void resolve_deaths(Runtime *rt) { enter(rt); resolve_deaths_(rt); rt->depth--; }

// ---- turn structure ----
static void status_start(Runtime *rt, int attacker, int target) {
  Card *a = CARD(attacker), *t = CARD(target);
  if (status_protected(rt, a->team)) clear_statuses(rt, attacker);
  if (has_ability(rt, target, A_LIGHTNING_STRIKE) && alive(rt, target) && alive(rt, attacker)) deal_damage(rt, target, attacker, 0.75, 0);
  double pp = or0(a->counter[C_poisonPercent]), pf = or0(a->counter[C_poisonFlat]);
  if (truthy(pp)) a->hp = jmax(0, a->hp + pp * a->maxHp);
  else if (truthy(pf)) a->hp = jmax(0, a->hp - pf);
  if (a->flag[F_hanged]) a->hp -= a->maxHp * 0.25;
  if (has_ability(rt, target, A_DECAY)) a->damage *= 0.75;
  if (has_ability(rt, target, A_STARVATION)) boost_stats(rt, attacker, 0.75);
  if (has_ability(rt, target, A_PURIFYING_FIRE)) a->hp *= 0.7;
  if (has_ability(rt, attacker, A_SACRIFICIAL_TIDES)) t->hp -= t->maxHp * 0.2;
}

static void tick_global_unholy(Runtime *rt) {
  for (int team = 0; team < 2; team++) {
    for (int i = 0; i < S->nteam[team]; i++) {
      Card *k = CARD(S->team[team][i]);
      if (!k->flag[F_unholyActive]) continue;
      double activated = or0(k->counter[C_unholyActivatedTurn]);
      double last_tick = orv(k->counter[C_unholyLastTick], activated);
      if (S->turn <= activated || last_tick >= S->turn) continue;
      k->counter[C_unholyLastTick] = S->turn;
      k->counter[C_unholyTurns] = jmax(0, or0(k->counter[C_unholyTurns]) - 1);
      if (or0(k->counter[C_unholyTurns]) <= 0) k->hp = 0;
    }
  }
}

static void tick_ouroboros(Runtime *rt, int attacker) {
  Card *a = CARD(attacker);
  if (!a->flag[F_ouroborosActive]) return;
  a->counter[C_ouroborosTurns] = jmax(0, or0(a->counter[C_ouroborosTurns]) - 1);
  if (or0(a->counter[C_ouroborosTurns]) > 0) return;
  double d = or0(a->counter[C_ouroborosBonusDamage]), m = or0(a->counter[C_ouroborosBonusMaxHp]), h = or0(a->counter[C_ouroborosBonusHp]);
  a->damage = jmax(0, a->damage - d);
  a->maxHp = jmax(1, a->maxHp - m);
  a->hp = jmax(0, jmin(a->maxHp, a->hp - h));
  a->counter[C_ouroborosBonusDamage] = 0; a->counter[C_ouroborosBonusMaxHp] = 0; a->counter[C_ouroborosBonusHp] = 0;
  a->flag[F_ouroborosActive] = 0;
}

static void final_tail_and_undying(Runtime *rt, int attacker) {
  Card *a = CARD(attacker);
  if (has_ability(rt, attacker, A_FINAL_TAIL)) {
    a->counter[C_finalTail] = or0(a->counter[C_finalTail]) + 1;
    if (a->counter[C_finalTail] >= 3) a->hp = 0;
  }
  if (a->flag[F_undyingActive]) {
    a->counter[C_undyingTurns] = jmax(0, or0(a->counter[C_undyingTurns]) - 1);
    if (or0(a->counter[C_undyingTurns]) <= 0) a->hp = 0;
  }
}

static void status_end(Runtime *rt, int attacker) {
  Card *a = CARD(attacker);
  tick_ouroboros(rt, attacker);
  if (status_protected(rt, a->team)) {
    clear_statuses(rt, attacker);
    final_tail_and_undying(rt, attacker);
    return;
  }
  if (a->burn > 0) { a->hp -= a->maxHp * 0.1; a->burn -= 1; }
  if (or0(a->counter[C_bleed]) > 0) { a->hp -= a->maxHp * 0.15; a->counter[C_bleed] -= 1; }
  if (or0(a->counter[C_poisonTurns]) > 0) {
    a->counter[C_poisonTurns] -= 1;
    if (a->counter[C_poisonTurns] <= 0) { a->counter[C_poisonPercent] = 0; a->counter[C_poisonFlat] = 0; }
  }
  if (or0(a->counter[C_frostbite]) > 0) a->counter[C_frostbite] -= 1;
  if (or0(a->counter[C_death]) > 0 && !has_ability(rt, attacker, A_EROSION)) {
    a->counter[C_death] -= 1;
    if (a->counter[C_death] <= 0) a->hp = 0;
  }
  final_tail_and_undying(rt, attacker);
  if (a->weakness && or0(a->counter[C_weaknessTurns]) > 0) {
    a->counter[C_weaknessTurns] -= 1;
    if (a->counter[C_weaknessTurns] <= 0) a->weakness = 0;
  }
}

static void prepare_turn(Runtime *rt, int attacker) {
  Card *a = CARD(attacker);
  Boosts *composer = &S->boosts[a->team];
  if (has_ability(rt, attacker, A_COSMIC_RIVALRY)) {
    double bd = orv(a->counter[C_normalDamage], a->damage), bm = orv(a->counter[C_normalMaxHp], a->maxHp);
    a->damage += bd * 0.1; a->maxHp += bm * 0.1; a->hp += bm * 0.1;
    a->counter[C_cosmicRivalryDR] = jmin(1, or0(a->counter[C_cosmicRivalryDR]) + 0.1);
  }
  if (has_ability(rt, attacker, A_SHIELD_OF_AHIMSA)) {
    a->counter[C_ahimsaTurns] = or0(a->counter[C_ahimsaTurns]) + 1;
    if (jmod(a->counter[C_ahimsaTurns], 2) == 0) a->shield += 1;
  }
  if (or0(composer->composerCount) > 0) {
    composer->composerThreshold = jmax(0.6, (composer->composerThresholdSet ? composer->composerThreshold : 1) - 0.1);
    composer->composerThresholdSet = 1;
    int target = active(rt, OTHER(a->team));
    if (target != NOCARD && chance_roll(rt, a->team, CMP_GT, composer->composerThreshold)) CARD(target)->confused = 2;
  }
  if (has_ability(rt, attacker, A_PERISH) && or0(a->counter[C_perishTurns]) > 0) {
    a->counter[C_perishTurns] -= 1;
    if (a->counter[C_perishTurns] <= 0) {
      int target = active(rt, OTHER(a->team));
      a->hp = 0;
      if (target != NOCARD) CARD(target)->hp = 0;
      return;
    }
  }
  if (a->flag[F_naughtyListDrain]) boost_stats(rt, attacker, 0.9);
  if (has_ability(rt, attacker, A_TOIL)) boost_stats(rt, attacker, 0.85);
  if (has_ability(rt, attacker, A_BLOODLUST)) {
    if (a->flag[F_bloodlustFirstTurn]) a->flag[F_bloodlustFirstTurn] = 0;
    else a->damage += or0(a->counter[C_bloodlustBase]);
  }
  if (has_ability(rt, attacker, A_CONSTELLARAQUARIUS)) {
    if (a->hp < a->maxHp / 2) a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.3);
    else a->maxHp *= 1.25;
  }
  if (has_ability(rt, attacker, A_FULL_MOON)) {
    a->counter[C_fullMoon] = or0(a->counter[C_fullMoon]) + 1;
    if (jmod(a->counter[C_fullMoon], 2) == 0) {
      int target = active(rt, OTHER(a->team));
      if (target != NOCARD && alive(rt, target)) deal_damage(rt, target, target, 1, 0);
    }
  }
  if (has_ability(rt, attacker, A_DARK_QI_MANIPULATION) && !a->flag[F_awakened]) {
    a->counter[C_ascension] = or0(a->counter[C_ascension]) + 1;
    if (a->counter[C_ascension] <= 2) boost_stats(rt, attacker, 1.3);
    else a->flag[F_awakened] = 1;
  }
  if (has_ability(rt, attacker, A_IMMORTAL_ASCENSION) && !a->flag[F_awakened]) {
    a->counter[C_ascension] = or0(a->counter[C_ascension]) + 1;
    if (a->counter[C_ascension] <= 2) boost_stats(rt, attacker, 1.3);
    else a->flag[F_awakened] = 1;
  }
  if (has_ability(rt, attacker, A_UPHEAVAL)) {
    a->counter[C_upheaval] = or0(a->counter[C_upheaval]) + 1;
    if (jmod(a->counter[C_upheaval], 3) == 0) {
      a->damage *= 2;
      int target = active(rt, OTHER(a->team));
      if (target != NOCARD && !status_protected(rt, CARD(target)->team)) CARD(target)->stunned = jmax(1, CARD(target)->stunned);
    }
  }
  if (has_ability(rt, attacker, A_FIRST_TAIL) && or0(a->counter[C_tail]) < 9) {
    a->counter[C_tail] = or0(a->counter[C_tail]) + 1;
    boost_stats(rt, attacker, 1.2);
  }
  if (has_ability(rt, attacker, A_SHAPESHIFTER) || a->flag[F_shapeshifterActive]) {
    a->flag[F_shapeshifterActive] = 1;
    int shape = random_battle_card(rt);
    a->identity = shape;
    a->abilityOverride = AO_UNDEFINED;
    a->entered = 0;
  }
  if (has_ability(rt, attacker, A_GRIND)) {
    a->counter[C_grind] = or0(a->counter[C_grind]) + 1;
    if (a->counter[C_grind] <= 5) boost_stats(rt, attacker, 1.1);
  }
  if (has_ability(rt, attacker, A_PATIENCE)) a->damage *= 1.3;
  if (has_ability(rt, attacker, A_SAFEGUARDING)) {
    for (int i = 1; i < S->nteam[a->team]; i++) {
      Card *d = CARD(S->team[a->team][i]);
      if (!(DEF_SETS[d->def] & SET_DRAGON)) continue;
      d->damage *= 1.2; d->maxHp *= 1.2; d->hp = d->maxHp;
    }
  }
  if (has_ability(rt, attacker, A_ABSOLUTE_SOVEREIGNTY))
    for (int i = 0; i < S->nteam[a->team]; i++) boost_stats(rt, S->team[a->team][i], 1.1);
  if (has_ability(rt, attacker, A_WORLD_CREATION)) {
    a->counter[C_worldCreation] = or0(a->counter[C_worldCreation]) + 1;
    if (jmod(a->counter[C_worldCreation], 3) == 0) boost_stats(rt, attacker, 2);
  }
  if (has_ability(rt, attacker, A_PERSISTENT)) {
    double normal = orv(a->counter[C_normalDamage], a->damage);
    if (a->damage < normal) a->damage = normal;
  }
  if (has_ability(rt, attacker, A_SKY_DROP)) a->counter[C_drop] = or0(a->counter[C_drop]) + 1;
  if (has_ability(rt, attacker, A_SNOWBOUND)) {
    a->counter[C_snowbound] = or0(a->counter[C_snowbound]) + 1;
    if (jmod(a->counter[C_snowbound], 2) == 0) a->stunned = jmax(1, a->stunned);
  }
  if (has_ability(rt, attacker, A_DEFENSIVE_MANEUVER)) {
    a->counter[C_defensiveManeuver] = or0(a->counter[C_defensiveManeuver]) + 1;
    if (jmod(a->counter[C_defensiveManeuver], 2) == 0) a->shield += 1;
  }
}

static void before_attack(Runtime *rt, int attacker) {
  Card *a = CARD(attacker);
  if (has_ability(rt, attacker, A_LIFE_TAP)) a->hp *= 0.85;
  if (has_ability(rt, attacker, A_BLADE_OF_MIQUELLA)) a->counter[C_waterfowl] = or0(a->counter[C_waterfowl]) + 1;
  if (has_ability(rt, attacker, A_VOID_HEART)) {
    a->counter[C_void_] = or0(a->counter[C_void_]) + 1;
    if (a->counter[C_void_] >= 3) { a->counter[C_void_] = 0; a->flag[F_voidReady] = 1; a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.25); }
  }
  int target = active(rt, OTHER(a->team));
  if (target != NOCARD && has_ability(rt, attacker, A_BLOOD_BATH)) {
    Card *t = CARD(target);
    double stolen = jmax(0, t->hp * 0.25);
    t->hp -= stolen;
    a->hp = jmin(a->maxHp, a->hp + stolen);
  }
  if (has_ability(rt, attacker, A_LAZY)) { a->hp = jmin(a->maxHp, a->hp + a->damage * 2); a->damage *= 0.9; }
  if (target != NOCARD && has_ability(rt, attacker, A_FORBIDDEN_BANQUET)) steal_stats(rt, target, attacker, 0.15);
  if (has_ability(rt, attacker, A_REJUVENATE)) a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.35);
  if (has_ability(rt, attacker, A_FIRST_PROGENITOR)) a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.1);
  if (has_ability(rt, attacker, A_TWILIGHT_SPARKLE) && chance_roll(rt, a->team, CMP_GT, 0.6)) a->hp = a->maxHp;
  if (target != NOCARD && has_ability(rt, attacker, A_VIRAL_BREATH)) CARD(target)->hp -= CARD(target)->maxHp * 0.25;
  if (has_ability(rt, attacker, A_HERBAL_ALCHEMY)) {
    a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.2);
    if (chance_roll(rt, a->team, CMP_GT, 0.5)) a->damage *= 1.3;
  }
  if (has_ability(rt, attacker, A_COMBATANT)) a->hp = jmin(a->maxHp, a->hp + a->maxHp * 0.1);
}

static double attack_count(Runtime *rt, int attacker) {
  double bonus = or0(CT(attacker, attacks)), base = 1;
  if (has_ability(rt, attacker, A_RAPID_BLOWS)) base = jmax(base, 3);
  if (has_ability(rt, attacker, A_CHAINSAW)) base = jmax(base, 8);
  if (has_ability(rt, attacker, A_FIREPOWER)) base = jmax(base, 5);
  if (has_ability(rt, attacker, A_BEHAVIORAL_THERAPY)) base = jmax(base, 2);
  return base + bonus;
}

static int can_normal_attack(Runtime *rt, int attacker) {
  if (has_ability(rt, attacker, A_RULER_OF_HUMANS)) return 0;
  if (has_ability(rt, attacker, A_DAGGER_STORM) || has_ability(rt, attacker, A_NAUGHTY_OR_NICE)
      || has_ability(rt, attacker, A_MEOW) || has_ability(rt, attacker, A_NEVER_FORGOTTEN)
      || has_ability(rt, attacker, A_ORIGIN) || has_ability(rt, attacker, A_LASER_GUN)
      || has_ability(rt, attacker, A_LOTUS_SUTRA)) return 0;
  if (has_ability(rt, attacker, A_SKY_DROP)) { double d = CT(attacker, drop); return truthy(d) && jmod(d, 2) == 0; }
  return 1;
}

static void do_lotus_sutra(Runtime *rt, int attacker) {
  Card *a = CARD(attacker);
  int team = a->team, dead_ally = NOCARD;
  if (!a->flag[F_lotusReviveUsed])
    for (int i = S->nfallen[team] - 1; i >= 0; i--) if (S->fallen[team][i] != attacker) { dead_ally = S->fallen[team][i]; break; }
  if (dead_ally != NOCARD) {
    a->flag[F_lotusReviveUsed] = 1;
    int idx = index_of(S->fallen[team], S->nfallen[team], dead_ally);
    if (idx >= 0) list_remove_at(S->fallen[team], &S->nfallen[team], idx);
    Card *k = CARD(dead_ally);
    k->dead = 0; k->hp = k->maxHp * 0.5; k->entered = 0;
    list_push(S->team[team], &S->nteam[team], dead_ally);
    return;
  }
  int target = NOCARD;  // the first ally with the lowest HP fraction (his stable sort, first element)
  for (int i = 0; i < S->nteam[team]; i++) {
    int c = S->team[team][i];
    if (c == attacker || !alive(rt, c)) continue;
    if (target == NOCARD || CARD(c)->hp / CARD(c)->maxHp - CARD(target)->hp / CARD(target)->maxHp < 0) target = c;
  }
  if (target == NOCARD) return;
  Card *t = CARD(target);
  t->hp = jmin(t->maxHp, t->hp + t->maxHp * 0.5);
  if (t->hp >= t->maxHp) {
    int idx = index_of(S->team[team], S->nteam[team], attacker);
    if (idx >= 0 && S->nteam[team] > 1) { list_remove_at(S->team[team], &S->nteam[team], idx); list_push(S->team[team], &S->nteam[team], attacker); }
  }
}

static void do_origin(Runtime *rt, int attacker) {
  int enemy_team = OTHER(CARD(attacker)->team);
  for (int hit = 0; hit < 4; hit++) {
    int deck[S->nteam[enemy_team] + 1], n = 0;
    for (int i = 0; i < S->nteam[enemy_team]; i++) if (alive(rt, S->team[enemy_team][i])) deck[n++] = S->team[enemy_team][i];
    if (!n || !alive(rt, attacker)) break;
    int target = deck[chance_pick(rt, n)];
    deal_damage(rt, attacker, target, 0.5, 0);
    resolve_deaths(rt);
  }
}

static void do_laser_gun(Runtime *rt, int attacker) {
  Card *a = CARD(attacker);
  if (!a->flag[F_laserCharged]) { a->flag[F_laserCharged] = 1; return; }
  a->flag[F_laserCharged] = 0;
  int enemy_team = OTHER(a->team);
  double targets = jmin(3, or0(S->boosts[a->team].fossils) + 1);
  int list[S->nteam[enemy_team] + 1], n = 0;
  for (int i = 0; i < S->nteam[enemy_team] && i < targets; i++) list[n++] = S->team[enemy_team][i];
  for (int i = 0; i < n; i++) {
    if (!alive(rt, attacker) || !alive(rt, list[i])) continue;
    deal_damage(rt, attacker, list[i], 0.75, 0);
  }
  resolve_deaths(rt);
}

static void apply_collateral(Runtime *rt, int attacker, int target, double dealt) {
  if (!(dealt > 0)) return;
  int enemy_team = OTHER(CARD(attacker)->team);
  if (has_ability(rt, attacker, A_RAILGUN)) {
    double splash = ceil(dealt * 0.3);
    for (int i = 0; i < S->nteam[enemy_team]; i++) { Card *e = CARD(S->team[enemy_team][i]); if (e->hp > 0) e->hp -= jmin(e->hp, splash); }
  }
  if (has_ability(rt, attacker, A_OUTSHINE)) {
    int idx = index_of(S->team[enemy_team], S->nteam[enemy_team], target);
    int next = idx >= 0 ? at(rt, enemy_team, idx + 1) : at(rt, enemy_team, 1);
    if (next != NOCARD && CARD(next)->hp > 0) {
      double before = CARD(next)->hp;
      CARD(next)->hp -= jmin(CARD(next)->hp, dealt);
      if (before > 0 && CARD(next)->hp <= 0) FL(next, suppressOnDeath) = 1;
    }
  }
}

static void process_team_turn_abilities(Runtime *rt, int moved_team, int moved) {
  int dispel = active(rt, OTHER(moved_team));
  if (dispel != NOCARD && alive(rt, dispel) && has_ability(rt, dispel, A_DISPEL) && alive(rt, moved)) {
    double drained = CARD(moved)->damage * 0.2;
    CARD(moved)->damage = jmax(0, CARD(moved)->damage - drained);
    CARD(dispel)->hp = jmin(CARD(dispel)->maxHp, CARD(dispel)->hp + drained);
  }
  for (int i = 0; i < S->nteam[moved_team]; i++) {
    int h = S->team[moved_team][i];
    if (!alive(rt, h) || !has_ability(rt, h, A_HEALING_MIRACLE) || h == moved) continue;
    CT(h, healingMiracle) = or0(CT(h, healingMiracle)) + 1;
    if (CT(h, healingMiracle) >= 3) { CT(h, healingMiracle) = 0; CARD(h)->hp = jmin(CARD(h)->maxHp, CARD(h)->hp + CARD(h)->maxHp); }
  }
}

static void do_dagger_storm(Runtime *rt, int attacker) {
  static const double mults[3] = {0.5, 1, 2};
  int enemy_team = OTHER(CARD(attacker)->team);
  for (int i = 0; i < 3; i++) {
    int target = active(rt, enemy_team);
    if (target == NOCARD || !alive(rt, attacker)) break;
    deal_damage(rt, attacker, target, mults[i], 0);
    resolve_deaths(rt);
  }
}

static void do_naughty_or_nice(Runtime *rt, int attacker) {
  int target = active(rt, OTHER(CARD(attacker)->team));
  if (target == NOCARD || !alive(rt, attacker)) return;
  if (chance_roll(rt, CARD(attacker)->team, CMP_LT, 0.8)) { deal_damage(rt, attacker, target, 4, 0); resolve_deaths(rt); }
  else CARD(target)->hp = jmin(CARD(target)->maxHp, CARD(target)->hp + CARD(target)->maxHp * 0.5);
}

static void do_turn(Runtime *rt, int attacker) {
  int team = CARD(attacker)->team, enemy_team = OTHER(team);
  int target = active(rt, enemy_team);
  if (target == NOCARD || !alive(rt, attacker)) return;
  prepare_turn(rt, attacker);
  resolve_deaths(rt);
  if (!alive(rt, attacker)) return;
  target = active(rt, enemy_team);
  if (target == NOCARD) return;
  status_start(rt, attacker, target);
  resolve_deaths(rt);
  if (!alive(rt, attacker)) return;
  target = active(rt, enemy_team);
  if (target == NOCARD) return;

  before_attack(rt, attacker);
  if (has_ability(rt, attacker, A_LOTUS_SUTRA)) do_lotus_sutra(rt, attacker);
  else if (has_ability(rt, attacker, A_ORIGIN)) do_origin(rt, attacker);
  else if (has_ability(rt, attacker, A_LASER_GUN)) do_laser_gun(rt, attacker);
  else if (has_ability(rt, attacker, A_DAGGER_STORM)) do_dagger_storm(rt, attacker);
  else if (has_ability(rt, attacker, A_NAUGHTY_OR_NICE)) do_naughty_or_nice(rt, attacker);

  if (has_ability(rt, attacker, A_CHAOS_DESTRUCTION) && chance_roll(rt, team, CMP_GT, 0.5)) {
    int *deck = S->team[enemy_team];
    if (S->nteam[enemy_team] > 1) {
      int swap = 1 + chance_pick(rt, S->nteam[enemy_team] - 1);
      int t = deck[0]; deck[0] = deck[swap]; deck[swap] = t;
      target = deck[0];
      on_entry(rt, target);
      resolve_deaths(rt);
    }
    FL(attacker, chaosTriple) = 1;
  }

  if (can_normal_attack(rt, attacker)) {
    int dance = has_ability(rt, attacker, A_BLADE_OF_MIQUELLA) && jmod(or0(CT(attacker, waterfowl)), 3) == 0;
    double count = dance ? 3 : attack_count(rt, attacker);
    for (int i = 0; i < count; i++) {
      target = active(rt, enemy_team);
      if (target == NOCARD || !alive(rt, attacker)) break;
      int critical = has_ability(rt, attacker, A_JACKPOT) && chance_roll(rt, team, CMP_LT, 0.3);
      double dealt = deal_damage(rt, attacker, target, dance ? 0.5 : critical ? 1.5 : 1, 0);
      int armageddon_lethal = FL(attacker, armageddonLethalThisHit);
      if (has_ability(rt, attacker, A_FORTIFY) && alive(rt, attacker)) {
        double block = CARD(attacker)->maxHp * 0.4, remaining = or0(CT(attacker, blockHp));
        if (ceil(remaining / block) < 3) CT(attacker, blockHp) = remaining + block;
      }
      if (critical && alive(rt, attacker) && alive(rt, target)) deal_damage(rt, attacker, target, 0.5, 0);
      apply_collateral(rt, attacker, target, dealt);
      resolve_deaths(rt);
      int chain = 0;
      while (FL(attacker, insatiableAttack) && alive(rt, attacker) && active(rt, enemy_team) != NOCARD) {
        if (++chain > 64) { FL(attacker, insatiableAttack) = 0; break; }
        FL(attacker, insatiableAttack) = 0;
        deal_damage(rt, attacker, active(rt, enemy_team), 1, 0);
        resolve_deaths(rt);
      }
      if (has_ability(rt, attacker, A_BLACK_FLASH) && alive(rt, attacker) && CARD(target)->hp > 0) deal_damage(rt, attacker, target, 0.5, 1);
      resolve_deaths(rt);

      double storm = S->boosts[team].stormSpirit;
      int storm_target = active(rt, enemy_team);
      if (truthy(storm) && storm_target != NOCARD && alive(rt, attacker) && (alive(rt, target) || armageddon_lethal)
          && chance_raw(rt, CMP_LT, storm / 100)) {
        if (armageddon_lethal && !alive(rt, target)) FL(attacker, suppressArmageddonOnce) = 1;
        double storm_damage = deal_damage(rt, attacker, storm_target, 0.5, 0);
        apply_collateral(rt, attacker, storm_target, storm_damage);
        resolve_deaths(rt);
      }
    }
  }

  int creep_target = active(rt, enemy_team);
  if (creep_target != NOCARD && alive(rt, attacker)) {
    int creeps[S->nteam[team] + 1], n = 0;  // teams.slice(1): a snapshot of the list
    for (int i = 1; i < S->nteam[team]; i++) creeps[n++] = S->team[team][i];
    for (int i = 0; i < n; i++) {
      int creep = creeps[i];
      if (has_ability(rt, creep, A_CREEP) && alive(rt, creep) && active(rt, enemy_team) != NOCARD) {
        deal_damage(rt, creep, active(rt, enemy_team), 0.25, 0);
        resolve_deaths(rt);
      }
    }
  }

  int current = active(rt, enemy_team);
  if (current != NOCARD && alive(rt, current) && alive(rt, attacker)) {
    double berserker = S->boosts[CARD(current)->team].berserker;
    int counter = (truthy(berserker) && chance_raw(rt, CMP_LT, berserker / 100))
      || has_ability(rt, current, A_HATRED) || has_ability(rt, current, A_PERSEVERANCE) || has_ability(rt, current, A_SPIKES)
      || has_ability(rt, current, A_BLOOD_DRINKER) || has_ability(rt, current, A_STOLEN_SPOTLIGHT) || has_ability(rt, current, A_POKE_THE_BEAST)
      || (has_ability(rt, current, A_ABSOLUTE_APEX) && or0(S->boosts[CARD(current)->team].fossils) > 2);
    if (counter) {
      // his debug message names the counter: the same hasAbility chain runs (End Times can fail any of them)
      if (!has_ability(rt, current, A_HATRED) && !has_ability(rt, current, A_PERSEVERANCE) && !has_ability(rt, current, A_SPIKES)
          && !has_ability(rt, current, A_BLOOD_DRINKER) && !has_ability(rt, current, A_STOLEN_SPOTLIGHT)
          && !has_ability(rt, current, A_POKE_THE_BEAST)) has_ability(rt, current, A_ABSOLUTE_APEX);
      deal_damage(rt, current, attacker, has_ability(rt, current, A_PERSEVERANCE) ? 0.1 : 1, 0);
    }
  }

  if (has_ability(rt, attacker, A_MARTIAL_WILL) && alive(rt, attacker)) CARD(attacker)->damage *= 1.3;

  if (has_ability(rt, attacker, A_ETERNAL_VOYAGE) && alive(rt, attacker)) {
    int *deck = S->team[team], n = S->nteam[team], self = index_of(deck, n, attacker);
    int choices[n + 1], m = 0;
    for (int i = 0; i < n; i++) if (i != self) choices[m++] = i;
    if (self >= 0 && m) { int swap = choices[chance_pick(rt, m)]; int t = deck[self]; deck[self] = deck[swap]; deck[swap] = t; }
  }

  if (FL(attacker, diesAfterAttack) && alive(rt, attacker)) CARD(attacker)->hp = 0;

  status_end(rt, attacker);
  process_team_turn_abilities(rt, team, attacker);
  resolve_deaths(rt);

  double lock = or0(S->boosts[team].noAbilities);
  if (lock > 0) S->boosts[team].noAbilities = lock > 1 ? lock - 1 : 0;
}

static void process_divination(Runtime *rt) {
  int all[S->nteam[0] + S->nfallen[0] + S->nteam[1] + S->nfallen[1] + 1], n = 0;
  for (int t = 0; t < 2; t++) {
    for (int i = 0; i < S->nteam[t]; i++) all[n++] = S->team[t][i];
    for (int i = 0; i < S->nfallen[t]; i++) all[n++] = S->fallen[t][i];
  }
  for (int i = 0; i < n; i++) {
    int c = all[i];
    if (!has_ability(rt, c, A_DIVINATION) || FL(c, divinationFired)) continue;
    double moves = or0(CT(c, divinationMoves));
    if (moves <= 0) continue;
    CT(c, divinationMoves) = moves - 1;
    if (CT(c, divinationMoves) <= 0) {
      FL(c, divinationFired) = 1;
      int target = active(rt, OTHER(CARD(c)->team));
      if (target != NOCARD) { deal_damage(rt, c, target, 3, 1); resolve_deaths(rt); }
    }
  }
}

static void grow_hidden_in_depths(Runtime *rt, int moving) {
  if (moving != 0) return;
  for (int i = 1; i < S->nteam[0]; i++) {
    int c = S->team[0][i];
    if (!has_ability(rt, c, A_HIDDEN_IN_THE_DEPTHS)) continue;
    Card *k = CARD(c);
    double bd = orv(k->counter[C_normalDamage], k->damage), bm = orv(k->counter[C_normalMaxHp], k->maxHp);
    double db = or0(k->counter[C_hiddenDepthsBonusDamage]), hb = or0(k->counter[C_hiddenDepthsBonusHp]);
    double add_d = jmin(bd * 0.1, jmax(0, bd * 2 - db)), add_h = jmin(bm * 0.1, jmax(0, bm * 2 - hb));
    k->damage += add_d; k->maxHp += add_h; k->hp += add_h;
    k->counter[C_hiddenDepthsBonusDamage] = db + add_d;
    k->counter[C_hiddenDepthsBonusHp] = hb + add_h;
  }
}

static int schedule_extra_turns(Runtime *rt, int attacker) {
  Card *a = CARD(attacker);
  int extra = a->flag[F_extraTurn];
  a->flag[F_extraTurn] = 0;
  if (!a->flag[F_onBonusTurn]) {
    double count = 0;
    if (has_ability(rt, attacker, A_BERSERK) && a->hp / a->maxHp < 0.5) count += 1;
    if (has_ability(rt, attacker, A_MELANCHOLY) && a->hp / a->maxHp > 0.5) count += 2;
    if (has_ability(rt, attacker, A_HASTE)) count += 1;
    if (has_ability(rt, attacker, A_FIRST_PROGENITOR)) count += 1;
    if (has_ability(rt, attacker, A_THE_WORLD)) {
      if (a->flag[F_worldCooldown]) a->flag[F_worldCooldown] = 0;
      else { count += 2; a->flag[F_worldCooldown] = 1; }
    }
    if (has_ability(rt, attacker, A_ACCELERATE)) {
      a->counter[C_turnsPerTurn] = or0(a->counter[C_turnsPerTurn]) + 1;
      count += a->counter[C_turnsPerTurn];
    }
    if (count > 0) {
      a->counter[C_extraTurns] = count;
      // his debug message lists the sources: the same hasAbility calls run (End Times can roll in each)
      if (has_ability(rt, attacker, A_BERSERK)) (void)(a->hp / a->maxHp < 0.5);
      if (has_ability(rt, attacker, A_MELANCHOLY)) (void)(a->hp / a->maxHp > 0.5);
      has_ability(rt, attacker, A_HASTE);
      has_ability(rt, attacker, A_FIRST_PROGENITOR);
      has_ability(rt, attacker, A_THE_WORLD);
      has_ability(rt, attacker, A_ACCELERATE);
    }
  }
  if (or0(a->counter[C_extraTurns]) > 0) { a->counter[C_extraTurns] -= 1; a->flag[F_onBonusTurn] = 1; extra = 1; }
  else a->flag[F_onBonusTurn] = 0;
  return extra;
}

static void resolve_constellar_arts(Runtime *rt) {
  for (int team = 0; team < 2; team++) {
    for (int i = 0; i < S->nteam[team]; i++) {
      Card *k = CARD(S->team[team][i]);
      if (DEF_ABILITY[k->def] == A_CONSTELLAR && !(k->abilityOverride >= 0)) k->abilityOverride = random_constellar(rt);
    }
    int count = 0;
    for (int i = 0; i < S->nteam[team]; i++) if (CARD(S->team[team][i])->def == D_ASTRAEUS) count++;
    for (int i = 0; i < S->nteam[team]; i++) {
      int c = S->team[team][i];
      if (ability(rt, c) == A_CONSTELLARGEMINI && !FL(c, constellarGeminiApplied)) { FL(c, constellarGeminiApplied) = 1; boost_stats(rt, c, 1 + count * 0.5); }
    }
  }
}

static const int NEXT_TEAM_FIRST_TURN = 0;

int simulate(Runtime *rt, int max_turns, const Counters *resume, void (*on_turn)(Runtime *, const Counters *)) {
  if (resume) rt->deathEpoch = resume->deathEpoch;
  else { rt->deathEpoch = 0; resolve_constellar_arts(rt); }
  double without_deaths = resume ? resume->turnsWithoutDeaths : 0;
  double last_epoch = resume ? resume->lastDeathEpoch : rt->deathEpoch;
  (void)NEXT_TEAM_FIRST_TURN;
  while (S->nteam[0] && S->nteam[1] && S->turn < max_turns) {
    if (on_turn) { Counters c = {without_deaths, last_epoch, rt->deathEpoch}; on_turn(rt, &c); }
    S->turn += 1;
    tick_global_unholy(rt);
    resolve_deaths(rt);
    int attacker = active(rt, S->moving), defender = active(rt, OTHER(S->moving));
    if (attacker == NOCARD || defender == NOCARD) break;
    on_entry(rt, attacker);
    defender = active(rt, OTHER(S->moving));
    if (defender != NOCARD) on_entry(rt, defender);
    resolve_deaths(rt);
    attacker = active(rt, S->moving);
    defender = active(rt, OTHER(S->moving));
    if (attacker == NOCARD || defender == NOCARD) break;
    if (rt->deathEpoch != last_epoch) { without_deaths = 0; last_epoch = rt->deathEpoch; }
    without_deaths += 1;
    if (without_deaths >= 150) {
      CARD(attacker)->hp = 0;
      CARD(defender)->hp = 0;
      resolve_deaths(rt);
      continue;
    }
    do_turn(rt, attacker);
    process_divination(rt);
    grow_hidden_in_depths(rt, S->moving);
    resolve_deaths(rt);
    if (!S->nteam[0] || !S->nteam[1]) break;

    int still = active(rt, S->moving);
    int extra = still == attacker && alive(rt, attacker) ? schedule_extra_turns(rt, attacker) : 0;
    if (!extra) {
      int next_team = OTHER(S->moving), next = active(rt, next_team);
      if (next != NOCARD && status_protected(rt, next_team)) clear_statuses(rt, next);
      int frozen_source = active(rt, S->moving);
      int can_freeze = next != NOCARD && !FL(next, frozenSolitudeFirstTurnUsed) && frozen_source != NOCARD
                       && has_ability(rt, frozen_source, A_FROZEN_SOLITUDE);
      if (next != NOCARD && !FL(next, frozenSolitudeFirstTurnUsed)) FL(next, frozenSolitudeFirstTurnUsed) = 1;
      if (can_freeze && !status_protected(rt, next_team)) CT(next, videoFrozen) = 1;
      else if (next != NOCARD && or0(CT(next, videoFrozen)) > 0) { CT(next, videoFrozen) = 0; S->moving = next_team; }
      else if (next != NOCARD && CARD(next)->stunned > 0) CARD(next)->stunned -= 1;
      else if (next != NOCARD && FL(next, slowed)) {
        CT(next, slowed) = or0(CT(next, slowed)) + 1;
        if (or0(CT(next, slowTurns)) > 0) { CT(next, slowTurns) -= 1; if (CT(next, slowTurns) <= 0) FL(next, slowed) = 0; }
        if (jmod(CT(next, slowed), 2) == 0) S->moving = next_team;
      } else S->moving = next_team;
    }
  }
  return S->nteam[0] ? (S->nteam[1] ? 2 : 0) : (S->nteam[1] ? 1 : 2);
}

// ---- keys for loading his start-of-battle state ----
#define NAME_ITEM(name) #name,
static const char *const FLAG_NAMES[NFLAG] = {FLAG_LIST(NAME_ITEM)};
static const char *const COUNTER_NAMES[NCOUNTER] = {COUNTER_LIST(NAME_ITEM)};
static int lookup(const char *const *names, int n, const char *key) {
  for (int i = 0; i < n; i++) {
    size_t len = strlen(names[i]);
    if (names[i][len - 1] == '_' ? strncmp(names[i], key, len - 1) == 0 && key[len - 1] == 0 : strcmp(names[i], key) == 0) return i;
  }
  return -1;
}
int ce_flag_index(const char *key) { return lookup(FLAG_NAMES, NFLAG, key); }
int ce_counter_index(const char *key) { return lookup(COUNTER_NAMES, NCOUNTER, key); }
int ce_ability_index(const char *name) { for (int i = 0; i < NABILITY; i++) if (strcmp(ABILITY_NAMES[i], name) == 0) return i; return -1; }
int ce_def_index(const char *name) { for (int i = 0; i < NDEF; i++) if (strcmp(DEF_NAME[i], name) == 0) return i; return -1; }

// ---- compact copies (search.c's saved turns): the scalars, then the used part of each list, then the cards ----
size_t state_packed_size(const State *s) {
  return offsetof(State, team) + sizeof(int) * (size_t)(s->nteam[0] + s->nteam[1] + s->nfallen[0] + s->nfallen[1])
    + sizeof(Struck) * (size_t)s->nstruck + sizeof(Card) * (size_t)s->ncard;
}
#define PUT(src, bytes) do { size_t b_ = (bytes); memcpy(p, (src), b_); p += b_; } while (0)
#define GET(dst, bytes) do { size_t b_ = (bytes); memcpy((dst), p, b_); p += b_; } while (0)
void state_pack(const State *s, void *out) {
  char *p = out;
  PUT(s, offsetof(State, team));
  for (int t = 0; t < 2; t++) { PUT(s->team[t], sizeof(int) * (size_t)s->nteam[t]); PUT(s->fallen[t], sizeof(int) * (size_t)s->nfallen[t]); }
  PUT(s->struck, sizeof(Struck) * (size_t)s->nstruck);
  PUT(s->card, sizeof(Card) * (size_t)s->ncard);
}
void state_unpack(const void *in, State *s) {
  const char *p = in;
  GET(s, offsetof(State, team));
  for (int t = 0; t < 2; t++) { GET(s->team[t], sizeof(int) * (size_t)s->nteam[t]); GET(s->fallen[t], sizeof(int) * (size_t)s->nfallen[t]); }
  GET(s->struck, sizeof(Struck) * (size_t)s->nstruck);
  GET(s->card, sizeof(Card) * (size_t)s->ncard);
}
