// Label search over the C engine: sim_js/search.ts in C, with the same node order, tape, playout pool and random
// generator, so for the same battle and options both give bit-identical answers. A branch is the battle saved at
// the start of the turn it split in plus the choices since; the engine packs the battle at each turn start into
// one buffer, and a branch keeps that copy (only the cards in use are copied).
#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"

typedef struct Saved { int refs; Counters counters; char state[]; } Saved;  // state: the battle, packed (state_pack)
typedef struct { double p; Saved *saved; int n; int *choices; } Node;
typedef struct { int nodeBudget, rollouts, maxTurns; double sampleBelow, rolloutError; unsigned seed; } Options;
typedef struct { double a, b, draw, turns; int exact, nodes, playouts, overflow; } Result;  // turns: expected battle length

#define MAXPROBS 512
typedef struct {
  const int *choices; int n, i;            // the branch's choices, and the next one to replay
  uint32_t *random;                         // playouts: mulberry state; NULL while expanding
  jmp_buf branch;                           // a new chance point while expanding: back to play()
  double probs[MAXPROBS]; int nprobs;
  Saved *saved; int from;                   // the latest saved turn and the tape position when it was saved
  char *turn_buffer; size_t turn_size; Counters turn_counters; int have_turn;  // the latest turn start, packed
} Tape;

static double mulberry(uint32_t *s) {
  *s += 0x6d2b79f5u;
  uint32_t t = (*s ^ (*s >> 15)) * (*s | 1u);
  t ^= t + (t ^ (t >> 7)) * (t | 61u);
  return (double)(t ^ (t >> 14)) / 4294967296.0;
}

static int draw(Runtime *rt, const double *probs, int n) {
  Tape *tape = rt->search;
  int only = -1, open = 0;
  for (int k = 0; k < n; k++) if (probs[k] > 0) { open++; only = k; }
  if (open <= 1) return only > 0 ? only : 0;  // forced outcome: not a chance point
  if (tape->i < tape->n) return tape->choices[tape->i++];
  if (!tape->random) { memcpy(tape->probs, probs, sizeof(double) * n); tape->nprobs = n; longjmp(tape->branch, 1); }
  double u = mulberry(tape->random);
  int k = 0;
  for (; k < n - 1; k++) { if (u < probs[k]) break; u -= probs[k]; }
  tape->i++;
  return k;
}

static double jmax(double a, double b) { return (isnan(a) || isnan(b)) ? NAN : (a > b ? a : b); }
static double jmin(double a, double b) { return (isnan(a) || isnan(b)) ? NAN : (a < b ? a : b); }
static double cdf(double x, double fate) { x = jmin(1, jmax(0, x)); return (1 - fate) * x + fate * x * x; }
static int holds(double v, int cmp, double t) { return cmp == CMP_LT ? v < t : cmp == CMP_LE ? v <= t : cmp == CMP_GT ? v > t : v >= t; }

int chance_roll(Runtime *rt, int team, int cmp, double t) {
  double fate;
  if (rollContext_zero(rt, team, &fate)) return holds(0, cmp, t);
  double below = cdf(t, fate), p = (cmp == CMP_LT || cmp == CMP_LE) ? below : 1 - below;
  double probs[2] = {p, 1 - p};
  return draw(rt, probs, 2) == 0;
}
int chance_pickroll(Runtime *rt, int team, int n) {
  double fate;
  if (rollContext_zero(rt, team, &fate) || n <= 1) return 0;
  double probs[MAXPROBS];
  for (int k = 0; k < n; k++) probs[k] = cdf((double)(k + 1) / n, fate) - cdf((double)k / n, fate);
  return draw(rt, probs, n);
}
int chance_raw(Runtime *rt, int cmp, double t) {
  double below = jmin(1, jmax(0, t)), p = (cmp == CMP_LT || cmp == CMP_LE) ? below : 1 - below;
  double probs[2] = {p, 1 - p};
  return draw(rt, probs, 2) == 0;
}
int chance_pick(Runtime *rt, int n) {
  if (n <= 1) return 0;
  double probs[MAXPROBS];
  for (int k = 0; k < n; k++) probs[k] = 1.0 / n;
  return draw(rt, probs, n);
}

static Tape *current;  // the tape of the battle being played (engine_overflow has no runtime at hand)
_Noreturn void engine_overflow(Runtime *rt) { (void)rt; longjmp(current->branch, 2); }

static void on_turn(Runtime *rt, const Counters *counters) {
  Tape *tape = rt->search;
  state_pack(&rt->s, tape->turn_buffer);
  tape->turn_size = state_packed_size(&rt->s);
  tape->turn_counters = *counters;
  tape->have_turn = 1;
  tape->from = tape->i;
}

static Saved *keep_turn(Tape *tape) {  // the turn buffer as a saved battle for a branch's children
  Saved *s = malloc(sizeof(Saved) + tape->turn_size);
  s->refs = 0;
  s->counters = tape->turn_counters;
  memcpy(s->state, tape->turn_buffer, tape->turn_size);
  return s;
}

// One battle along a node's choices. Returns 0/1/2 (A, B, draw), -1 at a new chance point (tape->probs), -2 on overflow.
static int play(const char *start, Node *node, uint32_t *random, const Options *o, Runtime *rt, Tape *tape) {
  tape->choices = node->choices; tape->n = node->n; tape->i = 0; tape->random = random;
  tape->saved = node->saved; tape->from = 0; tape->have_turn = 0;
  if (node->saved) state_unpack(node->saved->state, &rt->s);
  else state_unpack(start, &rt->s);
  rt->search = tape;
  rt->depth = 0;
  rt->ops = 0;
  current = tape;
  int jumped = setjmp(tape->branch);
  if (jumped) return jumped == 1 ? -1 : -2;
  return simulate(rt, o->maxTurns, node->saved ? &node->saved->counters : NULL, random ? NULL : on_turn);
}

static void release(Saved *s) { if (s && --s->refs <= 0) free(s); }

Result solve(const State *start, const Options *o) {
  Result out = {0, 0, 0, 0, 0, 0, 0, 0};
  double win[3] = {0, 0, 0}, turns = 0;
  int cap_open = 64, cap_rare = 64, nopen = 1, nrare = 0;
  Node *open = malloc(sizeof(Node) * cap_open), *rare = malloc(sizeof(Node) * cap_rare);
  open[0] = (Node){1, NULL, 0, NULL};
  // the battle and its turn buffer hold MAXC cards (megabytes), so one pair is kept per process, not per solve
  static Runtime *rt;
  static char *turn_buffer;
  if (!rt) { rt = malloc(sizeof(Runtime)); turn_buffer = malloc(sizeof(State)); }
  Tape *tape = calloc(1, sizeof(Tape));
  tape->turn_buffer = turn_buffer;
  char *packed = malloc(state_packed_size(start));  // the start, packed like a saved turn
  state_pack(start, packed);
  int nodes = 0;
  while (nopen && nodes < o->nodeBudget) {
    int best = 0;
    for (int i = 1; i < nopen; i++) if (open[i].p > open[best].p) best = i;
    Node node = open[best];
    open[best] = open[nopen - 1];
    nopen--;
    nodes++;
    int r = play(packed, &node, NULL, o, rt, tape);
    if (r == -2) { out.overflow = 1; release(node.saved); free(node.choices); break; }
    if (r >= 0) { win[r] += node.p; turns += node.p * rt->s.turn; }
    else {
      Saved *saved = node.saved;
      int from = 0;
      if (tape->have_turn) { saved = keep_turn(tape); from = tape->from; }
      int base = node.n - from;
      for (int k = 0; k < tape->nprobs; k++) {
        double q = tape->probs[k];
        if (!(q > 0)) continue;
        Node child = {node.p * q, saved, base + 1, malloc(sizeof(int) * (base + 1))};
        memcpy(child.choices, node.choices + from, sizeof(int) * base);
        child.choices[base] = k;
        if (saved) saved->refs++;
        if (child.p < o->sampleBelow) {
          if (nrare == cap_rare) rare = realloc(rare, sizeof(Node) * (cap_rare *= 2));
          rare[nrare++] = child;
        } else {
          if (nopen == cap_open) open = realloc(open, sizeof(Node) * (cap_open *= 2));
          open[nopen++] = child;
        }
      }
      if (saved && saved->refs == 0) free(saved);
    }
    release(node.saved);
    free(node.choices);
  }
  int nleft = nopen + nrare;
  Node *left = malloc(sizeof(Node) * (nleft ? nleft : 1));
  memcpy(left, open, sizeof(Node) * nopen);
  memcpy(left + nopen, rare, sizeof(Node) * nrare);
  double mass = 0;
  for (int i = 0; i < nleft; i++) mass += left[i].p;
  out.nodes = nodes;
  if (out.overflow) {
  } else if (!nleft || !(mass > 0)) {
    out.a = win[0]; out.b = win[1]; out.draw = win[2]; out.turns = turns; out.exact = 1;
  } else {
    // pooled playouts: each picks an open branch in proportion to its probability, then plays on at random
    uint32_t random = o->seed;
    double *cum = malloc(sizeof(double) * nleft), s = 0;
    for (int i = 0; i < nleft; i++) { s += left[i].p; cum[i] = s; }
    double tally[3] = {0, 0, 0}, tally_turns = 0;
    int n = 0;
    while (n < o->rollouts) {
      for (int j = 0; j < 32 && n < o->rollouts; j++, n++) {
        double u = mulberry(&random) * mass;
        int lo = 0, hi = nleft - 1;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (cum[mid] < u) lo = mid + 1; else hi = mid; }
        int r = play(packed, &left[lo], &random, o, rt, tape);
        if (r < 0) { out.overflow = 1; break; }
        tally[r]++;
        tally_turns += rt->s.turn;
      }
      if (out.overflow) break;
      double pa = tally[0] / n;
      if (mass * sqrt(jmax(pa * (1 - pa), 1.0 / n) / n) <= o->rolloutError) break;
    }
    out.a = win[0] + mass * tally[0] / n; out.b = win[1] + mass * tally[1] / n; out.draw = win[2] + mass * tally[2] / n;
    out.turns = turns + mass * tally_turns / n;
    out.playouts = n;
    free(cum);
  }
  for (int i = 0; i < nleft; i++) { release(left[i].saved); free(left[i].choices); }
  free(left); free(open); free(rare); free(tape); free(packed);
  return out;
}

// ---- Python interface (card_engine/simulator/kernel.py) ----
const char *ce_version(void) { return "card-engine-c-4"; }
State *ce_state_new(void) { State *s = calloc(1, sizeof(State)); return s; }
void ce_state_free(State *s) { free(s); }
void ce_state_set(State *s, double turn, int moving) { s->turn = turn; s->moving = moving; }
int ce_add_card(State *s, int list, const char *id, int def, int team, double index, int border, const double *stats,
                int entered, int dead, int boss, int identity, int ability_override, int nbonus, const int *bonus,
                const double *status, int weakness, int blind) {
  if (s->ncard >= MAXC || (list < 2 ? s->nteam[list] : s->nfallen[list - 2]) >= MAXT) return -1;
  int c = s->ncard++;
  Card *k = &s->card[c];
  memset(k, 0, sizeof *k);
  k->id = fnv(FNV_START, id);
  k->def = def; k->team = team; k->index = index; k->border = border;
  k->power = stats[0]; k->hp = stats[1]; k->maxHp = stats[2]; k->damage = stats[3];
  k->entered = entered; k->dead = dead; k->boss = boss; k->identity = identity; k->abilityOverride = ability_override;
  k->nbonus = nbonus;
  for (int i = 0; i < nbonus && i < 6; i++) k->bonus[i] = bonus[i];
  k->stunned = status[0]; k->confused = status[1]; k->burn = status[2]; k->shield = status[3];
  k->weakness = weakness; k->blind = blind;
  if (list == 0 || list == 1) s->team[list][s->nteam[list]++] = c;
  else s->fallen[list - 2][s->nfallen[list - 2]++] = c;
  return c;
}
void ce_set_flag(State *s, int c, int flag) { s->card[c].flag[flag] = 1; }
void ce_set_counter(State *s, int c, int counter, double v) { s->card[c].counter[counter] = v; }
void ce_set_boosts(State *s, int team, const double *v, int composer_threshold_set, int skill_aura) {
  Boosts *b = &s->boosts[team];
  b->shielder = v[0]; b->fate = v[1]; b->flameWizard = v[2]; b->phantom = v[3]; b->berserker = v[4]; b->synthHuman = v[5];
  b->endTimes = v[6]; b->vampireMatron = v[7]; b->stormSpirit = v[8]; b->guardianAngel = v[9]; b->executioner = v[10];
  b->mirrorKnight = v[11]; b->finalTestament = v[12]; b->fossils = v[13]; b->composerCount = v[14];
  b->composerThreshold = v[15]; b->noAbilities = v[16];
  b->composerThresholdSet = composer_threshold_set; b->skillAura = skill_aura;
}
void ce_solve(const State *start, int nodeBudget, double sampleBelow, int rollouts, double rolloutError, int maxTurns,
              unsigned seed, double *out) {  // out[8]: a, b, draw, exact, nodes, playouts, overflow, turns
  Options o = {nodeBudget, rollouts, maxTurns, sampleBelow, rolloutError, seed};
  Result r = solve(start, &o);
  out[0] = r.a; out[1] = r.b; out[2] = r.draw; out[3] = r.exact; out[4] = r.nodes; out[5] = r.playouts; out[6] = r.overflow; out[7] = r.turns;
}
