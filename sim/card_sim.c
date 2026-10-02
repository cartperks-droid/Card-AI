#include "card_sim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double hp[2][CE_MAX_FIGHTERS];
    double max_hp[2][CE_MAX_FIGHTERS];
    double attack[2][CE_MAX_FIGHTERS];
    uint32_t used[2][CE_MAX_FIGHTERS];
    uint32_t survivals_used[2][CE_MAX_FIGHTERS];
    uint32_t completed_actions[2][CE_MAX_FIGHTERS];
    uint32_t order[2][CE_MAX_FIGHTERS]; /* team position -> fighter index; stats are per fighter */
    uint32_t entered[2][CE_MAX_FIGHTERS];
    uint32_t granted_dodges[2][CE_MAX_FIGHTERS];
    uint32_t stolen_from[2][CE_MAX_FIGHTERS];
    uint32_t turns[2][CE_MAX_FIGHTERS]; /* own turns started; only for turn-dependent cards */
    uint32_t hits_taken[2][CE_MAX_FIGHTERS]; /* only for alternating dodgers */
    uint32_t own_dodges_used[2][CE_MAX_FIGHTERS];
    uint32_t burn[2][CE_MAX_FIGHTERS];
    uint32_t bleed[2][CE_MAX_FIGHTERS];
    uint32_t frozen[2][CE_MAX_FIGHTERS];
    uint32_t slowed[2][CE_MAX_FIGHTERS];
    uint32_t frostbite[2][CE_MAX_FIGHTERS];
    uint32_t confused[2][CE_MAX_FIGHTERS];
    uint32_t dodge_charged[2][CE_MAX_FIGHTERS];
    uint32_t revived[2][CE_MAX_FIGHTERS];
    uint32_t guarded[2][CE_MAX_FIGHTERS];
    uint32_t faded[2][CE_MAX_FIGHTERS];
    uint32_t resting[2][CE_MAX_FIGHTERS];
    uint32_t debuffed[2][CE_MAX_FIGHTERS];
    uint32_t marked[2][CE_MAX_FIGHTERS];
    double incoming_scale[2][CE_MAX_FIGHTERS];
    double stored[2][CE_MAX_FIGHTERS];
    uint32_t poison[2][CE_MAX_FIGHTERS]; /* turns left */
    double poison_damage[2][CE_MAX_FIGHTERS]; /* HP lost per tick */
    double weakness[2][CE_MAX_FIGHTERS]; /* incoming damage multiplier from Ghoul */
    uint32_t triggered[2][CE_MAX_FIGHTERS]; /* one-time abilities: 1 low-HP heal, 2 zero-HP survival */
    uint32_t undying[2][CE_MAX_FIGHTERS]; /* untouchable until its doom ends (Zombie Dragon) */
    uint32_t doom[2][CE_MAX_FIGHTERS]; /* turns until death (Kira, Zombie Dragon) */
    uint32_t strike_timer[2][CE_MAX_FIGHTERS]; /* turns until this card's delayed strike (Onmyoji) */
    double strike_damage[2][CE_MAX_FIGHTERS];
    uint32_t blind[2][CE_MAX_FIGHTERS]; /* attacks always miss (Rudolph) */
    double block_pool[2][CE_MAX_FIGHTERS]; /* damage its blocks can still absorb (Steve) */
    uint32_t death_timer[2][CE_MAX_FIGHTERS]; /* Velociraptor: turn ends until this card dies */
    uint32_t fossils[2]; /* team fossil counters */
    int32_t ability[2][CE_MAX_FIGHTERS]; /* whose ability each card uses: 0 own, ABILITY_NONE removed, 1 + side * 16 + index */
    uint32_t suppressed[2]; /* Fuxi: turn ends left during which a side cannot use abilities */
    uint32_t size[2]; /* lineup positions (grows with summons) */
    int32_t ability2[2][CE_MAX_FIGHTERS]; /* a second ability (Pandora); 0 = none */
    uint32_t shields[2][CE_MAX_FIGHTERS]; /* whole attacks blocked */
    uint32_t timer[2][CE_MAX_FIGHTERS]; /* per-card counter (perish, doctor, composer, Tsukuyomi, Eonus, Dutchman) */
    double bonus_hp[2][CE_MAX_FIGHTERS]; /* Eonus: stolen stats that decay */
    double bonus_attack[2][CE_MAX_FIGHTERS];
    uint32_t evades[2][CE_MAX_FIGHTERS]; /* Eclipseborn Luminant dodges used */
    uint32_t extra[2][CE_MAX_FIGHTERS]; /* Gingerbread Man: extra actions next turn */
    uint32_t link[2][CE_MAX_FIGHTERS]; /* Fate Seamstress: linked card index + 1 */
    uint32_t curse[2][CE_MAX_FIGHTERS]; /* turns of 25% max HP loss (The Hanged Man, Naga) */
    uint32_t negated[2]; /* Marrowclaw: the side's blue aura is negated */
    uint32_t active[2];
    uint32_t side;
    uint32_t pair_actions;
    uint32_t hits_remaining;
    uint32_t counter_pending;
    uint32_t counter_from_entry;
    uint32_t entry_queue[2];
    uint32_t entry_count;
    uint32_t turn_actions_remaining;
} state;

typedef struct { state value; double probability; size_t insertion_order; } weighted_state;
typedef struct {
    weighted_state *rows;
    size_t *slots;
    size_t size;
    size_t capacity;
    size_t slot_count;
} frontier;

static int fail(char *error, size_t cap, int code, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
    return code;
}

const char *ce_version(void) { return "card-engine-kernel-0.55-provisional"; }

static unsigned hit_actor(const state *s) {
    if (s->counter_pending) return s->counter_pending - 1;
    return s->entry_count ? s->entry_queue[0] : s->side;
}

/* Only active cards can have queued entries. Clear unused cells for hashing. */
static void cancel_entry(state *s, unsigned side) {
    unsigned i, kept = 0;
    for (i = 0; i < s->entry_count; ++i)
        if (s->entry_queue[i] != side) s->entry_queue[kept++] = s->entry_queue[i];
    s->entry_count = kept;
    while (kept < 2) s->entry_queue[kept++] = 0;
}

/* Fighter index of the active card; callers check the side is not depleted. */
static unsigned front(const state *s, unsigned side) { return s->order[side][s->active[side]]; }

/* Statuses (user-observed): durations tick at the end of every turn, both
 * sides'; re-applying refreshes, different statuses stack. Tick sizes are fits. */
#define BURN_MAX_HP_FRACTION 0.10
#define BLEED_MAX_HP_FRACTION 0.16
/* Video 0290: frostbite rolls once per tick; on success 20% max HP and a lost
 * turn. User: confusion self-attacks at about 50%. */
#define FROSTBITE_CHANCE 0.5
#define FROSTBITE_MAX_HP_FRACTION 0.20
#define CONFUSION_CHANCE 0.5
#define MAX_CHANCE_ROLLS 32

/* Random rolls inside one step: sample mode draws lazily from rng; branch mode
 * replays forced outcomes and defaults later rolls to 0, recording them. */
typedef struct {
    const uint16_t *forced;
    int forced_count;
    int count;
    int overflow;
    double p[MAX_CHANCE_ROLLS];
    uint64_t *rng;
} chance_ctx;

static double uniform_random(uint64_t *rng);

static int roll(chance_ctx *c, double p) {
    int index;
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    if (c->count >= MAX_CHANCE_ROLLS) { c->overflow = 1; return 0; }
    index = c->count;
    c->p[c->count++] = p;
    if (c->rng) return uniform_random(c->rng) < p;
    return index < c->forced_count ? c->forced[index] != 0 : 0;
}

/* A uniform choice among n outcomes; branch mode enumerates all of them (stored as -n). */
static unsigned pick(chance_ctx *c, unsigned n) {
    int index;
    if (n <= 1) return 0;
    if (c->count >= MAX_CHANCE_ROLLS) { c->overflow = 1; return 0; }
    index = c->count;
    c->p[c->count++] = -(double)n;
    if (c->rng) {
        unsigned k = (unsigned)(uniform_random(c->rng) * n);
        return k < n ? k : n - 1;
    }
    return index < c->forced_count ? c->forced[index] : 0;
}

static void refresh(uint32_t *status, int32_t turns) {
    if (turns > 0 && *status < (uint32_t)turns) *status = (uint32_t)turns;
}

#define ABILITY_NONE (-1)
#define FOSSIL_DEATH_TIMER_CAP 2 /* Velociraptor: the fossil count is capped at 2 */
#define POISON_PERMANENT (1 << 20) /* user: Dilophosaurus's poison lasted the whole battle */

/* Re-poisoning refreshes the duration and keeps the stronger tick (extrapolated). */
static void poison_card(state *s, unsigned side, unsigned index, int32_t turns, double amount) {
    if (turns > 0 && amount > 0) {
        refresh(&s->poison[side][index], turns);
        s->poison_damage[side][index] = fmax(s->poison_damage[side][index], amount);
    }
}

static int counts_turns(const ce_fighter *f) {
    return f->alternate_rest || f->periodic_block_period || f->recharge_turns || f->first_turn_actions || f->actions_growth_per_turn || f->first_attack_multiplier != 1 ||
        f->action_end_stat_turns || f->periodic_stat_period || f->periodic_attack_period || f->turn_start_stat_growth > 0 ||
        f->turn_start_incoming_reduction > 0 || f->growth_turns || f->awaken_turn || f->shield_period;
}

static int periodic_turn(const ce_fighter *f, uint32_t turns_done) {
    return f->periodic_attack_period > 0 && (turns_done + 1) % (uint32_t)f->periodic_attack_period == 0;
}

/* Hits in a normal action: the periodic special's count, plus a follow-up hit. */
static unsigned action_hits(const ce_fighter *f, uint32_t turns_done) {
    unsigned hits = periodic_turn(f, turns_done) && f->periodic_attack_hits ? (unsigned)f->periodic_attack_hits : (unsigned)f->attacks_per_action;
    return hits + (f->followup_multiplier > 0 ? 1u : 0u);
}

/* Normal actions in the fighter's own turn `turn_index` (0-based). */
static unsigned allowance(const ce_fighter *f, int turn_index, int faded, int low) {
    int value;
    if (f->low_hp_extra_actions && low) {
        value = (int)allowance(f, turn_index, faded, 0) + f->low_hp_extra_actions;
        return (unsigned)(value > 16 ? 16 : value);
    }
    if (f->fade_extra_actions && !faded) {
        value = (int)allowance(f, turn_index, 1, 0) + f->fade_extra_actions;
        return (unsigned)(value > 16 ? 16 : value);
    }
    if (turn_index < 0) turn_index = 0;
    if (f->first_turn_actions && turn_index == 0) return (unsigned)f->first_turn_actions;
    value = f->actions_per_turn + f->actions_growth_per_turn * turn_index;
    return (unsigned)(value > 16 ? 16 : value);
}

static int has_entry_hit(const ce_fighter *f, unsigned position) {
    return f->entry_hit_multiplier > 0 && (!f->entry_hit_requires_fallen || position > 0);
}

static uint64_t next_random(uint64_t *rng) {
    uint64_t x = *rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng = x;
    return x * UINT64_C(2685821657736338717);
}

static double uniform_random(uint64_t *rng) {
    return (double)(next_random(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static int terminal(const ce_battle *battle, const state *s) {
    int a_dead = s->active[0] >= s->size[0];
    int b_dead = s->active[1] >= s->size[1];
    return a_dead && b_dead ? (battle->first_side == 0 ? 2 : 1) : b_dead ? 1 : a_dead ? 2 : 0;
}

static void add_terminal(ce_result *out, int outcome, double probability) {
    if (outcome == 1) out->p_a += probability;
    else if (outcome == 2) out->p_b += probability;
    else if (outcome == 3) out->tie += probability;
}

static double rounded(double value, int mode) {
    if (mode == CE_CEIL) value = ceil(value);
    else if (mode == CE_FLOOR) value = floor(value);
    else if (mode == CE_NEAREST_HALF_UP) value = floor(value + 0.5);
    return value == 0 ? 0 : value;
}

static void scale(state *s, unsigned side, unsigned slot, double hp_factor, double attack_factor, int r) {
    s->hp[side][slot] = rounded(s->hp[side][slot] * hp_factor, r);
    s->max_hp[side][slot] = rounded(s->max_hp[side][slot] * hp_factor, r);
    s->attack[side][slot] = rounded(s->attack[side][slot] * attack_factor, r);
}

static void transfer(state *s, unsigned ss, unsigned si, unsigned ts, unsigned ti, double fraction, int r) {
    double take_hp = rounded(s->hp[ss][si] * fraction, r), take_attack = rounded(s->attack[ss][si] * fraction, r);
    s->hp[ss][si] = rounded(s->hp[ss][si] - take_hp, r);
    s->max_hp[ss][si] = rounded(s->max_hp[ss][si] - take_hp, r);
    s->attack[ss][si] = rounded(s->attack[ss][si] - take_attack, r);
    s->hp[ts][ti] = rounded(s->hp[ts][ti] + take_hp, r);
    s->max_hp[ts][ti] = rounded(s->max_hp[ts][ti] + take_hp, r);
    s->attack[ts][ti] = rounded(s->attack[ts][ti] + take_attack, r);
}

static double field_multiplier(const ce_battle *battle, const state *s) {
    double product = 1;
    unsigned side, position;
    for (side = 0; side < 2; ++side)
        for (position = s->active[side]; position < s->size[side]; ++position) {
            unsigned index = s->order[side][position];
            if (s->hp[side][index] > 0) product *= battle->fighters[side][index].field_damage_multiplier;
        }
    return product;
}

static void scale_allies(const ce_battle *battle, state *s, unsigned side, double factor, int r) {
    (void)battle;
    unsigned position;
    for (position = s->active[side] + 1; position < s->size[side]; ++position) {
        unsigned index = s->order[side][position];
        if (s->hp[side][index] > 0) scale(s, side, index, factor, factor, r);
    }
}

/* Enemy-entry theft against the card now at the front of `side`. User-observed:
 * Chronus steals when Piccolo swaps in, never from the same card twice. A
 * returning card never stolen from is assumed to be stolen from. */
static int steal(const ce_battle *battle, const ce_options *options, state *s, unsigned side) {
    unsigned other = 1 - side, index, other_index;
    double fraction, hp, attack;
    if (s->active[side] >= s->size[side] || s->active[other] >= s->size[other]) return 0;
    index = front(s, side);
    other_index = front(s, other);
    fraction = battle->fighters[other][other_index].enemy_entry_steal_fraction;
    if (fraction <= 0 || s->stolen_from[side][index] || s->hp[side][index] <= 0) return 0; /* nothing to steal from 0 HP */
    s->stolen_from[side][index] = 1;
    hp = rounded(s->hp[side][index] * fraction, options->stat_rounding);
    attack = rounded(s->attack[side][index] * fraction, options->stat_rounding);
    s->hp[side][index] = rounded(s->hp[side][index] - hp, options->stat_rounding);
    s->max_hp[side][index] = rounded(s->max_hp[side][index] - hp, options->stat_rounding);
    s->attack[side][index] = rounded(s->attack[side][index] - attack, options->stat_rounding);
    s->hp[other][other_index] = rounded(s->hp[other][other_index] + hp, options->stat_rounding);
    s->max_hp[other][other_index] = rounded(s->max_hp[other][other_index] + hp, options->stat_rounding);
    s->attack[other][other_index] = rounded(s->attack[other][other_index] + attack, options->stat_rounding);
    return s->hp[side][index] <= 0 || s->attack[side][index] < 0 || !isfinite(s->hp[side][index]) || !isfinite(s->attack[side][index]) ||
        !isfinite(s->hp[other][other_index]) || !isfinite(s->attack[other][other_index]) ||
        !isfinite(s->max_hp[side][index]) || !isfinite(s->max_hp[other][other_index]);
}

static int32_t ability_code(const state *s, unsigned side, unsigned index) {
    return s->ability[side][index] == 0 ? (int32_t)(1 + side * 16 + index) : s->ability[side][index];
}

#define POOL_CODE 33

static const ce_fighter *code_source(const ce_battle *base, int32_t code, unsigned side, unsigned index) {
    if (code == ABILITY_NONE) return NULL;
    if (code == 0) return &base->fighters[side][index];
    if (code >= POOL_CODE) return (uint32_t)(code - POOL_CODE) < base->pool_count ? &base->pool[code - POOL_CODE] : NULL;
    code -= 1;
    return &base->fighters[code / 16][code % 16];
}

static const ce_fighter *ability_source(const ce_battle *base, const state *s, unsigned side, unsigned index) {
    return code_source(base, s->ability[side][index], side, index);
}

/* Two abilities at once (Pandora, the Great Nirvana Sword): the second's non-default fields win.
 * ce_fighter is all doubles, then all int32 (checked in validate). */
#define FIGHTER_DOUBLES (offsetof(ce_fighter, block_mode) / sizeof(double))
#define FIGHTER_INTS ((sizeof(ce_fighter) - offsetof(ce_fighter, block_mode)) / sizeof(int32_t))
static void merge(ce_fighter *into, const ce_fighter *second, const ce_fighter *defaults) {
    double *d = (double *)(void *)into;
    const double *sd = (const double *)(const void *)second, *dd = (const double *)(const void *)defaults;
    int32_t *i = (int32_t *)(void *)((char *)into + offsetof(ce_fighter, block_mode));
    const int32_t *si = (const int32_t *)(const void *)((const char *)second + offsetof(ce_fighter, block_mode));
    const int32_t *di = (const int32_t *)(const void *)((const char *)defaults + offsetof(ce_fighter, block_mode));
    size_t k;
    for (k = 0; k < FIGHTER_DOUBLES; ++k) if (sd[k] != dd[k]) d[k] = sd[k];
    for (k = 0; k < FIGHTER_INTS; ++k) if (si[k] != di[k]) i[k] = si[k];
}

/* The Curse's domain (user Part A): enemy dodges, blocks, damage reduction and survivals stop working
 * (not Parallax's reflection). */
static void strip_defenses(ce_fighter *f) {
    f->dodge_probability = 0; f->block_mode = 0; f->damage_reduction_max_hp = 0; f->damage_cap_max_hp = 0;
    f->dodge_below_max_hp = 0; f->nullify_below_attack = 0; f->lethal_survivals = 0; f->invincible = 0;
    f->own_lethal_dodges = 0; f->alternate_dodge = 0; f->chance_survival_probability = 0; f->guardian_chance = 0;
    f->hit_chance_squared = 0; f->status_dodge_max = 0; f->zero_hp_survival_turns = 0; f->aura_incoming_multiplier = 1;
    f->lost_hp_incoming_bonus = 0;
    f->incoming_multiplier = fmax(1, f->incoming_multiplier);
    f->low_hp_incoming_multiplier = fmax(1, f->low_hp_incoming_multiplier);
    f->disadvantage_incoming_multiplier = fmax(1, f->disadvantage_incoming_multiplier);
}

/* The battle as each card currently acts: abilities removed, stolen, copied or cancelled
 * (Samurai, Fuxi, Black Cat, Sable); identity fields always stay the card's own. */
/* End Times on `side`: the chance an enemy ability fails (none while Marrowclaw negates the support). */
static double fail_chance(const ce_battle *base, const state *s, unsigned side) {
    double p = 0;
    unsigned i;
    if (s->negated[side]) return 0;
    for (i = 0; i < base->counts[side]; ++i) p = fmax(p, base->fighters[side][i].ability_fail_chance);
    return p;
}

/* End Times: each step, the front card facing it may have its ability fail for that step. */
static const unsigned *end_times(const ce_battle *base, const state *s, chance_ctx *ch, unsigned *failed) {
    unsigned side;
    for (side = 0; side < 2; ++side) {
        double p = fail_chance(base, s, 1 - side);
        if (p > 0 && s->active[side] < s->size[side] && s->hp[side][s->order[side][s->active[side]]] > 0)
            failed[side] = (unsigned)roll(ch, p);
    }
    return failed;
}

static const ce_battle *ability_view(const ce_battle *base, const state *s, ce_battle *storage, const unsigned *failed) {
    unsigned side, i, dynamic = 0;
    int front_index[2], samurai[2], sable[2];
    for (side = 0; side < 2; ++side) {
        if (s->suppressed[side] || (failed != NULL && failed[side])) dynamic = 1;
        for (i = 0; i < base->counts[side]; ++i) {
            const ce_fighter *f = &base->fighters[side][i];
            if (s->ability[side][i] || s->ability2[side][i] || f->cancels_all_abilities || f->disable_enemy_class_mask ||
                f->copies_enemy_in_play || f->domain) dynamic = 1;
        }
    }
    if (!dynamic) return base;
    for (side = 0; side < 2; ++side) {
        unsigned q = s->active[side];
        front_index[side] = q < s->size[side] && s->hp[side][s->order[side][q]] > 0 ? (int)s->order[side][q] : -1;
    }
    for (side = 0; side < 2; ++side) {
        const ce_fighter *src = front_index[side] >= 0 ? ability_source(base, s, side, (unsigned)front_index[side]) : NULL;
        samurai[side] = src != NULL && src->cancels_all_abilities;
        sable[side] = src != NULL && src->copies_enemy_in_play; /* Sable (user): takes and disables the enemy's ability */
    }
    /* Only what views are read for: the header and the fighters in use (blanks and defaults are read from base). */
    storage->first_side = base->first_side;
    storage->pool_count = base->pool_count;
    storage->pool = base->pool;
    for (side = 0; side < 2; ++side) {
        storage->counts[side] = base->counts[side];
        storage->lineup[side] = base->lineup[side];
        memcpy(storage->fighters[side], base->fighters[side], base->counts[side] * sizeof(ce_fighter));
    }
    for (side = 0; side < 2; ++side) {
        unsigned e = 1 - side;
        const ce_fighter *enemy_front = front_index[e] >= 0 ? ability_source(base, s, e, (unsigned)front_index[e]) : NULL;
        for (i = 0; i < base->counts[side]; ++i) {
            const ce_fighter *own = &base->fighters[side][i], *source = ability_source(base, s, side, i);
            int cancelled = s->suppressed[side] > 0 || (failed != NULL && failed[side] && front_index[side] == (int)i), t;
            for (t = 0; t < 2; ++t)
                if (samurai[t] && !((unsigned)t == side && front_index[t] == (int)i)) cancelled = 1;
            if (enemy_front != NULL && (enemy_front->disable_enemy_class_mask & own->class_mask)) cancelled = 1; /* Black Cat */
            if (sable[e] && front_index[side] == (int)i) cancelled = 1; /* Sable */
            if (source != NULL && source->copies_enemy_in_play && enemy_front != NULL) source = enemy_front; /* Sable */
            if (cancelled || source == NULL) storage->fighters[side][i] = base->blanks[side][i];
            else {
                const ce_fighter *extra = s->ability2[side][i] ? code_source(base, s->ability2[side][i], side, i) : NULL;
                if (source != own || extra != NULL) {
                    ce_fighter f = *source;
                    if (extra != NULL) merge(&f, extra, &base->defaults);
                    f.hp = own->hp; f.attack = own->attack; f.borderless_hp = own->borderless_hp; f.borderless_attack = own->borderless_attack;
                    f.card_rarity = own->card_rarity; f.border_rarity = own->border_rarity; f.pack_index = own->pack_index;
                    f.class_mask = own->class_mask; f.dinosaur = own->dinosaur; f.card_id = own->card_id; f.spare = own->spare;
                    storage->fighters[side][i] = f;
                }
            }
            if (enemy_front != NULL && enemy_front->domain) strip_defenses(&storage->fighters[side][i]);
        }
    }
    return storage;
}


#define SAM_BOOST_MULTIPLIER 5.0 /* Uncle Sam (user Part A): the next card deals 5x and dies after attacking */
#define EONUS_TURNS 3
#define LUMINANT_STEP 0.1
#define LUMINANT_FLOOR 0.2
#define LUMINANT_MAX 2u
#define LUMINANT_GAIN 0.1
#define LUMINANT_CAP 3.0
#define BUDDHA_FRACTION 0.5
#define POOL_FORM 4
#define CURSE_MAX_HP_FRACTION 0.25 /* The Hanged Man, Naga (user) */
#define CURSE_PERMANENT (1u << 20)
#define CONFUSION_UNTIL_SELF_HIT (1u << 20) /* Kuchisake-onna */
#define CONFUSION_ETERNAL (1u << 21) /* Cthulu */
#define SANTA_DECAY 0.9
#define SPLIT_HP 0.5
#define SPLIT_ATTACK 0.6
#define PARTY_REVIVED (1u << 10)
#define SPENT (1u << 11)
#define SAM_BOOST (1u << 12)
#define SANTA (1u << 13)
#define DUTCH_ALLY (1u << 14)
#define DUTCH_AWAY (1u << 15)
#define TRANSFORMED (1u << 16)
#define AWAKENED (1u << 17)
#define SHIFTED (1u << 18) /* Glamour has changed this turn (its new form's entry attack comes first) */

static int enter(const ce_battle *battle, const ce_options *options, state *s, unsigned side, chance_ctx *ch);
static int enter_card(const ce_battle *battle, const ce_options *options, state *s, unsigned side, chance_ctx *ch, int body_only);
static int heal(state *s, unsigned side, unsigned slot, double raw_amount, int rounding);

/* Damage from an ability rather than an attack. Marrowclaw cannot be killed this way; Juggernoid reflects a share. */
static void ability_hit(const ce_battle *battle, state *s, unsigned side, unsigned index, double amount, int rounding) {
    const ce_fighter *card = &battle->fighters[side][index];
    double before;
    if (amount <= 0) return;
    before = s->hp[side][index];
    s->hp[side][index] = before - amount;
    if (card->ability_death_immune && before > 0 && s->hp[side][index] < 1) s->hp[side][index] = fmin(before, 1);
    if (card->ability_reflect_fraction > 0) {
        unsigned enemy = 1 - side;
        if (s->active[enemy] < s->size[enemy]) {
            unsigned other = front(s, enemy);
            if (s->hp[enemy][other] > 0)
                s->hp[enemy][other] -= rounded(fmin(amount * card->ability_reflect_fraction,
                                                    card->ability_reflect_cap * s->attack[side][index]), rounding);
        }
    }
}

static int has_status(const state *s, unsigned side, unsigned i) {
    return s->burn[side][i] || s->bleed[side][i] || s->frozen[side][i] || s->slowed[side][i] || s->frostbite[side][i] ||
        s->confused[side][i] || s->poison[side][i] || s->weakness[side][i] > 1 || s->doom[side][i] || s->blind[side][i] ||
        s->death_timer[side][i] || s->curse[side][i];
}

/* Serket: while it is alive in the party, its whole side is immune to statuses (cleared as they land). */
static void cleanse(const ce_battle *battle, state *s) {
    unsigned side, p;
    for (side = 0; side < 2; ++side) {
        int immune = 0;
        for (p = 0; p < s->size[side]; ++p) {
            unsigned i = s->order[side][p];
            if (s->hp[side][i] > 0 && battle->fighters[side][i].team_status_immunity) immune = 1;
        }
        if (!immune) continue;
        for (p = 0; p < s->size[side]; ++p) {
            unsigned i = s->order[side][p];
            s->burn[side][i] = s->bleed[side][i] = s->frozen[side][i] = s->frostbite[side][i] = s->confused[side][i] = 0;
            s->poison[side][i] = s->doom[side][i] = s->blind[side][i] = s->death_timer[side][i] = s->curse[side][i] = 0;
            s->poison_damage[side][i] = 0;
            s->weakness[side][i] = 1;
        }
    }
}

/* Positions of living waiting cards; returns the count. */
static unsigned waiting(const state *s, unsigned side, unsigned *positions) {
    unsigned p, n = 0;
    for (p = s->active[side] + 1; p < s->size[side]; ++p)
        if (s->hp[side][s->order[side][p]] > 0) positions[n++] = p;
    return n;
}

static int in_order(const state *s, unsigned side, unsigned index) {
    unsigned p;
    for (p = 0; p < s->size[side]; ++p) if (s->order[side][p] == index) return 1;
    return 0;
}

static unsigned position_of(const state *s, unsigned side, unsigned index) {
    unsigned p;
    for (p = 0; p < s->size[side]; ++p) if (s->order[side][p] == index) return p;
    return s->size[side];
}

static void order_insert(state *s, unsigned side, unsigned position, unsigned index) {
    unsigned p;
    for (p = s->size[side]; p > position; --p) s->order[side][p] = s->order[side][p - 1];
    s->order[side][position] = index;
    s->size[side]++;
}

static void order_delete(state *s, unsigned side, unsigned position) {
    unsigned p;
    for (p = position; p + 1 < s->size[side]; ++p) s->order[side][p] = s->order[side][p + 1];
    s->size[side]--;
    s->order[side][s->size[side]] = 0;
}

static int eligible(const ce_fighter *f, int32_t exclude) { return !(f->pool_flags & (exclude | POOL_FORM)); }

/* A uniformly random ability from the pool (never a form; never `skip` or `other`); -1 if none. */
static int32_t pool_pick(const ce_battle *b, chance_ctx *ch, int32_t exclude, int skip, int other) {
    unsigned i, count = 0, k;
    for (i = 0; i < b->pool_count; ++i) if (eligible(&b->pool[i], exclude) && (int)i != skip && (int)i != other) ++count;
    if (!count) return -1;
    k = pick(ch, count);
    for (i = 0; i < b->pool_count; ++i)
        if (eligible(&b->pool[i], exclude) && (int)i != skip && (int)i != other && !k--) return POOL_CODE + (int32_t)i;
    return -1;
}

/* Put a free spare slot into the lineup (back, or at `position`); -1 when none is left. */
static int summon(const ce_battle *b, state *s, unsigned side, double hp, double attack, int32_t code, int position) {
    unsigned i;
    for (i = 0; i < b->counts[side]; ++i)
        if (b->fighters[side][i].spare && !in_order(s, side, i) && !(s->triggered[side][i] & SPENT)) break;
    if (i >= b->counts[side]) return -1;
    s->triggered[side][i] |= SPENT;
    s->hp[side][i] = s->max_hp[side][i] = hp;
    s->attack[side][i] = attack;
    s->ability[side][i] = code;
    s->entered[side][i] = 0;
    order_insert(s, side, position < 0 ? s->size[side] : (unsigned)position, i);
    return (int)i;
}

/* Move a card to the back of its lineup (revivals, Buddha). */
static void relocate(state *s, unsigned side, unsigned index) {
    unsigned position = position_of(s, side, index);
    order_delete(s, side, position);
    if (position < s->active[side]) s->active[side]--;
    s->order[side][s->size[side]++] = index;
}

static void restore(state *s, unsigned side, unsigned i, double hp) {
    s->hp[side][i] = hp;
    s->burn[side][i] = s->bleed[side][i] = s->frozen[side][i] = s->slowed[side][i] = s->frostbite[side][i] = 0;
    s->confused[side][i] = s->poison[side][i] = s->undying[side][i] = s->doom[side][i] = s->blind[side][i] = 0;
    s->death_timer[side][i] = s->curse[side][i] = 0;
    s->poison_damage[side][i] = 0;
    s->weakness[side][i] = 1;
}

/* Index of an Eclipseborn Luminant alive in the party (not the defending card itself); -1 if none. */
static int luminant(const ce_battle *battle, const state *s, unsigned side, unsigned index) {
    unsigned p;
    for (p = 0; p < s->size[side]; ++p) {
        unsigned i = s->order[side][p];
        if (i != index && s->hp[side][i] > 0 && battle->fighters[side][i].luminant_evasion > 0) return (int)i;
    }
    return -1;
}

/* Buddha: revive the first dead ally at 50% HP (to the back) or heal the weakest living card by 50% of its
 * max HP; then move to the back if every living card is at full HP. Returns 1 when it moved. */
static int buddha(state *s, unsigned side, unsigned index, const ce_options *options) {
    /* User: revive the dead one per turn in setup order (staying in front); with none dead, heal the weakest
     * other card (never itself) and move to the back once a heal fully heals it (or every other card is full). */
    unsigned p, i, dead = CE_MAX_FIGHTERS, living = 0, full = 1, healed_full = 0;
    for (i = 0; i < CE_MAX_FIGHTERS && dead == CE_MAX_FIGHTERS; ++i)
        for (p = 0; p < s->size[side]; ++p)
            if (s->order[side][p] == i && i != index && s->hp[side][i] <= 0) { dead = i; break; }
    if (dead < CE_MAX_FIGHTERS) {
        restore(s, side, dead, s->max_hp[side][dead]); /* user: Buddha revives at the back with full HP */
        relocate(s, side, dead);
        return 0;
    } else {
        int weakest = -1;
        for (p = s->active[side]; p < s->size[side]; ++p) {
            i = s->order[side][p];
            if (i != index && s->hp[side][i] > 0 && (weakest < 0 ||
                s->hp[side][i] / s->max_hp[side][i] < s->hp[side][weakest] / s->max_hp[side][weakest])) weakest = (int)i;
        }
        if (weakest >= 0 && s->hp[side][weakest] < s->max_hp[side][weakest]) {
            if (heal(s, side, (unsigned)weakest, s->max_hp[side][weakest] * BUDDHA_FRACTION, options->rounding)) return -1;
            healed_full = s->hp[side][weakest] >= s->max_hp[side][weakest];
        }
    }
    for (p = s->active[side]; p < s->size[side]; ++p) {
        i = s->order[side][p];
        if (i != index && s->hp[side][i] > 0) {
            ++living;
            if (s->hp[side][i] < s->max_hp[side][i]) full = 0;
        }
    }
    if (living && (healed_full || full)) { relocate(s, side, index); return 1; }
    return 0;
}

/* Batch-12 death triggers, in the reference's order: The Broken One's split, Naga's curse, Anubis / Bloody Mary,
 * Control Freak, Anubis & Hades, the Flying Dutchman's return, Time Lord Stryx. */
static void death_effects(const ce_battle *battle, const ce_options *options, state *s, unsigned side, unsigned dead,
                          chance_ctx *ch, int in_deck) {
    const ce_fighter *f = &battle->fighters[side][dead];
    unsigned enemy = 1 - side, p, n, snapshot[CE_MAX_FIGHTERS];
    int r = options->stat_rounding;
    if (f->split_on_death) {
        unsigned position = in_deck ? position_of(s, side, dead) + 1 : s->active[side], k;
        for (k = 0; k < 2; ++k)
            if (summon(battle, s, side, rounded(s->max_hp[side][dead] * SPLIT_HP, r), rounded(s->attack[side][dead] * SPLIT_ATTACK, r), 0,
                       (int)(position + k)) < 0) break;
    }
    if (f->death_curse_turns && s->active[enemy] < s->size[enemy]) {
        unsigned other = front(s, enemy);
        if (s->hp[enemy][other] > 0 && s->curse[enemy][other] < (uint32_t)f->death_curse_turns) s->curse[enemy][other] = (uint32_t)f->death_curse_turns;
    }
    n = s->size[side];
    for (p = 0; p < n; ++p) snapshot[p] = s->order[side][p];
    for (p = 0; p < n; ++p) {
        unsigned i = snapshot[p];
        const ce_fighter *card = &battle->fighters[side][i];
        if (i != dead && card->revive_on_ally_death_hp > 0 && s->hp[side][i] <= 0 && roll(ch, card->revive_on_ally_death_chance)) {
            restore(s, side, i, fmax(1, rounded(s->max_hp[side][i] * card->revive_on_ally_death_hp, r)));
            relocate(s, side, i);
        }
    }
    {
        int recruit = 0;
        for (p = 0; p < s->size[enemy]; ++p) {
            unsigned i = s->order[enemy][p];
            if (s->hp[enemy][i] > 0 && battle->fighters[enemy][i].recruit_defeated) recruit = 1;
        }
        if (recruit) summon(battle, s, enemy, f->hp, f->attack, ability_code(s, side, dead), -1);
    }
    n = s->size[enemy];
    for (p = 0; p < n; ++p) snapshot[p] = s->order[enemy][p];
    for (p = 0; p < n; ++p) {
        unsigned i = snapshot[p], q;
        int target = -1;
        double fraction = battle->fighters[enemy][i].enemy_death_revive_fraction;
        if (fraction <= 0) continue;
        if (s->hp[enemy][i] <= 0) target = (int)i;
        else for (q = 0; q < s->size[enemy]; ++q) if (s->hp[enemy][s->order[enemy][q]] <= 0) { target = (int)s->order[enemy][q]; break; }
        if (target < 0) continue;
        s->max_hp[enemy][target] = rounded(s->max_hp[enemy][i] * fraction, r);
        s->attack[enemy][target] = rounded(s->attack[enemy][i] * fraction, r);
        restore(s, enemy, (unsigned)target, s->max_hp[enemy][target]);
        relocate(s, enemy, (unsigned)target);
    }
    if ((s->triggered[side][dead] & DUTCH_ALLY) && !in_deck) {
        s->triggered[side][dead] &= ~DUTCH_ALLY;
        for (p = 0; p < s->size[side]; ++p) {
            unsigned i = s->order[side][p];
            if ((s->triggered[side][i] & DUTCH_AWAY) && s->hp[side][i] > 0) {
                s->triggered[side][i] &= ~DUTCH_AWAY;
                order_delete(s, side, p);
                order_insert(s, side, s->active[side], i);
                break;
            }
        }
    }
    if (f->revive_party_on_death && !(s->triggered[side][dead] & PARTY_REVIVED)) {
        unsigned lineup = battle->lineup[side], rebuilt[CE_MAX_FIGHTERS], count = 0, i;
        s->triggered[side][dead] |= PARTY_REVIVED;
        for (i = 0; i < lineup; ++i)
            if (in_order(s, side, i)) {
                if (s->hp[side][i] <= 0) restore(s, side, i, s->max_hp[side][i]);
                rebuilt[count++] = i;
            }
        for (p = 0; p < s->size[side]; ++p) if (s->order[side][p] >= lineup) rebuilt[count++] = s->order[side][p];
        for (p = 0; p < count; ++p) s->order[side][p] = rebuilt[p];
        s->active[side] = 0;
    }
}


/* Batch-12 entry effects, in the reference's order. */
static int entry_effects(const ce_battle *battle, const ce_options *options, state *s, unsigned side, unsigned index, chance_ctx *ch) {
    const ce_fighter *f = &battle->fighters[side][index];
    unsigned enemy = 1 - side, p;
    int has_enemy = s->active[enemy] < s->size[enemy];
    unsigned enemy_index = has_enemy ? front(s, enemy) : 0;
    int alive = has_enemy && s->hp[enemy][enemy_index] > 0;
    int r = options->stat_rounding;
    const ce_fighter *sources[3];
    unsigned source_count = 1, k;
    if (f->aura_negate_chance > 0 && roll(ch, f->aura_negate_chance)) s->negated[enemy] = 1; /* Marrowclaw */
    if (f->persistent_confusion && alive && s->confused[enemy][enemy_index] < CONFUSION_UNTIL_SELF_HIT)
        s->confused[enemy][enemy_index] = CONFUSION_UNTIL_SELF_HIT; /* Kuchisake-onna */
    if (f->entry_reset_enemy_stats && alive) { /* Poison Witch */
        const ce_fighter *base = &battle->fighters[enemy][enemy_index];
        s->hp[enemy][enemy_index] = rounded(base->hp * s->hp[enemy][enemy_index] / s->max_hp[enemy][enemy_index], r);
        s->max_hp[enemy][enemy_index] = base->hp;
        s->attack[enemy][enemy_index] = base->attack;
    }
    if (f->random_abilities) { /* Pandora */
        int32_t first = pool_pick(battle, ch, f->pool_exclude_mask, f->pool_self - 1, -1);
        if (first >= 0) {
            s->ability[side][index] = first;
            if (f->random_abilities > 1) {
                int32_t second = pool_pick(battle, ch, f->pool_exclude_mask, f->pool_self - 1, first - POOL_CODE);
                if (second >= 0) s->ability2[side][index] = second;
            }
        }
    }
    if (f->entry_ability_swap_random && alive) { /* Loki */
        int32_t taken = ability_code(s, enemy, enemy_index);
        int32_t given = pool_pick(battle, ch, f->pool_exclude_mask, f->pool_self - 1, -1);
        if (given >= 0) s->ability[enemy][enemy_index] = given;
        s->ability[side][index] = taken;
    }
    sources[0] = f;
    if (f->form_count) { /* Astraeus's sign, The Awakened One's weapon (the last option: two weapons) */
        unsigned choice = pick(ch, (unsigned)f->form_count);
        s->ability2[side][index] = 0;
        if (f->form_pair_last && choice == (unsigned)f->form_count - 1) {
            unsigned a = pick(ch, (unsigned)f->form_count - 1), b = pick(ch, (unsigned)f->form_count - 2);
            b += b >= a;
            s->ability[side][index] = POOL_CODE + f->form_first - 1 + (int32_t)a;
            s->ability2[side][index] = POOL_CODE + f->form_first - 1 + (int32_t)b;
        } else s->ability[side][index] = POOL_CODE + f->form_first - 1 + (int32_t)choice;
        if (s->ability[side][index] >= POOL_CODE) sources[source_count++] = &battle->pool[s->ability[side][index] - POOL_CODE];
        if (s->ability2[side][index] >= POOL_CODE) sources[source_count++] = &battle->pool[s->ability2[side][index] - POOL_CODE];
    }
    for (k = 0; k < source_count; ++k) {
        if (sources[k]->entry_same_card_stat_bonus > 0) { /* Gemini */
            unsigned count = 0;
            double factor;
            for (p = 0; p < s->size[side]; ++p) if (battle->fighters[side][s->order[side][p]].card_id == f->card_id) ++count;
            factor = 1 + sources[k]->entry_same_card_stat_bonus * count;
            scale(s, side, index, factor, factor, r);
        }
        if (sources[k]->entry_shields) s->shields[side][index] += (uint32_t)sources[k]->entry_shields; /* Virgo */
    }
    if (f->sack) { /* The Sack (user): independent rolls */
        unsigned bonus_attack = pick(ch, 50) + 1, bonus_hp = pick(ch, 50) + 1, reduction = pick(ch, 30) + 1;
        scale(s, side, index, 1 + bonus_hp / 100.0, 1 + bonus_attack / 100.0, r);
        s->incoming_scale[side][index] *= 1 - reduction / 100.0;
    }
    if (f->entry_absorb_behind > 0) { /* Cosmic Pop Star */
        for (p = s->active[side] + 1; p < s->size[side]; ++p) {
            unsigned victim = s->order[side][p];
            if (s->hp[side][victim] <= 0) continue;
            s->hp[side][index] = rounded(s->hp[side][index] + s->hp[side][victim] * f->entry_absorb_behind, r);
            s->max_hp[side][index] = rounded(s->max_hp[side][index] + s->max_hp[side][victim] * f->entry_absorb_behind, r);
            s->attack[side][index] = rounded(s->attack[side][index] + s->attack[side][victim] * f->entry_absorb_behind, r);
            order_delete(s, side, p);
            break;
        }
    }
    for (k = 0; k < (unsigned)f->summon_copies; ++k) /* Fresno */
        summon(battle, s, side, rounded(s->max_hp[side][index] * f->summon_copy_fraction, r),
               rounded(s->attack[side][index] * f->summon_copy_fraction, r), 0, -1);
    for (k = 0; k < (unsigned)f->summon_random; ++k) { /* Nuwa */
        int32_t code = pool_pick(battle, ch, f->pool_exclude_mask, -1, -1);
        if (code >= 0) summon(battle, s, side, s->max_hp[side][index], s->attack[side][index], code, -1);
    }
    if (f->entry_global_steal > 0) { /* Eonus */
        unsigned s2;
        for (s2 = 0; s2 < 2; ++s2)
            for (p = 0; p < s->size[s2]; ++p) {
                unsigned i = s->order[s2][p];
                double take_hp, take_attack;
                if ((s2 == side && i == index) || s->hp[s2][i] <= 0) continue;
                take_hp = rounded(s->hp[s2][i] * f->entry_global_steal, r);
                take_attack = rounded(s->attack[s2][i] * f->entry_global_steal, r);
                s->hp[s2][i] -= take_hp;
                s->max_hp[s2][i] -= take_hp;
                s->attack[s2][i] -= take_attack;
                s->hp[side][index] += take_hp;
                s->max_hp[side][index] += take_hp;
                s->attack[side][index] += take_attack;
                s->bonus_hp[side][index] += take_hp;
                s->bonus_attack[side][index] += take_attack;
            }
        s->timer[side][index] = EONUS_TURNS;
    }
    if (f->entry_ally_stat_multiplier != 1) /* Santa Claws */
        for (p = 0; p < s->size[side]; ++p) {
            unsigned i = s->order[side][p];
            if (i != index && s->hp[side][i] > 0) {
                scale(s, side, i, f->entry_ally_stat_multiplier, f->entry_ally_stat_multiplier, r);
                s->triggered[side][i] |= SANTA;
            }
        }
    if (f->link_first_two) { /* Fate Seamstress */
        unsigned linked[2], n = 0;
        for (p = s->active[enemy]; p < s->size[enemy] && n < 2; ++p)
            if (s->hp[enemy][s->order[enemy][p]] > 0) linked[n++] = s->order[enemy][p];
        if (n == 2) { s->link[enemy][linked[0]] = linked[1] + 1; s->link[enemy][linked[1]] = linked[0] + 1; }
    }
    if (f->perish_turns) s->timer[side][index] = (uint32_t)f->perish_turns + 1; /* Sleep Paralysis: both die at its 4th turn start */
    if (f->entry_shuffle_enemy && alive) { /* Jersey Devil */
        unsigned positions[CE_MAX_FIGHTERS], cards[CE_MAX_FIGHTERS], n = 0, i;
        for (p = s->active[enemy]; p < s->size[enemy]; ++p)
            if (s->hp[enemy][s->order[enemy][p]] > 0) { positions[n] = p; cards[n] = s->order[enemy][p]; ++n; }
        for (i = n; i-- > 1;) {
            unsigned j = pick(ch, i + 1), t = cards[i];
            cards[i] = cards[j]; cards[j] = t;
        }
        for (i = 0; i < n; ++i) s->order[enemy][positions[i]] = cards[i];
        if (front(s, enemy) != enemy_index && !s->entered[enemy][front(s, enemy)] && enter(battle, options, s, enemy, ch)) return 1;
    }
    cleanse(battle, s);
    return 0;
}

/* A card entering play; body_only re-applies the entry effects of a new form (Glamour, IMG_0330). */
/* Awakened Toy Jack-in-the-Box (IMG_0355): every enemy that enters is confused for a turn per unique Toy. */
static void enemy_entry(const ce_battle *battle, state *s, unsigned side) {
    unsigned index = front(s, side), enemy = 1 - side, turns;
    if (s->active[enemy] >= s->size[enemy] || s->hp[side][index] <= 0) return;
    turns = (unsigned)battle->fighters[enemy][front(s, enemy)].enemy_entry_confusion_turns;
    if (turns && s->hp[enemy][front(s, enemy)] > 0 && s->confused[side][index] < turns) s->confused[side][index] = turns;
}

static int arrive(const ce_battle *battle, const ce_options *options, state *s, unsigned side) {
    if (steal(battle, options, s, side)) return 1;
    enemy_entry(battle, s, side);
    return 0;
}

static int enter_card(const ce_battle *battle, const ce_options *options, state *s, unsigned side, chance_ctx *ch, int body_only) {
    unsigned index;
    double arriving_hp;
    unsigned other = 1 - side;
    const ce_fighter *fighter;
    if (s->active[side] >= s->size[side]) return 0;
    index = front(s, side);
    fighter = &battle->fighters[side][index];
    /* A returning card repeats no entry effect (user-observed Poseidon,
     * Knightmare, Good Boy); only an enemy thief may still react. */
    if (s->entered[side][index] && !body_only) return arrive(battle, options, s, side);
    s->entered[side][index] = 1;
    arriving_hp = s->hp[side][index]; /* may already be <= 0 if hit while waiting */
    s->hp[side][index] = rounded(s->hp[side][index] * fighter->entry_self_multiplier * fighter->entry_hp_multiplier, options->stat_rounding);
    s->max_hp[side][index] = rounded(s->max_hp[side][index] * fighter->entry_self_multiplier * fighter->entry_hp_multiplier, options->stat_rounding);
    s->attack[side][index] = rounded(s->attack[side][index] * fighter->entry_self_multiplier * fighter->entry_attack_multiplier, options->stat_rounding);
    if ((s->hp[side][index] <= 0 && arriving_hp > 0) || !isfinite(s->hp[side][index]) || !isfinite(s->attack[side][index]) ||
        !isfinite(s->max_hp[side][index])) return 1;
    {
        unsigned fallen = s->active[side], p;
        int r = options->stat_rounding;
        if (fighter->entry_fallen_stat_bonus > 0 && fallen) {
            double factor = 1 + fighter->entry_fallen_stat_bonus * fallen;
            s->hp[side][index] = rounded(s->hp[side][index] * factor, r);
            s->max_hp[side][index] = rounded(s->max_hp[side][index] * factor, r);
            s->attack[side][index] = rounded(s->attack[side][index] * factor, r);
        }
        if (fighter->entry_fallen_toy_stat_bonus > 0) { /* awakened Toy Bear: per fallen awakened Toy */
            unsigned toy, toys = 0;
            for (toy = 0; toy < battle->counts[side]; ++toy)
                if (battle->fighters[side][toy].awakened_toy && s->hp[side][toy] <= 0) toys++;
            if (toys) scale(s, side, index, 1 + fighter->entry_fallen_toy_stat_bonus * toys,
                            1 + fighter->entry_fallen_toy_stat_bonus * toys, r);
        }
        if (fighter->entry_fallen_attack_fraction > 0 && fallen) {
            double total = 0;
            for (p = 0; p < fallen; ++p) total += s->attack[side][s->order[side][p]];
            s->attack[side][index] = rounded(s->attack[side][index] + fighter->entry_fallen_attack_fraction * total, r);
        }
        if (fighter->fade_stat_multiplier != 1) {
            s->hp[side][index] = rounded(s->hp[side][index] * fighter->fade_stat_multiplier, r);
            s->max_hp[side][index] = rounded(s->max_hp[side][index] * fighter->fade_stat_multiplier, r);
            s->attack[side][index] = rounded(s->attack[side][index] * fighter->fade_stat_multiplier, r);
        }
        if (fighter->entry_next_card_stat_multiplier != 1 && fallen + 1 < s->size[side]) {
            unsigned nxt = s->order[side][fallen + 1];
            double m = fighter->entry_next_card_stat_multiplier;
            s->hp[side][nxt] = rounded(s->hp[side][nxt] * m, r);
            s->max_hp[side][nxt] = rounded(s->max_hp[side][nxt] * m, r);
            s->attack[side][nxt] = rounded(s->attack[side][nxt] * m, r);
        }
        if (fighter->entry_party_attack_bonus > 0)
            s->attack[side][index] = rounded(s->attack[side][index] *
                (1 + fighter->entry_party_attack_bonus * s->size[side]), r); /* User: dead or alive. */
        if (fighter->entry_fossil_attack_multiplier != 1 && s->fossils[side]) /* Tyrannodon (IMG_0314) */
            s->attack[side][index] = rounded(s->attack[side][index] * pow(fighter->entry_fossil_attack_multiplier, (double)s->fossils[side]), r);
        refresh(&s->slowed[side][index], fighter->entry_self_slow_turns);
        if (!isfinite(s->hp[side][index]) || !isfinite(s->attack[side][index])) return 1;
    }
    if (s->active[other] < s->size[other]) {
        unsigned other_index = front(s, other);
        s->attack[other][other_index] = rounded(s->attack[other][other_index] * fighter->entry_enemy_attack_multiplier
                                                * fighter->entry_enemy_stat_multiplier, options->stat_rounding);
        s->hp[other][other_index] = rounded(s->hp[other][other_index] * fighter->entry_enemy_hp_multiplier
                                            * fighter->entry_enemy_stat_multiplier, options->stat_rounding);
        s->max_hp[other][other_index] = rounded(s->max_hp[other][other_index] * fighter->entry_enemy_stat_multiplier, options->stat_rounding);
        if (!isfinite(s->attack[other][other_index]) || !isfinite(s->hp[other][other_index])) return 1;
        refresh(&s->burn[other][other_index], fighter->entry_burn_turns);
        refresh(&s->frozen[other][other_index], fighter->entry_freeze_turns);
        refresh(&s->slowed[other][other_index], fighter->entry_slow_turns);
        {
            const ce_fighter *e = &battle->fighters[other][other_index];
            int r = options->stat_rounding;
            if (fighter->entry_enemy_stat_subtract_fraction > 0) {
                /* Clamped: entry effects cannot kill in this subset. */
                double cut = fighter->entry_enemy_stat_subtract_fraction;
                s->hp[other][other_index] = fmax(1, rounded(s->hp[other][other_index] - s->hp[side][index] * cut, r));
                s->max_hp[other][other_index] = fmax(s->hp[other][other_index],
                    rounded(s->max_hp[other][other_index] - s->hp[side][index] * cut, r));
                s->attack[other][other_index] = fmax(0, rounded(s->attack[other][other_index] - s->attack[side][index] * cut, r));
            }
            if (fighter->entry_steal_attack_fraction > 0 &&
                (fighter->entry_steal_attack_chance <= 0 || roll(ch, fighter->entry_steal_attack_chance))) {
                double take = rounded(s->attack[other][other_index] * fighter->entry_steal_attack_fraction, r);
                s->attack[other][other_index] = rounded(s->attack[other][other_index] - take, r);
                s->attack[side][index] = rounded(s->attack[side][index] + take, r);
            }
            if (fighter->entry_steal_fraction > 0) transfer(s, other, other_index, side, index, fighter->entry_steal_fraction, r);
            if (fighter->entry_borderless_reset_chance > 0 && roll(ch, fighter->entry_borderless_reset_chance)) {
                s->hp[other][other_index] = s->max_hp[other][other_index] = e->borderless_hp > 0 ? e->borderless_hp : e->hp;
                s->attack[other][other_index] = e->borderless_attack > 0 ? e->borderless_attack : e->attack;
            }
            if (fighter->entry_random_debuff_turns) {
                /* Old Man Winter: frostbite, slow or freeze, equally likely (extrapolated). */
                uint32_t *target = roll(ch, 1.0 / 3) ? &s->frostbite[other][other_index] :
                    roll(ch, .5) ? &s->slowed[other][other_index] : &s->frozen[other][other_index];
                refresh(target, fighter->entry_random_debuff_turns);
            }
            if (fighter->entry_confusion_turns && !(fighter->confusion_fail_attack_ratio > 0 &&
                s->attack[other][other_index] >= s->attack[side][index] * fighter->confusion_fail_attack_ratio))
                refresh(&s->confused[other][other_index], fighter->entry_confusion_turns);
        }
        {
            /* Ice King / Scarecrow: statuses for every enemy card met while this card is active. */
            unsigned k;
            for (k = 0; k < 2; ++k) {
                unsigned ss = k ? other : side, si = k ? other_index : index, ds = k ? side : other, di = k ? index : other_index;
                const ce_fighter *source = &battle->fighters[ss][si], *met = &battle->fighters[ds][di];
                if (s->hp[ss][si] > 0 && s->hp[ds][di] > 0 &&
                    (!source->encounter_class_mask || (source->encounter_class_mask & met->class_mask))) {
                    refresh(&s->frozen[ds][di], source->encounter_freeze_turns);
                    refresh(&s->confused[ds][di], source->encounter_confusion_turns);
                    if (source->encounter_confusion_eternal && s->confused[ds][di] < CONFUSION_ETERNAL)
                        s->confused[ds][di] = CONFUSION_ETERNAL; /* Cthulu */
                    if (source->encounter_doom_turns && !s->doom[ds][di])
                        s->doom[ds][di] = (uint32_t)source->encounter_doom_turns; /* Kira: an existing doom is not extended */
                }
            }
        }
        if (s->hp[other][other_index] > 0 && (fighter->entry_disable_enemy ||
            (fighter->entry_disable_enemy_class_mask & battle->fighters[other][other_index].class_mask)))
            s->ability[other][other_index] = ABILITY_NONE; /* Hell's Army; Night Witch's hex on RNG cards */
        refresh(&s->suppressed[other], fighter->entry_suppress_enemy_turns); /* Fuxi */
        if (fighter->copy_fallen_ally) {
            int q; /* Hades, Legends: the most recently fallen ally */
            for (q = (int)s->active[side] - 1; q >= 0; --q) {
                unsigned ally = s->order[side][q];
                if (s->hp[side][ally] <= 0) { s->ability[side][index] = ability_code(s, side, ally); break; }
            }
        }
        if (fighter->entry_blind && s->hp[other][other_index] > 0) s->blind[other][other_index] = 1;
        if (fighter->delayed_strike_turns) {
            s->strike_timer[side][index] = (uint32_t)fighter->delayed_strike_turns;
            s->strike_damage[side][index] = s->attack[side][index] * fighter->delayed_strike_multiplier;
        }
        if (fighter->entry_poison_turns) {
            double amount = s->attack[side][index] * fighter->poison_fraction;
            unsigned q, last = fighter->entry_poison_all ? s->size[other] : s->active[other] + 1;
            for (q = s->active[other]; q < last; ++q)
                if (s->hp[other][s->order[other][q]] > 0)
                    poison_card(s, other, s->order[other][q], fighter->poison_permanent ? POISON_PERMANENT : fighter->entry_poison_turns, amount);
        }
    }
    if (entry_effects(battle, options, s, side, index, ch)) return 1;
    return body_only ? 0 : arrive(battle, options, s, side);
}

static int enter(const ce_battle *battle, const ce_options *options, state *s, unsigned side, chance_ctx *ch) {
    return enter_card(battle, options, s, side, ch, 0);
}

/* Whether the card now in front has an entry attack, in its current form (Awakened One's War Scythe). */
static int queued_entry(const ce_battle *base, const state *s, unsigned side) {
    ce_battle storage;
    const ce_battle *view = ability_view(base, s, &storage, NULL);
    return has_entry_hit(&view->fighters[side][front(s, side)], s->active[side]);
}

/* Savior / Frankenstein / Valentine's Specter: on death, the enemy front card loses a
 * share of the dead card's max HP (direct loss, extrapolated). */
static void blast(const ce_battle *battle, const ce_options *options, state *s, unsigned side, unsigned dead,
                  unsigned victim_side, unsigned victim, chance_ctx *ch) {
    const ce_fighter *f = &battle->fighters[side][dead];
    if (f->death_damage_max_hp_fraction > 0 && s->hp[victim_side][victim] > 0 && roll(ch, f->death_damage_chance))
        s->hp[victim_side][victim] -= rounded(s->max_hp[side][dead] * f->death_damage_max_hp_fraction, options->rounding);
}


/* Call after advancing active past a dead fighter. True Prophet (user-stated
 * on-death ability) grants its dodge to the card that comes in next. The heir is the
 * next living card to play; for a card killed in the deck that is the current front
 * card (user: Arthur used True Prophet's grant after Prophet died waiting). */
static void on_death(const ce_battle *battle, const ce_options *options, state *s, unsigned side, unsigned dead, chance_ctx *ch,
                     int in_deck) {
    const ce_fighter *f = &battle->fighters[side][dead];
    unsigned heir;
    double add_hp, add_attack;
    s->fossils[side] += (uint32_t)(f->dinosaur + f->death_fossils); /* user: every dying dinosaur adds one */
    if (f->death_frostbite_turns && s->active[1 - side] < s->size[1 - side] &&
        s->hp[1 - side][front(s, 1 - side)] > 0)
        refresh(&s->frostbite[1 - side][front(s, 1 - side)], f->death_frostbite_turns);
    if (f->death_ally_stat_multiplier != 1) {
        unsigned q;
        for (q = s->active[side]; q < s->size[side]; ++q) {
            unsigned ally = s->order[side][q];
            if (ally != dead && s->hp[side][ally] > 0)
                scale(s, side, ally, f->death_ally_stat_multiplier, f->death_ally_stat_multiplier, options->stat_rounding);
        }
    }
    death_effects(battle, options, s, side, dead, ch, in_deck);
    {
        /* User: a front card's effects go to the next card, dead or alive (they can be wasted);
         * a card killed in the deck gives them to the card still playing. */
        unsigned q = s->active[side];
        if (in_deck)
            while (q < s->size[side] && (s->order[side][q] == dead || s->hp[side][s->order[side][q]] <= 0)) ++q;
        if (q >= s->size[side]) return;
        heir = s->order[side][q];
    }
    s->granted_dodges[side][heir] += (uint32_t)f->next_card_dodges + s->granted_dodges[side][dead]; /* unused grants pass on */
    s->granted_dodges[side][dead] = 0;
    add_hp = s->max_hp[side][dead] * (f->next_card_stat_add_fraction + f->next_card_max_hp_add_fraction);
    add_attack = s->attack[side][dead] * f->next_card_stat_add_fraction;
    if ((add_hp != 0 || add_attack != 0) && f->next_card_gift_chance < 1 && !roll(ch, f->next_card_gift_chance))
        add_hp = add_attack = 0;
    if (s->stored[side][dead] != 0)
        s->attack[side][heir] = rounded(s->attack[side][heir] + s->stored[side][dead], options->stat_rounding);
    if (f->next_card_stat_multiplier != 1)
        scale(s, side, heir, f->next_card_stat_multiplier, f->next_card_stat_multiplier, options->stat_rounding);
    refresh(&s->frozen[side][heir], f->next_card_freeze_turns);
    if (f->next_card_boost_multiplier != 1) s->triggered[side][heir] |= SAM_BOOST; /* Uncle Sam */
    if (add_hp != 0 || add_attack != 0) {
        s->hp[side][heir] = rounded(s->hp[side][heir] + add_hp, options->stat_rounding);
        s->max_hp[side][heir] = rounded(s->max_hp[side][heir] + add_hp, options->stat_rounding);
        s->attack[side][heir] = rounded(s->attack[side][heir] + add_attack, options->stat_rounding);
    }
}

static int initialize(const ce_battle *battle, const ce_options *options, state *s, chance_ctx *ch) {
    unsigned side, i;
    memset(s, 0, sizeof(*s));
    for (side = 0; side < 2; ++side) {
        s->size[side] = battle->lineup[side];
        for (i = 0; i < battle->counts[side]; ++i) {
            s->hp[side][i] = battle->fighters[side][i].hp;
            s->max_hp[side][i] = battle->fighters[side][i].hp;
            s->attack[side][i] = battle->fighters[side][i].attack == 0 ? 0 : battle->fighters[side][i].attack;
            if (i < battle->lineup[side]) s->order[side][i] = i;
            s->incoming_scale[side][i] = 1;
            s->weakness[side][i] = 1;
        }
        for (i = 0; i < s->size[side]; ++i) {
            /* Longmu's Mother of Dragons: dragons in its deck are shielded from attacks from the start. */
            const ce_fighter *card = &battle->fighters[side][s->order[side][i]];
            unsigned j;
            if (!card->team_class_shields) continue;
            for (j = 0; j < s->size[side]; ++j) {
                unsigned ally = s->order[side][j];
                if (ally != s->order[side][i] && (battle->fighters[side][ally].class_mask & card->ally_class_mask))
                    s->shields[side][ally] += (uint32_t)card->team_class_shields;
            }
        }
    }
    s->side = battle->first_side;
    if (enter(battle, options, s, 0, ch) || enter(battle, options, s, 1, ch)) return 1;
    for (i = 0; i < 2; ++i) {
        side = i == 0 ? battle->first_side : 1-battle->first_side;
        if (queued_entry(battle, s, side))
            s->entry_queue[s->entry_count++] = side;
    }
    return 0;
}

static int heal(state *s, unsigned side, unsigned slot, double raw_amount, int rounding) {
    double before = s->hp[side][slot], amount, after;
    if (before <= 0) return 0;
    if (!isfinite(raw_amount)) return 1;
    amount = rounded(raw_amount, rounding);
    after = before + amount;
    if (!isfinite(after)) return 1;
    s->hp[side][slot] = fmax(before, fmin(after, s->max_hp[side][slot]));
    return 0;
}

/* Extrapolated on-hit reactions after a damaging hit, before counters. Mirrors
 * reference._hit_effects: attacker effects, defender reactions, retaliation. */
static int hit_effects(const ce_options *options, state *s, unsigned side, unsigned a_slot, unsigned other,
                       unsigned d_slot, const ce_fighter *a, const ce_fighter *b, double loss) {
    int r = options->stat_rounding;
    double retaliation = 0;
    refresh(&s->burn[other][d_slot], a->hit_burn_turns);
    refresh(&s->bleed[other][d_slot], a->hit_bleed_turns);
    refresh(&s->burn[side][a_slot], b->burn_on_attacked_turns);
    refresh(&s->bleed[side][a_slot], b->bleed_on_attacked_turns);
    poison_card(s, other, d_slot, a->poison_permanent && a->hit_poison_turns ? POISON_PERMANENT : a->hit_poison_turns,
                a->poison_target_max_hp_fraction > 0 ?
                s->max_hp[other][d_slot] * a->poison_target_max_hp_fraction : s->attack[side][a_slot] * a->poison_fraction);
    poison_card(s, side, a_slot, b->attacked_poison_turns, s->attack[other][d_slot] * b->poison_fraction);
    s->weakness[other][d_slot] = fmax(s->weakness[other][d_slot], a->hit_weakness_multiplier);
    if (a->steal_first_attacked && !(s->triggered[side][a_slot] & 8)) {
        s->triggered[side][a_slot] |= 8; /* Hecate: takes the first damaged card's ability; the victim loses it */
        s->ability[side][a_slot] = ability_code(s, other, d_slot);
        s->ability[other][d_slot] = ABILITY_NONE;
    }
    if (a->hit_disable) s->ability[other][d_slot] = ABILITY_NONE; /* Set */
    if (a->fossil_death_timer && !s->death_timer[other][d_slot] && s->hp[other][d_slot] > 0) {
        int timer = a->fossil_death_timer - (int)(s->fossils[side] < FOSSIL_DEATH_TIMER_CAP ? s->fossils[side] : FOSSIL_DEATH_TIMER_CAP);
        s->death_timer[other][d_slot] = (uint32_t)(timer > 1 ? timer : 1);
    }
    {
        double threshold = s->max_hp[other][d_slot] * b->low_hp_trigger_threshold;
        if (s->hp[other][d_slot] > 0 && s->hp[other][d_slot] < threshold && threshold <= s->hp[other][d_slot] + loss)
            if (b->low_hp_stun_turns && !(s->triggered[other][d_slot] & 4)) {
                s->triggered[other][d_slot] |= 4; /* Banshee (user): stuns only once */
                refresh(&s->slowed[side][a_slot], b->low_hp_stun_turns);
            }
        if (b->low_hp_heal_fraction > 0 && !(s->triggered[other][d_slot] & 1) && s->hp[other][d_slot] > 0 &&
            s->hp[other][d_slot] < threshold) {
            s->triggered[other][d_slot] |= 1;
            if (heal(s, other, d_slot, s->max_hp[other][d_slot] * b->low_hp_heal_fraction, options->rounding)) return 1;
        }
    }
    if (a->hit_attack_gain_fraction > 0)
        s->attack[side][a_slot] = rounded(s->attack[side][a_slot] + loss * a->hit_attack_gain_fraction, r);
    if (a->hit_attack_drain_fraction > 0) {
        const double taken = fmin(s->attack[other][d_slot], rounded(loss * a->hit_attack_drain_fraction, r));
        s->attack[other][d_slot] = rounded(s->attack[other][d_slot] - taken, r);
        s->attack[side][a_slot] = rounded(s->attack[side][a_slot] + taken, r);
    }
    if (a->hit_hp_gain_fraction > 0 && s->hp[side][a_slot] > 0) {
        s->hp[side][a_slot] = rounded(s->hp[side][a_slot] + loss * a->hit_hp_gain_fraction, r);
        s->max_hp[side][a_slot] = fmax(s->max_hp[side][a_slot], s->hp[side][a_slot]);
    }
    if (s->hp[other][d_slot] > 0) {
        if (a->hit_target_attack_reduction_fraction > 0) {
            double v = rounded(s->attack[other][d_slot] - loss * a->hit_target_attack_reduction_fraction, r);
            s->attack[other][d_slot] = v < 0 ? 0 : v;
        }
        if (a->hit_target_max_hp_reduction_fraction > 0)
            s->max_hp[other][d_slot] = fmax(s->hp[other][d_slot],
                rounded(s->max_hp[other][d_slot] - loss * a->hit_target_max_hp_reduction_fraction, r));
        if (a->hit_target_attack_multiplier != 1)
            s->attack[other][d_slot] = rounded(s->attack[other][d_slot] * a->hit_target_attack_multiplier, r);
        if (a->hit_steal_fraction > 0) transfer(s, other, d_slot, side, a_slot, a->hit_steal_fraction, r);
        if (b->damaged_attack_multiplier != 1)
            s->attack[other][d_slot] = rounded(s->attack[other][d_slot] * b->damaged_attack_multiplier, r);
        if (b->damaged_stat_multiplier != 1)
            scale(s, other, d_slot, b->damaged_stat_multiplier, b->damaged_stat_multiplier, r);
        if (b->damaged_heal_max_hp_fraction > 0 &&
            heal(s, other, d_slot, s->max_hp[other][d_slot] * b->damaged_heal_max_hp_fraction, options->rounding)) return 1;
        if (b->attacked_steal_fraction > 0) transfer(s, side, a_slot, other, d_slot, b->attacked_steal_fraction, r);
    }
    if (b->thorns_attack_fraction > 0 || b->reflect_damage_fraction > 0)
        retaliation = rounded(s->attack[other][d_slot] * b->thorns_attack_fraction + loss * b->reflect_damage_fraction,
                              options->rounding);
    if (retaliation != 0) s->hp[side][a_slot] -= retaliation;
    if (a->recoil_damage_fraction > 0) s->hp[side][a_slot] -= rounded(loss * a->recoil_damage_fraction, options->rounding);
    if (s->hp[other][d_slot] > 0 && s->hp[other][d_slot] < s->max_hp[other][d_slot] * a->execute_after_below_max_hp)
        s->hp[other][d_slot] = 0;
    if (s->hp[other][d_slot] > 0 && s->hp[other][d_slot] < s->max_hp[other][d_slot] * b->mutual_destruction_below_max_hp)
        s->hp[other][d_slot] = s->hp[side][a_slot] = 0;
    return !isfinite(s->hp[side][a_slot]) || !isfinite(s->attack[side][a_slot]) ||
        !isfinite(s->hp[other][d_slot]) || !isfinite(s->attack[other][d_slot]) || !isfinite(s->max_hp[side][a_slot]);
}

/* Damage every waiting card of `side`. Attacks can be dodged, absorbed by Prophet
 * grants, invincibility or lethal survival; plain HP loss cannot. Cards reduced to 0 HP
 * die in the deck (user Seraphim tests); their on-death effects go to the next card to play. */
static void bench_hits(const ce_battle *battle, const ce_options *options, state *s, unsigned side, double amount,
                       int as_attack, chance_ctx *ch, unsigned limit, int on_death_effects, const unsigned *positions, unsigned position_count) {
    unsigned position, killed[CE_MAX_FIGHTERS], count = 0, k, step;
    unsigned end = limit && s->active[side] + 1 + limit < s->size[side] ? s->active[side] + 1 + limit : s->size[side];
    unsigned steps = positions ? position_count : (end > s->active[side] + 1 ? end - s->active[side] - 1 : 0);
    for (step = 0; step < steps; ++step) {
        unsigned index;
        double hp_before;
        position = positions ? positions[step] : s->active[side] + 1 + step;
        index = s->order[side][position];
        const ce_fighter *card = &battle->fighters[side][index];
        double damage;
        if (s->hp[side][index] <= 0) continue;
        if (as_attack) {
            if (roll(ch, card->dodge_probability) || card->invincible) continue;
            if (card->alternate_dodge) {
                /* User: Deus Ex always dodges the entry while waiting; deck hits count toward the alternation. */
                s->hits_taken[side][index]++;
                if (s->hits_taken[side][index] % 2 == (uint32_t)card->alternate_dodge % 2) continue;
            }
            if (s->granted_dodges[side][index]) { s->granted_dodges[side][index]--; continue; }
            damage = rounded(amount * card->incoming_multiplier, options->rounding);
            if (on_death_effects && damage >= s->hp[side][index] && s->survivals_used[side][index] < (unsigned)card->lethal_survivals) {
                s->survivals_used[side][index]++;
                s->hp[side][index] = card->lethal_survival_hp_fraction > 0 ?
                    fmax(1, rounded(s->max_hp[side][index] * card->lethal_survival_hp_fraction, options->stat_rounding)) : 1;
                continue;
            }
        } else damage = rounded(amount, options->rounding);
        if (as_attack && s->shields[side][index]) { s->shields[side][index]--; continue; } /* a shield blocks the whole attack */
        if (on_death_effects && s->hp[side][index] > 0 && s->hp[side][index] <= damage && card->death_reflects > 0 &&
            s->survivals_used[side][index] >= (unsigned)card->lethal_survivals &&
            s->survivals_used[side][index] - (unsigned)card->lethal_survivals < (unsigned)card->death_reflects) {
            /* User Seraphim tests: Parallax hit in the deck reflects its death onto the attacker. */
            unsigned killer = s->order[1 - side][s->active[1 - side]];
            s->survivals_used[side][index]++;
            s->hp[1 - side][killer] = fmin(s->hp[1 - side][killer], 0);
            continue;
        }
        hp_before = s->hp[side][index];
        ability_hit(battle, s, side, index, damage, options->rounding);
        if (s->hp[side][index] <= 0 && hp_before > 0) killed[count++] = index;
    }
    for (k = 0; k < count && on_death_effects; ++k) {
        blast(battle, options, s, side, killed[k], 1 - side, s->order[1 - side][s->active[1 - side]], ch);
        on_death(battle, options, s, side, killed[k], ch, 1);
    }
}

/* Resolve one hit. Dodges consume a hit but no block; actions alternate only
 * after every hit resolves. Remaining hits are part of the hashed state. */
static int advance(const ce_battle *base_battle, const ce_options *options,
                   const state *before, state *after, int dodged, int critical, chance_ctx *ch) {
    ce_battle view_storage;
    unsigned failed[2] = {0, 0};
    const ce_battle *battle = ability_view(base_battle, before, &view_storage, end_times(base_battle, before, ch, failed));
    unsigned normal_side = before->side;
    unsigned side = hit_actor(before), other = 1-side;
    int is_entry = before->entry_count && !before->counter_pending;
    int entry_context = is_entry || (before->counter_pending && before->counter_from_entry);
    unsigned attacker = front(before, side), defender = front(before, other);
    /* A card dies only when its HP is actually lowered to <= 0 in this step. */
    double actor_hp_start = before->hp[side][attacker], enemy_hp_start = before->hp[other][defender];
    unsigned normal_slot = front(before, normal_side);
    unsigned changed[2] = {0, 0};
    unsigned total_hits = before->counter_pending || is_entry ? 1 :
        action_hits(&battle->fighters[side][attacker], before->turns[side][attacker]);
    unsigned remaining = before->counter_pending || is_entry ? 1 : before->hits_remaining ? before->hits_remaining : total_hits;
    unsigned hit_index = total_hits - remaining + 1;
    int periodic = !before->counter_pending && !is_entry && periodic_turn(&battle->fighters[side][attacker], before->turns[side][attacker]);
    int followup_hit = battle->fighters[side][attacker].followup_multiplier > 0 && !before->counter_pending && !is_entry &&
        hit_index == total_hits;
    unsigned team;
    int intercepted = -1; /* side whose active card was displaced */
    int missed, displaced_actor, turn_end, attacker_died, target_died, alternate = 0, first_action, turn_index, skipped = 0, turn_ends, kill_extra = 0;
    int relocated = 0, reflect = 0, redirected = 0, deck_target = -1, stole = 0;
    ce_battle stolen_view;
    const ce_fighter *nf;
    unsigned kraken = 1, chosen[CE_MAX_FIGHTERS], chosen_count = 0, deck_hits = 0, deck_side[3 * CE_MAX_FIGHTERS], deck_index[3 * CE_MAX_FIGHTERS];
    double dealt = 0, actor_lost, swap_boost = 1, punish = 1, deck_before[3 * CE_MAX_FIGHTERS];
    if (battle->fighters[side][attacker].random_hits_max && !before->counter_pending && !is_entry) {
        /* The Curse: 2-5 slashes, equally likely, decided on the first. */
        const ce_fighter *a = &battle->fighters[side][attacker];
        total_hits = before->hits_remaining ? before->hits_remaining + 1 :
            (unsigned)a->random_hits_min + pick(ch, (unsigned)(a->random_hits_max - a->random_hits_min + 1));
        remaining = before->hits_remaining ? before->hits_remaining : total_hits;
        hit_index = total_hits - remaining + 1;
        followup_hit = battle->fighters[side][attacker].followup_multiplier > 0 && hit_index == total_hits;
    }
    memcpy(after, before, sizeof(*after));
    if (is_entry) cancel_entry(after, side);
    if (!before->counter_pending && !is_entry && !before->hits_remaining && !before->turn_actions_remaining) {
        const ce_fighter *own = &base_battle->fighters[side][attacker];
        int changed_ability = 0, shifted = 0;
        if (own->turn_start_random_ability && after->ability[side][attacker] != ABILITY_NONE && !(after->triggered[side][attacker] & SHIFTED)) {
            int32_t code = pool_pick(base_battle, ch, own->pool_exclude_mask, own->pool_self - 1, -1); /* Glamour */
            if (code >= 0) { after->ability[side][attacker] = code; after->ability2[side][attacker] = 0; changed_ability = shifted = 1; }
        }
        if (battle->fighters[side][attacker].awaken_turn &&
            after->turns[side][attacker] == (uint32_t)(battle->fighters[side][attacker].awaken_turn - 1) &&
            !(after->triggered[side][attacker] & AWAKENED)) { /* Demon / Immortal Cultivator */
            after->triggered[side][attacker] |= AWAKENED;
            after->ability[side][attacker] = POOL_CODE + battle->fighters[side][attacker].transform_form - 1;
            changed_ability = 1;
        }
        if (changed_ability) battle = ability_view(base_battle, after, &view_storage, failed);
        if (shifted) { /* IMG_0330: the new form's entry effects apply; its entry attack comes first */
            if (enter_card(battle, options, after, side, ch, 1)) return 1;
            battle = ability_view(base_battle, after, &view_storage, failed);
            if (has_entry_hit(&battle->fighters[side][attacker], after->active[side])) {
                after->triggered[side][attacker] |= SHIFTED;
                if (after->entry_count >= 2) return 1;
                after->entry_queue[after->entry_count++] = side;
                return 0;
            }
        }
    }
    if (!before->counter_pending && !is_entry && !before->hits_remaining) {
        const ce_fighter *a = &battle->fighters[side][attacker];
        const ce_fighter *e = &battle->fighters[other][defender];
        double drain;
        cleanse(battle, after);
        if (e->enemy_turn_start_attack_multiplier != 1 || e->enemy_turn_start_stat_multiplier != 1)
            scale(after, side, attacker, e->enemy_turn_start_stat_multiplier,
                  e->enemy_turn_start_attack_multiplier * e->enemy_turn_start_stat_multiplier, options->stat_rounding);
        drain = after->hp[side][attacker] * (e->enemy_turn_start_hp_loss_fraction + a->self_turn_start_hp_loss_fraction) +
            after->max_hp[side][attacker] * (e->enemy_turn_start_max_hp_loss_fraction + a->self_turn_start_max_hp_loss_fraction);
        if (drain != 0) ability_hit(battle, after, side, attacker, rounded(drain, options->rounding), options->rounding);
        if (!before->turn_actions_remaining) {
            /* Poison ticks at the start of the poisoned side's turn (IMG_0297, IMG_0300, IMG_0305),
             * for waiting cards too (Black Plague). */
            unsigned q, n = after->size[side], snapshot[CE_MAX_FIGHTERS];
            for (q = 0; q < n; ++q) snapshot[q] = after->order[side][q];
            for (q = before->active[side] + 1; q < n; ++q) {
                unsigned index = snapshot[q];
                if (after->poison[side][index] && after->hp[side][index] > 0) {
                    double hp_before = after->hp[side][index];
                    after->poison[side][index]--;
                    ability_hit(battle, after, side, index, rounded(after->poison_damage[side][index], options->rounding), options->rounding);
                    if (after->hp[side][index] <= 0 && hp_before > 0) on_death(battle, options, after, side, index, ch, 1);
                }
            }
            if (after->poison[side][attacker]) {
                after->poison[side][attacker]--;
                ability_hit(battle, after, side, attacker, rounded(after->poison_damage[side][attacker], options->rounding), options->rounding);
            }
            if (after->curse[side][attacker]) { /* The Hanged Man / Naga (user): 25% max HP each turn */
                if (after->curse[side][attacker] < CURSE_PERMANENT) after->curse[side][attacker]--;
                ability_hit(battle, after, side, attacker, rounded(after->max_hp[side][attacker] * CURSE_MAX_HP_FRACTION, options->rounding),
                            options->rounding);
            }
            if (a->dad_card_id) { /* Milk (user): gives its stats to the Dad cards and dies */
                int dads = 0;
                for (q = 0; q < after->size[side]; ++q) {
                    unsigned i = after->order[side][q];
                    if (i != attacker && after->hp[side][i] > 0 && battle->fighters[side][i].card_id == a->dad_card_id) {
                        after->hp[side][i] += after->hp[side][attacker];
                        after->max_hp[side][i] += after->max_hp[side][attacker];
                        after->attack[side][i] += after->attack[side][attacker];
                        dads = 1;
                    }
                }
                if (dads) after->hp[side][attacker] = 0;
            }
        }
        if (!isfinite(after->hp[side][attacker]) || !isfinite(after->attack[side][attacker])) return 1;
        if (after->hp[side][attacker] <= 0 && after->hp[side][attacker] < actor_hp_start) {
            /* Extrapolated: the drained card dies and its side loses the turn. */
            int returning;
            after->active[side]++;
            on_death(battle, options, after, side, attacker, ch, 0);
            while (after->active[side] < after->size[side] && after->hp[side][front(after, side)] <= 0)
                after->active[side]++; /* skip cards that died waiting (e.g. poisoned in the deck) */
            cancel_entry(after, side);
            after->pair_actions = 0;
            if (terminal(battle, after)) return 0;
            returning = (int)after->entered[side][front(after, side)];
            if (enter(battle, options, after, side, ch)) return 1;
            if (!returning && queued_entry(base_battle, after, side)) {
                if (after->entry_count >= 2) return 1;
                after->entry_queue[after->entry_count++] = side;
            }
            after->side = 1 - side;
            after->turn_actions_remaining = 0;
            return 0;
        }
        if (a->action_start_heal_max_hp_fraction > 0) {
            if (heal(after, side, attacker, after->max_hp[side][attacker] *
                     a->action_start_heal_max_hp_fraction, options->rounding)) return 1;
        }
        if (a->action_start_heal_attack_fraction > 0) {
            if (heal(after, side, attacker, after->attack[side][attacker] *
                     a->action_start_heal_attack_fraction, options->rounding)) return 1;
        }
        if (a->restore_lowered_attack && after->attack[side][attacker] < a->attack) after->attack[side][attacker] = a->attack;
        after->resting[side][attacker] = 0; /* Its own turn ends the rest's protection. */
        if (a->periodic_block_period && after->turns[side][attacker] % (uint32_t)a->periodic_block_period == 0)
            after->used[side][attacker] = 0;
        if (a->turn_start_stat_growth > 0) { /* Limitless Rivals: fixed gain; the max HP gain also adds to HP */
            const double t = after->turns[side][attacker], g = a->turn_start_stat_growth;
            const double growth = (1 + g * (t + 1)) / (1 + g * t);
            const double gain = rounded(after->max_hp[side][attacker] * growth, options->stat_rounding) - after->max_hp[side][attacker];
            after->attack[side][attacker] = rounded(after->attack[side][attacker] * growth, options->stat_rounding);
            after->max_hp[side][attacker] += gain;
            after->hp[side][attacker] += gain;
        }
        if (a->turn_start_incoming_reduction > 0) { /* fixed steps up to the cap (IMG_0336, IMG_0337) */
            const double t = after->turns[side][attacker], g = a->turn_start_incoming_reduction, cap = a->incoming_reduction_cap;
            after->incoming_scale[side][attacker] *= (1 - fmin(cap, g * (t + 1))) / (1 - fmin(cap, g * t));
        }
        if (a->turn_start_steal_fraction > 0 && after->hp[other][defender] > 0) /* Mother of Beasts (IMG_0317) */
            transfer(after, other, defender, side, attacker, a->turn_start_steal_fraction, options->stat_rounding);
        if (a->turn_start_hp_steal_fraction > 0 && after->hp[other][defender] > 0) {
            double take = rounded(after->hp[other][defender] * a->turn_start_hp_steal_fraction, options->stat_rounding);
            after->hp[other][defender] -= take;
            after->hp[side][attacker] += take;
            after->max_hp[side][attacker] = fmax(after->max_hp[side][attacker], after->hp[side][attacker]);
        }
        if (roll(ch, a->turn_start_full_heal_chance))
            after->hp[side][attacker] = fmax(after->hp[side][attacker], after->max_hp[side][attacker]);
        if (roll(ch, a->turn_start_attack_chance))
            after->attack[side][attacker] = rounded(after->attack[side][attacker] * a->turn_start_attack_multiplier, options->stat_rounding);
        /* Astraeus's Aquarius (user): heals 30% at or below half HP, otherwise +25% max HP. */
        if (a->turn_start_low_hp_heal_fraction > 0 && after->hp[side][attacker] <= after->max_hp[side][attacker] * a->turn_start_low_hp_threshold) {
            if (heal(after, side, attacker, after->max_hp[side][attacker] * a->turn_start_low_hp_heal_fraction, options->rounding)) return 1;
        } else if (a->turn_start_max_hp_multiplier != 1) {
            double gain = rounded(after->max_hp[side][attacker] * a->turn_start_max_hp_multiplier, options->stat_rounding) - after->max_hp[side][attacker];
            after->max_hp[side][attacker] += gain;
            after->hp[side][attacker] += gain;
        }
        if (a->shield_period && after->turns[side][attacker] % (uint32_t)a->shield_period == 0 && !after->shields[side][attacker])
            after->shields[side][attacker] = 1; /* Shield of Ahimsa */
        if (!before->turn_actions_remaining) {
            unsigned q;
            for (q = after->active[side]; q < after->size[side]; ++q) {
                unsigned c = after->order[side][q];
                const ce_fighter *card = &battle->fighters[side][c];
                if (card->composer_chance_step > 0 && after->hp[side][c] > 0) { /* The Composer */
                    after->timer[side][c]++;
                    if (roll(ch, fmin(1, card->composer_chance_step * after->timer[side][c])) && after->hp[other][defender] > 0 &&
                        !after->confused[other][defender]) after->confused[other][defender] = 1;
                }
            }
            if (e->reflect_even_turns) { /* Tsukuyomi (user): the opponent's even turns are reflected */
                after->timer[other][defender]++;
                reflect = after->timer[other][defender] % 2 == 0;
            }
        }
        if (a->perish_turns && after->timer[side][attacker] && !before->turn_actions_remaining && !--after->timer[side][attacker]) {
            /* Sleep Paralysis (IMG_0345): at the start of its 4th turn both active cards die (deaths resolve below). */
            ability_hit(battle, after, other, defender, after->hp[other][defender], options->rounding);
            ability_hit(battle, after, side, attacker, after->hp[side][attacker], options->rounding);
            skipped = 1;
        }
        if (!skipped && after->slowed[side][attacker] && !after->frozen[side][attacker]) {
            after->slowed[side][attacker]--;
            skipped = 1;
        }
        if (!skipped && !after->frozen[side][attacker] && roll(ch, e->skip_enemy_turn_chance)) skipped = 1;
        if (!skipped && !after->frozen[side][attacker] && after->confused[side][attacker] && roll(ch, CONFUSION_CHANCE)) {
            /* User: confusion makes the card attack itself (about 50%); raw own damage. */
            after->hp[side][attacker] -= rounded(after->attack[side][attacker] * a->outgoing_multiplier, options->rounding);
            skipped = 1;
            if (after->confused[side][attacker] >= CONFUSION_ETERNAL) {
            } else if (after->confused[side][attacker] >= CONFUSION_UNTIL_SELF_HIT) { /* Kuchisake-onna */
                after->attack[side][attacker] = rounded(after->attack[side][attacker] * 0.8, options->stat_rounding);
                after->confused[side][attacker] = 0;
            }
            if (e->confused_self_hit_stat_multiplier != 1 && after->hp[other][defender] > 0) /* Cthulu */
                scale(after, other, defender, e->confused_self_hit_stat_multiplier, e->confused_self_hit_stat_multiplier, options->stat_rounding);
        }
        if (!skipped && !after->frozen[side][attacker] && reflect) { /* Tsukuyomi: the attack lands on the attacker */
            after->hp[side][attacker] -= rounded(after->attack[side][attacker] * a->outgoing_multiplier *
                                                 action_hits(a, after->turns[side][attacker]), options->rounding);
            refresh(&after->burn[side][attacker], a->hit_burn_turns);
            refresh(&after->bleed[side][attacker], a->hit_bleed_turns);
            refresh(&after->frostbite[side][attacker], a->hit_frostbite_turns);
            poison_card(after, side, attacker, a->poison_permanent && a->hit_poison_turns ? POISON_PERMANENT : a->hit_poison_turns,
                        after->attack[side][attacker] * a->poison_fraction);
            after->weakness[side][attacker] = fmax(after->weakness[side][attacker], a->hit_weakness_multiplier);
            skipped = 1;
        }
        if (!skipped && a->buddha) {
            int moved = buddha(after, side, attacker, options);
            if (moved < 0) return 1;
            relocated = moved;
            skipped = 1;
        }
        if (!skipped && a->never_attacks) skipped = 1;
        if (!skipped && a->alternate_rest && after->turns[side][attacker] % 2 == 1) {
            skipped = 1;
            after->resting[side][attacker] = 1; /* Untouchable until its next turn. */
        }
    }
    if (!before->counter_pending && !is_entry && after->frozen[side][attacker]) skipped = 1; /* user: entry hits ignore freeze */
    if (!skipped && !before->counter_pending && !is_entry && !before->hits_remaining) {
        /* User: Onmyoji and Kira count the normal attacks of the ally in play, just before each attack,
         * even after they died; they hit the enemy card in play. */
        double before_count = after->hp[other][defender];
        unsigned source;
        for (source = 0; source < battle->counts[side]; ++source)
            if (after->strike_timer[side][source] && !--after->strike_timer[side][source])
                ability_hit(battle, after, other, defender, rounded(after->strike_damage[side][source], options->rounding), options->rounding);
        if (after->doom[other][defender] && after->hp[other][defender] > 0 && !--after->doom[other][defender])
            after->hp[other][defender] = 0;
        if (after->hp[other][defender] <= 0 && before_count > 0) skipped = 1; /* provisional: the attack is spent */
    }
    if (!skipped && !before->counter_pending && !is_entry && hit_index == 1) {
        const ce_fighter *a = &battle->fighters[side][attacker];
        unsigned list[CE_MAX_FIGHTERS], n;
        if (a->swap_attack_chance > 0 && roll(ch, a->swap_attack_chance)) { /* Heavenly Demon */
            n = waiting(after, other, list);
            if (n) {
                unsigned p = list[pick(ch, n)];
                after->order[other][after->active[other]] = after->order[other][p];
                after->order[other][p] = defender;
                defender = front(after, other);
                enemy_hp_start = after->hp[other][defender];
                after->entered[other][defender] = 1;
                swap_boost = a->swap_attack_multiplier;
            }
        }
        if (a->random_deck_target) { /* Brachiosaurus */
            n = waiting(after, other, list);
            if (n) { deck_target = (int)list[pick(ch, n)]; redirected = 1; }
        }
        if (a->chaos_targets) { /* Chaos (user): each enemy hit independently */
            unsigned k;
            redirected = !roll(ch, a->chaos_hit_chance);
            n = waiting(after, other, list);
            for (k = 0; k < n && k < (unsigned)a->chaos_targets - 1; ++k)
                if (roll(ch, a->chaos_hit_chance)) chosen[chosen_count++] = list[k];
        }
    }
    if (!skipped && followup_hit && battle->fighters[side][attacker].followup_chance > 0 &&
        !roll(ch, battle->fighters[side][attacker].followup_chance)) redirected = 1; /* Storm Spirit */
    if (!skipped && is_entry && battle->fighters[side][attacker].entry_hit_count_max)
        kraken = 1 + pick(ch, (unsigned)battle->fighters[side][attacker].entry_hit_count_max); /* Kraken */
    if (!skipped && dodged && battle->fighters[side][attacker].evade_punish_multiplier > 0) {
        dodged = 0; /* Sagittarius */
        punish = battle->fighters[side][attacker].evade_punish_multiplier;
    }
    if (!skipped && battle->fighters[other][defender].alternate_dodge) {
        /* User-observed: 1 dodges odd incoming hits (Deus Ex), 2 even hits (Amaterasu). */
        alternate = after->hits_taken[other][defender] % 2 == (uint32_t)battle->fighters[other][defender].alternate_dodge - 1;
        after->hits_taken[other][defender]++;
    }
    if (!skipped && periodic && hit_index == 1) {
        const ce_fighter *a = &battle->fighters[side][attacker];
        if (a->periodic_attack_heal_max_hp_fraction > 0 && heal(after, side, attacker,
            after->max_hp[side][attacker] * a->periodic_attack_heal_max_hp_fraction, options->rounding)) return 1;
        refresh(&after->slowed[other][defender], a->periodic_enemy_slow_turns);
    }
    actor_lost = 1 - after->hp[side][attacker] / after->max_hp[side][attacker];
    missed = !skipped && after->blind[side][attacker] != 0; /* Rudolph: a blinded card's attacks miss */
    if (!skipped && !missed && is_entry && battle->fighters[side][attacker].entry_hit_next_card)
        missed = 1; /* Ra: its entry hit goes to the card behind instead */
    if (!skipped && !missed && redirected) missed = 1; /* Brachiosaurus / a missed Chaos hit */
    if (!skipped && !missed && dodged) {
        const ce_fighter *b = &battle->fighters[other][defender];
        int li = battle->fighters[side][attacker].domain ? -1 : luminant(battle, after, other, defender);
        if (li >= 0 && after->evades[other][defender] < LUMINANT_MAX) { /* Eclipseborn Luminant */
            after->evades[other][defender]++;
            after->attack[other][li] = fmin(rounded(after->attack[other][li] + after->attack[side][attacker] *
                                                    battle->fighters[side][attacker].outgoing_multiplier * LUMINANT_GAIN, options->stat_rounding),
                                            fmax(after->attack[other][li], LUMINANT_CAP * battle->fighters[other][li].attack));
        }
        if (b->dodge_attack_multiplier != 1)
            after->attack[other][defender] = rounded(after->attack[other][defender] * b->dodge_attack_multiplier, options->stat_rounding);
        if (b->dodge_bonus_multiplier != 1) after->dodge_charged[other][defender] = 1;
    }
    if (!skipped && !missed && !dodged && !battle->fighters[other][defender].invincible && !after->resting[other][defender] &&
        !after->undying[other][defender]) {
        const ce_fighter *a = &battle->fighters[side][attacker];
        const ce_fighter *b = &battle->fighters[other][defender];
        double *hp_d = &after->hp[other][defender], max_d = after->max_hp[other][defender];
        double hp_a = after->hp[side][attacker];
        double damage = (after->attack[side][attacker] + a->target_attack_damage_fraction * after->attack[other][defender] +
                         a->lost_hp_damage_fraction * (after->max_hp[side][attacker] - hp_a) +
                         a->max_hp_damage_fraction * after->max_hp[side][attacker]) * a->outgoing_multiplier +
                         a->target_current_hp_damage_fraction * after->hp[other][defender];
        double incoming = b->incoming_multiplier, dodge_threshold, nullify_threshold, reduction, cap;
        unsigned behind = after->active[other] + 1;
        int bypass = a->bypass_defenses || (is_entry && a->entry_bypass_defenses) || (followup_hit && a->followup_bypass);
        int interceptor = -1, threshold_dodge, nullified, sign_nullified, gamble_failed = 0, evaded;
        if (is_entry) damage *= a->entry_hit_multiplier * kraken;
        damage *= swap_boost * punish;
        if (!before->counter_pending && !is_entry && (after->triggered[side][attacker] & SAM_BOOST)) damage *= SAM_BOOST_MULTIPLIER;
        if (critical) damage *= a->critical_multiplier;
        if (before->counter_pending) damage *= a->counter_multiplier;
        if (!before->counter_pending && !is_entry) {
            if (hit_index == 1 && after->turns[side][attacker] == 0) damage *= a->first_attack_multiplier;
            /* User: recharge turns attack normally, without the charged bonus. */
            if (a->recharge_turns > 0 && after->turns[side][attacker] % (uint32_t)(1 + a->recharge_turns) == 0)
                damage *= a->charged_attack_multiplier;
            damage *= pow(a->hit_growth_multiplier, (double)(hit_index - 1));
        }
        if (!before->counter_pending && !is_entry) {
            if (after->dodge_charged[side][attacker]) {
                damage *= a->dodge_bonus_multiplier;
                after->dodge_charged[side][attacker] = 0;
            }
            if (a->random_multiplier_max > 1) {
                int k, factor = a->random_multiplier_max;
                for (k = 1; k < a->random_multiplier_max; ++k)
                    if (roll(ch, 1.0 / (a->random_multiplier_max - k + 1))) { factor = k; break; }
                damage *= factor;
            }
            if (a->gamble_chance > 0) {
                if (roll(ch, a->gamble_chance)) damage *= a->gamble_multiplier;
                else gamble_failed = 1;
            }
        }
        if (after->frostbite[side][attacker]) damage *= b->frostbite_incoming_multiplier;
        if (!after->faded[side][attacker]) damage *= a->fade_outgoing_multiplier;
        damage *= field_multiplier(battle, after);
        if (a->border_rarity > b->border_rarity) damage *= a->border_advantage_multiplier;
        if (b->pack_index > a->pack_index) damage *= a->younger_target_multiplier;
        if (a->bonus_vs_class_mask & b->class_mask) damage *= a->class_damage_multiplier;
        if (!before->counter_pending && !is_entry && a->first_hit_per_enemy_multiplier != 1 && !after->marked[other][defender]) {
            damage *= a->first_hit_per_enemy_multiplier;
            after->marked[other][defender] = 1;
            bypass = 1;
        }
        if (*hp_d <= max_d * a->weak_target_threshold) damage *= a->weak_target_multiplier;
        if (hp_a > *hp_d) damage *= a->advantage_damage_multiplier;
        damage *= 1 + a->lost_hp_damage_bonus * actor_lost;
        if (periodic) damage *= a->periodic_attack_multiplier;
        if (followup_hit) damage *= a->followup_multiplier;
        if (after->frozen[other][defender]) damage *= a->frozen_target_multiplier;
        if (a->max_hp_damage_costs && a->max_hp_damage_fraction > 0) {
            /* User (#161): the converted max HP is lost, decaying exponentially. */
            double keep = 1 - a->max_hp_damage_fraction;
            after->hp[side][attacker] = rounded(after->hp[side][attacker] * keep, options->stat_rounding);
            after->max_hp[side][attacker] = rounded(after->max_hp[side][attacker] * keep, options->stat_rounding);
        }
        incoming *= after->incoming_scale[other][defender];
        incoming *= after->weakness[other][defender];
        if (!after->faded[other][defender]) incoming *= b->fade_incoming_multiplier;
        if (b->border_rarity > a->border_rarity) incoming *= b->border_advantage_incoming_multiplier;
        if (a->card_rarity * a->border_rarity < b->card_rarity * b->border_rarity) incoming *= b->lower_rarity_incoming_multiplier;
        if (b->guard_vs_class_mask & a->class_mask) incoming *= b->class_incoming_multiplier;
        if (*hp_d < after->hp[side][attacker]) incoming *= b->disadvantage_incoming_multiplier; /* after any HP cost */
        if (*hp_d < max_d * b->low_hp_incoming_threshold) incoming *= b->low_hp_incoming_multiplier;
        if (b->lost_hp_incoming_bonus > 0) incoming /= 1 + b->lost_hp_incoming_bonus * (1 - *hp_d / max_d); /* Taurus */
        if (!after->negated[other]) incoming *= b->aura_incoming_multiplier; /* blue support (Marrowclaw negates) */
        /* Extrapolated: bypass ignores thresholds, reductions, caps and blocks,
         * keeping damage amplifiers; dodges still apply. */
        dodge_threshold = bypass ? 0 : max_d * b->dodge_below_max_hp;
        nullify_threshold = bypass ? 0 : after->attack[other][defender] * b->nullify_below_attack;
        reduction = bypass ? 0 : max_d * b->damage_reduction_max_hp;
        cap = max_d * b->damage_cap_max_hp;
        if (bypass) incoming = fmax(1, incoming);
        if (!isfinite(damage) || !isfinite(dodge_threshold) || !isfinite(nullify_threshold) || !isfinite(reduction) || !isfinite(cap)) return 1;
        threshold_dodge = damage < dodge_threshold;
        nullified = damage < nullify_threshold;
        sign_nullified = !bypass && b->nullify_max_hp_threshold > 0 && damage < max_d * fmax(0, b->nullify_max_hp_threshold -
                         b->nullify_threshold_step * after->timer[other][defender]);
        damage = fmax(0, damage - reduction) * incoming;
        if (!isfinite(damage)) return 1;
        if (b->damage_cap_max_hp > 0 && !bypass) damage = fmin(damage, cap);
        damage = rounded(damage, options->rounding);
        if (a->execute_below_max_hp > 0 && *hp_d < max_d * a->execute_below_max_hp) damage = fmax(damage, *hp_d);
        if (after->block_pool[other][defender] > 0 && !bypass) {
            double absorbed = fmin(after->block_pool[other][defender], damage); /* Steve's blocks soak damage first */
            after->block_pool[other][defender] -= absorbed;
            damage -= absorbed;
        }
        /* Own lethal survival is checked before a Piccolo-style interception. */
        if (damage >= *hp_d && after->survivals_used[other][defender] >= (unsigned)b->lethal_survivals &&
            behind < after->size[other] && after->hp[other][after->order[other][behind]] > 0 &&
            battle->fighters[other][after->order[other][behind]].intercept_lethal_multiplier > 0)
            interceptor = (int)after->order[other][behind];
        evaded = b->hit_chance_squared && !roll(ch, fmin(1, (damage / max_d) * (damage / max_d)));
        if (gamble_failed) {
            if (heal(after, other, defender, max_d * a->gamble_fail_heal_fraction, options->rounding)) return 1;
        } else if (!before->counter_pending && !is_entry && b->normal_attack_dodge_cost > 0) {
            *hp_d -= rounded(*hp_d * b->normal_attack_dodge_cost, options->rounding); /* dodge paid with HP */
        } else if (evaded) {
            /* missed */
        } else if (threshold_dodge || nullified) {
            /* no damage */
        } else if (sign_nullified) {
            after->timer[other][defender]++; /* Astraeus: the threshold drops 15% per nullification */
        } else if (b->status_dodge_max && after->own_dodges_used[other][defender] < (uint32_t)b->status_dodge_max &&
                   has_status(after, side, attacker)) {
            after->own_dodges_used[other][defender]++; /* Mist Spirit */
        } else if (alternate) {
            if (b->alternate_dodge_heal_fraction > 0 &&
                heal(after, other, defender, damage * b->alternate_dodge_heal_fraction, options->rounding)) return 1;
            if (b->dodge_extra_action) after->extra[other][defender]++; /* Gingerbread Man */
        } else if (!bypass && !after->used[other][defender] &&
                   (b->block_mode == 1 || (b->block_mode == 2 && damage < *hp_d))) {
            after->used[other][defender] = 1;
        } else if (after->shields[other][defender] && !bypass && !a->domain) {
            after->shields[other][defender]--;
        } else if (b->parry_chance > 0 && !bypass && roll(ch, b->parry_chance)) { /* Vajra Short Sword */
            if (after->hp[side][attacker] > 0)
                after->hp[side][attacker] -= rounded(fmin(damage, after->hp[side][attacker]) * b->parry_reflect_fraction, options->rounding);
        } else if (after->granted_dodges[other][defender]) {
            /* User-observed True Prophet/Shu (Raze and mirror): the next hit is dodged,
             * lethal or not; Shu still recovers from the dodged damage, capped at max HP. */
            after->granted_dodges[other][defender]--;
            if (b->heal_damage_taken_fraction > 0 && heal(after, other, defender,
                fmin(*hp_d, damage) * b->heal_damage_taken_fraction, options->rounding)) return 1;
        } else if (damage >= *hp_d && after->own_dodges_used[other][defender] < (uint32_t)b->own_lethal_dodges) {
            after->own_dodges_used[other][defender]++;
            if (b->lethal_dodge_heal_max_hp_fraction > 0 && heal(after, other, defender,
                max_d * b->lethal_dodge_heal_max_hp_fraction, options->rounding)) return 1;
            after->attack[other][defender] = rounded(after->attack[other][defender] * b->lethal_dodge_attack_multiplier,
                                                     options->stat_rounding);
            if (b->lethal_dodge_swap && behind < after->size[other] && after->hp[other][after->order[other][behind]] > 0) {
                /* Dilophosaurus: it steps back and the next ally comes in (entering normally). */
                int returning;
                after->order[other][after->active[other]] = after->order[other][behind];
                after->order[other][behind] = defender;
                cancel_entry(after, other);
                intercepted = (int)other;
                returning = (int)after->entered[other][front(after, other)];
                if (enter(battle, options, after, other, ch)) return 1;
                if (!returning && queued_entry(base_battle, after, other)) {
                    if (after->entry_count >= 2) return 1;
                    after->entry_queue[after->entry_count++] = other;
                }
            }
        } else if (interceptor >= 0) {
            /* Provisional: the displaced card swaps into the interceptor's slot. */
            unsigned p = (unsigned)interceptor;
            double factor = b->piccolo_factor_override > 0 ? b->piccolo_factor_override : battle->fighters[other][p].intercept_lethal_multiplier;
            after->order[other][after->active[other]] = p;
            after->order[other][behind] = defender;
            after->hp[other][p] = rounded(after->hp[other][p] * factor, options->stat_rounding);
            after->max_hp[other][p] = rounded(after->max_hp[other][p] * factor, options->stat_rounding);
            after->attack[other][p] = rounded(after->attack[other][p] * factor, options->stat_rounding);
            if (!isfinite(after->hp[other][p]) || !isfinite(after->max_hp[other][p]) || !isfinite(after->attack[other][p])) return 1;
            after->entered[other][p] = 1;
            cancel_entry(after, other);
            intercepted = (int)other;
            /* User-observed: Chronus steals from a swapping-in Piccolo (after its 2x). */
            if (arrive(battle, options, after, other)) return 1;
        } else {
            double old_hp = *hp_d;
            if (damage >= old_hp && after->survivals_used[other][defender] >= (unsigned)b->lethal_survivals &&
                !after->revived[other][defender] && roll(ch, b->chance_survival_probability)) {
                after->revived[other][defender] = 1;
                *hp_d = b->chance_survival_hp_fraction > 0 ?
                    fmax(1, rounded(max_d * b->chance_survival_hp_fraction, options->stat_rounding)) : 1;
                refresh(&after->frozen[side][attacker], b->chance_survival_freeze_turns);
            } else if (damage >= old_hp && after->survivals_used[other][defender] >= (unsigned)b->lethal_survivals &&
                       !after->guarded[other][defender] && roll(ch, after->negated[other] ? 0 : b->guardian_chance)) {
                after->guarded[other][defender] = 1; /* Guardian Angel support: once per ally, 1 HP. */
                *hp_d = 1;
            } else if (damage >= old_hp && after->survivals_used[other][defender] < (unsigned)b->lethal_survivals) {
                if (b->survival_damage_max_hp_fraction > 0)
                    after->hp[side][attacker] -= rounded(max_d * b->survival_damage_max_hp_fraction, options->rounding);
                after->survivals_used[other][defender]++;
                *hp_d = b->lethal_survival_hp_fraction > 0 ?
                    fmax(1, rounded(max_d * b->lethal_survival_hp_fraction, options->stat_rounding)) : 1;
                after->attack[other][defender] = rounded(after->attack[other][defender] * b->survival_attack_multiplier,
                                                         options->stat_rounding);
            } else if (damage >= old_hp && after->survivals_used[other][defender] - (unsigned)b->lethal_survivals < (unsigned)b->death_reflects) {
                after->survivals_used[other][defender]++; /* Parallax: hit negated, the attacker dies */
                after->hp[side][attacker] = fmin(after->hp[side][attacker], 0);
            } else if (damage >= old_hp && b->awaken_on_death && !(after->triggered[other][defender] & AWAKENED)) {
                after->triggered[other][defender] |= AWAKENED; /* Ultimate Brawler */
                *hp_d = after->max_hp[other][defender];
                scale(after, other, defender, b->transform_stat_multiplier, b->transform_stat_multiplier, options->stat_rounding);
                if (b->transform_form) after->ability[other][defender] = POOL_CODE + b->transform_form - 1;
            } else if (damage >= old_hp && b->zero_hp_survival_turns && !(after->triggered[other][defender] & 2)) {
                /* Zombie Dragon: survives at 0 HP (shown as 1), untouchable, and dies when its doom ends. */
                after->triggered[other][defender] |= 2;
                *hp_d = 1;
                after->undying[other][defender] = (uint32_t)b->zero_hp_survival_turns;
            } else {
                unsigned q;
                int redirect = -1;
                for (q = after->active[other] + 1; q < after->size[other]; ++q) {
                    unsigned i = after->order[other][q];
                    if (after->hp[other][i] > 0 && battle->fighters[other][i].ally_redirect_fraction > 0) { redirect = (int)i; break; }
                }
                if (redirect >= 0 && damage > 0) { /* Bei Fang Xuan Wu */
                    double part = rounded(damage * battle->fighters[other][redirect].ally_redirect_fraction, options->rounding);
                    damage -= part;
                    deck_side[deck_hits] = other; deck_index[deck_hits] = (unsigned)redirect; deck_before[deck_hits++] = after->hp[other][redirect];
                    ability_hit(battle, after, other, (unsigned)redirect, part, options->rounding);
                }
                *hp_d -= damage;
                if (a->overkill_carry && *hp_d < 0) { /* Tricerotops */
                    unsigned list[CE_MAX_FIGHTERS];
                    if (waiting(after, other, list)) {
                        unsigned nxt = after->order[other][list[0]];
                        deck_side[deck_hits] = other; deck_index[deck_hits] = nxt; deck_before[deck_hits++] = after->hp[other][nxt];
                        ability_hit(battle, after, other, nxt, -*hp_d, options->rounding);
                    }
                }
            }
            dealt = fmax(0, old_hp - *hp_d);
            if (!isfinite(*hp_d)) return 1;
            if (dealt > 0) {
                double actual_loss = fmin(old_hp, dealt);
                double lifesteal = a->heal_damage_dealt_fraction + a->lost_hp_lifesteal * actor_lost;
                if (lifesteal != 0 && heal(after, side, attacker, actual_loss * lifesteal, options->rounding)) return 1;
                if (b->heal_damage_taken_fraction > 0 && heal(after, other, defender,
                    actual_loss * b->heal_damage_taken_fraction, options->rounding)) return 1;
                if (b->damaged_toy_stat_gain > 0) { /* awakened Nutcracker: every living awakened Toy, itself included */
                    unsigned toy;
                    for (toy = 0; toy < base_battle->counts[other]; ++toy)
                        if (base_battle->fighters[other][toy].awakened_toy && after->hp[other][toy] > 0)
                            scale(after, other, toy, 1 + b->damaged_toy_stat_gain, 1 + b->damaged_toy_stat_gain, options->stat_rounding);
                }
                if (hit_effects(options, after, side, attacker, other, defender, a, b, actual_loss)) return 1;
                if (after->link[other][defender] && after->hp[other][after->link[other][defender] - 1] > 0) { /* Fate Seamstress */
                    unsigned partner = after->link[other][defender] - 1;
                    deck_side[deck_hits] = other; deck_index[deck_hits] = partner; deck_before[deck_hits++] = after->hp[other][partner];
                    ability_hit(battle, after, other, partner, actual_loss, options->rounding);
                }
                if (a->splash_damage_fraction > 0)
                    bench_hits(battle, options, after, other, actual_loss * a->splash_damage_fraction, 0, ch, 0, 1, NULL, 0);
                if (*hp_d > 0) {
                    refresh(&after->frostbite[other][defender], a->hit_frostbite_turns);
                    if (a->hit_freeze_turns && roll(ch, a->hit_freeze_chance))
                        refresh(&after->frozen[other][defender], a->hit_freeze_turns);
                    if (a->hit_stun_chance > 0 && roll(ch, a->hit_stun_chance)) refresh(&after->slowed[other][defender], 1); /* Staff */
                    if (a->hit_burn_chance > 0 && roll(ch, a->hit_burn_chance)) refresh(&after->burn[other][defender], 2); /* Flame Wizard */
                    if (a->ratio_kill && roll(ch, fmin(1, actual_loss / old_hp))) *hp_d = 0;
                }
                if (b->dodge_growth > 0) after->hits_taken[other][defender]++;
                if (*hp_d > 0) {
                    int r = options->stat_rounding;
                    if (b->damaged_attack_loss_fraction > 0)
                        after->attack[other][defender] = fmax(0, rounded(after->attack[other][defender] -
                                                                         actual_loss * b->damaged_attack_loss_fraction, r));
                    if (b->stored_damage_to_next_attack_fraction > 0)
                        after->stored[other][defender] = rounded(after->stored[other][defender] +
                                                                 actual_loss * b->stored_damage_to_next_attack_fraction, r);
                    if (b->damage_taken_to_party_attack_fraction > 0) {
                        unsigned position;
                        for (position = after->active[other] + 1; position < after->size[other]; ++position) {
                            unsigned ally = after->order[other][position];
                            if (after->hp[other][ally] > 0)
                                after->attack[other][ally] = rounded(after->attack[other][ally] +
                                                                     actual_loss * b->damage_taken_to_party_attack_fraction, r);
                        }
                    }
                    if (b->low_hp_transform_threshold > 0 && !(after->triggered[other][defender] & TRANSFORMED) &&
                        *hp_d < after->max_hp[other][defender] * b->low_hp_transform_threshold) { /* Sun Wukong, Naga */
                        after->triggered[other][defender] |= TRANSFORMED;
                        scale(after, other, defender, b->transform_stat_multiplier, b->transform_stat_multiplier, r);
                        if (b->transform_heal) *hp_d = after->max_hp[other][defender];
                        if (b->transform_form) after->ability[other][defender] = POOL_CODE + b->transform_form - 1;
                    }
                    if (b->fade_threshold > 0 && !after->faded[other][defender] && *hp_d < after->max_hp[other][defender] * b->fade_threshold) {
                        after->faded[other][defender] = 1;
                        if (b->fade_stat_multiplier != 1) scale(after, other, defender, 1 / b->fade_stat_multiplier, 1 / b->fade_stat_multiplier, r);
                    }
                }
            }
        }
    }
    if (!skipped && is_entry && battle->fighters[side][attacker].entry_hit_all_enemies) {
        /* User: the entry attack hits the whole team for 25%; waiting cards can dodge and die for
         * real (the game's display drops one card from the front per death). */
        const ce_fighter *a = &battle->fighters[side][attacker];
        bench_hits(battle, options, after, other, after->attack[side][attacker] * a->outgoing_multiplier * a->entry_hit_multiplier * kraken *
                   field_multiplier(battle, after), 1, ch, 0, 1, NULL, 0);
    }
    if (!skipped && is_entry && battle->fighters[side][attacker].entry_hit_next_card) {
        /* Ra (IMG_0325): the next enemy card takes the entry hit; its on-death ability does not trigger. */
        const ce_fighter *a = &battle->fighters[side][attacker];
        bench_hits(battle, options, after, other, after->attack[side][attacker] * a->outgoing_multiplier * a->entry_hit_multiplier *
                   field_multiplier(battle, after), 1, ch, 1, 0, NULL, 0);
    }
    if (!skipped && is_entry && battle->fighters[side][attacker].entry_hit_also_next) { /* War Scythe: the next enemy too */
        const ce_fighter *a = &battle->fighters[side][attacker];
        bench_hits(battle, options, after, other, after->attack[side][attacker] * a->outgoing_multiplier * a->entry_hit_multiplier *
                   field_multiplier(battle, after), 1, ch, 1, 0, NULL, 0);
    }
    if (!skipped && !before->counter_pending && !is_entry && battle->fighters[side][attacker].fossil_deck_hits && hit_index == total_hits) {
        /* Cyberdon: its attack also hits fossils + 1 waiting enemies (at most fossil_deck_hits). */
        const ce_fighter *a = &battle->fighters[side][attacker];
        unsigned limit = after->fossils[side] + 1 < (unsigned)a->fossil_deck_hits ? after->fossils[side] + 1 : (unsigned)a->fossil_deck_hits;
        bench_hits(battle, options, after, other, after->attack[side][attacker] * a->outgoing_multiplier * field_multiplier(battle, after),
                   1, ch, limit, 1, NULL, 0);
    }
    if (!skipped && (chosen_count || deck_target >= 0)) { /* Chaos's extra targets; Brachiosaurus's waiting target */
        const ce_fighter *a = &battle->fighters[side][attacker];
        if (deck_target >= 0) chosen[chosen_count++] = (unsigned)deck_target;
        bench_hits(battle, options, after, other, after->attack[side][attacker] * a->outgoing_multiplier * field_multiplier(battle, after),
                   1, ch, 0, 1, chosen, chosen_count);
    }
    if (!skipped && !before->counter_pending && !is_entry && hit_index == 1) {
        unsigned q;
        for (q = after->active[side] + 1; q < after->size[side]; ++q) {
            unsigned ally = after->order[side][q];
            const ce_fighter *card = &battle->fighters[side][ally];
            if (card->rake_fraction > 0 && after->hp[side][ally] > 0 && after->hp[other][defender] > 0) /* The Rake */
                ability_hit(battle, after, other, defender, rounded(after->attack[side][ally] * card->rake_fraction, options->rounding),
                            options->rounding);
        }
    }
    {
        unsigned k;
        for (k = 0; k < deck_hits; ++k) {
            unsigned ds = deck_side[k], di = deck_index[k];
            if (after->hp[ds][di] <= 0 && deck_before[k] > 0 && position_of(after, ds, di) > after->active[ds])
                on_death(battle, options, after, ds, di, ch, 1);
        }
    }
    {
        const ce_fighter *b = &battle->fighters[other][defender];
        if (!skipped && !before->counter_pending && !is_entry && hit_index == 1 && after->debuffed[other][defender] < (uint32_t)b->debuff_attacker_count) {
            after->debuffed[other][defender]++;
            after->attack[side][attacker] = rounded(after->attack[side][attacker] * b->debuff_attacker_multiplier, options->stat_rounding);
        }
    }
    target_died = after->hp[other][defender] <= 0 && after->hp[other][defender] < enemy_hp_start;
    if (!skipped && !before->counter_pending && !is_entry && (remaining == 1 || target_died)) {
        double fraction = battle->fighters[side][attacker].after_attack_heal_max_hp_fraction;
        if (fraction > 0 && heal(after, side, attacker,
            after->max_hp[side][attacker] * fraction, options->rounding)) return 1;
    }
    /* Provisional: a displaced actor's action ends, as it would on death. */
    displaced_actor = intercepted == (int)normal_side;
    if (intercepted >= 0) after->pair_actions = 0;
    after->hits_remaining = entry_context ? 0 : before->counter_pending ? before->hits_remaining : remaining - 1;
    if (!before->counter_pending && !entry_context && battle->fighters[side][attacker].followup_on_critical && !critical &&
        hit_index == total_hits - 1)
        after->hits_remaining = 0; /* Durante: the follow-up only comes after a critical strike. */
    after->counter_pending = 0;
    after->counter_from_entry = 0;
    /* User-observed Dancer rule: a KO ends the action, without retargeting. */
    target_died = after->hp[other][defender] <= 0 && after->hp[other][defender] < enemy_hp_start;
    if (target_died) blast(battle, options, after, other, defender, side, attacker, ch);
    attacker_died = after->hp[side][attacker] <= 0 && after->hp[side][attacker] < actor_hp_start; /* retaliation etc. */
    if (attacker_died) {
        blast(battle, options, after, side, attacker, other, defender, ch);
        target_died = after->hp[other][defender] <= 0 && after->hp[other][defender] < enemy_hp_start;
    }
    if (target_died || displaced_actor || attacker_died || skipped) after->hits_remaining = 0;
    else if (!before->counter_pending && dealt > 0 &&
             (battle->fighters[other][defender].counter_on_damage || battle->fighters[other][defender].counter_chance > 0) &&
             after->fossils[other] >= (uint32_t)battle->fighters[other][defender].counter_min_fossils &&
             !after->frozen[other][defender] &&
             (battle->fighters[other][defender].counter_on_damage || roll(ch, battle->fighters[other][defender].counter_chance))) {
        after->counter_pending = other + 1;
        after->counter_from_entry = is_entry;
        return 0;
    }
    if (after->hits_remaining) {
        return 0;
    }
    if (target_died) {
        const ce_fighter *k = &battle->fighters[side][attacker];
        if (!attacker_died) {
            int r = options->stat_rounding;
            if (k->kill_steal_ability && !(after->triggered[side][attacker] & 16)) {
                after->triggered[side][attacker] |= 16; /* Mother of Beasts: its first victim's ability */
                stole = 1;
                after->ability[side][attacker] = ability_code(after, other, defender);
            }
            if (k->kill_steal_fraction > 0) {
                double gain_hp = after->max_hp[other][defender] * k->kill_steal_fraction;
                after->hp[side][attacker] = rounded(after->hp[side][attacker] + gain_hp, r);
                after->max_hp[side][attacker] = rounded(after->max_hp[side][attacker] + gain_hp, r);
                after->attack[side][attacker] = rounded(after->attack[side][attacker] +
                    after->attack[other][defender] * k->kill_steal_fraction, r);
            }
            if (k->kill_hp_multiplier != 1 || k->kill_attack_multiplier != 1)
                scale(after, side, attacker, k->kill_hp_multiplier, k->kill_attack_multiplier, r);
            if (k->heal_on_kill) after->hp[side][attacker] = after->max_hp[side][attacker];
            if (k->kill_heal_max_hp_fraction > 0 && heal(after, side, attacker,
                after->max_hp[side][attacker] * k->kill_heal_max_hp_fraction, options->rounding)) return 1;
            if (k->kill_ally_stat_multiplier != 1) scale_allies(battle, after, side, k->kill_ally_stat_multiplier, r);
            if (k->kill_stat_add_base > 0) { /* awakened cultivators: +50% of base stats per kill */
                double gain_hp = rounded(k->hp * k->kill_stat_add_base, r);
                after->hp[side][attacker] += gain_hp;
                after->max_hp[side][attacker] += gain_hp;
                after->attack[side][attacker] = rounded(after->attack[side][attacker] + k->attack * k->kill_stat_add_base, r);
            }
            if (k->kill_summon_form) /* Dr. Frankenstein */
                summon(battle, after, side, after->max_hp[side][attacker], after->attack[side][attacker], POOL_CODE + k->kill_summon_form - 1, -1);
            if (battle->fighters[other][defender].curse_killer) after->curse[side][attacker] = CURSE_PERMANENT; /* The Hanged Man */
            if (k->kill_extends_undying && after->undying[side][attacker]) after->undying[side][attacker] += 2; /* Walking Dead */
            if (!isfinite(after->hp[side][attacker]) || !isfinite(after->attack[side][attacker])) return 1;
        }
        after->active[other]++;
        on_death(battle, options, after, other, defender, ch, 0);
        cancel_entry(after, other);
        changed[other] = 1;
    }
    if (attacker_died) {
        after->active[side]++;
        on_death(battle, options, after, side, attacker, ch, 0);
        cancel_entry(after, side);
        changed[side] = 1;
    }
    turn_end = !entry_context && !displaced_actor;
    first_action = !before->turn_actions_remaining;
    /* User: Mother of Beasts uses a stolen ability at once (Priest's extra action right after the kill). */
    nf = stole && normal_side == side ? &ability_view(base_battle, after, &stolen_view, failed)->fighters[normal_side][normal_slot]
                                      : &battle->fighters[normal_side][normal_slot];
    if (turn_end) after->triggered[normal_side][normal_slot] &= ~SHIFTED;
    if (turn_end && counts_turns(nf) && first_action)
        after->turns[normal_side][normal_slot]++;
    turn_index = (int)after->turns[normal_side][normal_slot] - 1;
    if (turn_end) {
        unsigned lifetime = (unsigned)(*nf).lifetime_actions;
        double end_multiplier = (*nf).action_end_attack_multiplier;
        double end_heal = (*nf).action_end_heal_max_hp_fraction;
        if (end_heal > 0 && heal(after, normal_side, normal_slot,
            after->max_hp[normal_side][normal_slot] * end_heal, options->rounding)) return 1;
        if (after->hp[normal_side][normal_slot] > 0 && end_multiplier != 1) {
            double scaled = after->attack[normal_side][normal_slot] * end_multiplier;
            if (!isfinite(scaled)) return 1;
            after->attack[normal_side][normal_slot] = rounded(scaled, options->stat_rounding);
        }
        if (after->hp[normal_side][normal_slot] > 0) {
            const ce_fighter *f = nf;
            int r = options->stat_rounding;
            if (f->action_end_stat_multiplier != 1 && (!f->action_end_stat_turns || turn_index < f->action_end_stat_turns))
                scale(after, normal_side, normal_slot, f->action_end_stat_multiplier, f->action_end_stat_multiplier, r);
            if (f->periodic_stat_period && first_action && (turn_index + 1) % f->periodic_stat_period == 0)
                scale(after, normal_side, normal_slot, f->periodic_stat_multiplier, f->periodic_stat_multiplier, r);
            if (f->action_end_attack_add_base > 0)
                after->attack[normal_side][normal_slot] = rounded(after->attack[normal_side][normal_slot] +
                                                                  f->attack * f->action_end_attack_add_base, r);
            if (f->stat_gamble && !skipped) { /* "after attacking": a skipped turn does not count */
                /* Gambler (user): ATK x one of -10%, +15%, +30%, equally likely (provisional). */
                static const double factors[] = {0.9, 1.15, 1.3};
                double factor = factors[2];
                unsigned g;
                for (g = 0; g < 2; ++g) if (roll(ch, 1.0 / (3 - g))) { factor = factors[g]; break; }
                scale(after, normal_side, normal_slot, 1, factor, r);
            }
            if (!skipped) refresh(&after->slowed[normal_side][normal_slot], f->action_end_self_slow_turns);
            if (f->block_max_hp_fraction > 0 && !skipped) {
                double block = after->max_hp[normal_side][normal_slot] * f->block_max_hp_fraction;
                after->block_pool[normal_side][normal_slot] = fmin(after->block_pool[normal_side][normal_slot] + block, block * f->block_max);
            }
            if (!isfinite(after->hp[normal_side][normal_slot]) || !isfinite(after->attack[normal_side][normal_slot])) return 1;
        }
        if (after->hp[normal_side][normal_slot] > 0) {
            const ce_fighter *f = nf;
            if (f->action_end_ally_stat_multiplier != 1)
                scale_allies(battle, after, normal_side, f->action_end_ally_stat_multiplier, options->stat_rounding);
            if (f->action_end_incoming_multiplier != 1) after->incoming_scale[normal_side][normal_slot] *= f->action_end_incoming_multiplier;
            if (f->action_end_self_hp_loss_fraction > 0)
                after->hp[normal_side][normal_slot] -= rounded(after->hp[normal_side][normal_slot] * f->action_end_self_hp_loss_fraction,
                                                               options->rounding);
        }
        if (after->hp[normal_side][normal_slot] > 0 && (*nf).action_end_hp_multiplier != 1)
            scale(after, normal_side, normal_slot, (*nf).action_end_hp_multiplier, 1,
                  options->stat_rounding);
        if (after->hp[normal_side][normal_slot] > 0) {
            const ce_fighter *f = nf;
            int r = options->stat_rounding;
            unsigned q;
            if (f->growth_add_base > 0 && turn_index < f->growth_turns) { /* cultivators: +30% of base stats for 2 turns */
                double growth = (1 + f->growth_add_base * (turn_index + 1)) / (1 + f->growth_add_base * turn_index);
                scale(after, normal_side, normal_slot, growth, growth, r);
            }
            if (after->triggered[normal_side][normal_slot] & SANTA) scale(after, normal_side, normal_slot, SANTA_DECAY, SANTA_DECAY, r);
            if (f->ally_class_stat_multiplier != 1) /* Longmu's Safeguarding */
                for (q = after->active[normal_side] + 1; q < after->size[normal_side]; ++q) {
                    unsigned ally = after->order[normal_side][q];
                    if (after->hp[normal_side][ally] > 0 && (battle->fighters[normal_side][ally].class_mask & f->ally_class_mask))
                        scale(after, normal_side, ally, f->ally_class_stat_multiplier, f->ally_class_stat_multiplier, r);
                }
            if (after->timer[normal_side][normal_slot] && f->entry_global_steal > 0 && !--after->timer[normal_side][normal_slot]) {
                /* Eonus: the stolen stats decay */
                after->hp[normal_side][normal_slot] = fmax(1, after->hp[normal_side][normal_slot] - after->bonus_hp[normal_side][normal_slot]);
                after->max_hp[normal_side][normal_slot] -= after->bonus_hp[normal_side][normal_slot];
                after->attack[normal_side][normal_slot] -= after->bonus_attack[normal_side][normal_slot];
                after->bonus_hp[normal_side][normal_slot] = after->bonus_attack[normal_side][normal_slot] = 0;
            }
        }
        {
            unsigned o = 1 - normal_side, q;
            if (after->active[o] < after->size[o]) {
                unsigned pw = front(after, o);
                double drain = battle->fighters[o][pw].enemy_turn_end_attack_drain;
                if (drain > 0 && after->hp[o][pw] > 0 && after->hp[normal_side][normal_slot] > 0) { /* Poison Witch */
                    double take = rounded(after->attack[normal_side][normal_slot] * drain, options->stat_rounding);
                    after->attack[normal_side][normal_slot] -= take;
                    if (heal(after, o, pw, take, options->rounding)) return 1;
                }
            }
            for (q = 0; q < after->size[normal_side]; ++q) {
                unsigned c = after->order[normal_side][q];
                const ce_fighter *card = &battle->fighters[normal_side][c];
                if (card->doctor_period && after->hp[normal_side][c] > 0) { /* Divine Doctor */
                    after->timer[normal_side][c]++;
                    if (after->timer[normal_side][c] % (uint32_t)card->doctor_period == 0) {
                        unsigned w;
                        for (w = 0; w < after->size[normal_side]; ++w) {
                            unsigned ally = after->order[normal_side][w];
                            if (after->hp[normal_side][ally] > 0)
                                after->hp[normal_side][ally] = fmax(after->hp[normal_side][ally], after->max_hp[normal_side][ally]);
                        }
                    }
                }
            }
        }
        if (lifetime && !changed[normal_side]) {
            after->completed_actions[normal_side][normal_slot]++;
            if (after->completed_actions[normal_side][normal_slot] >= lifetime) {
                after->hp[normal_side][normal_slot] = 0;
                after->active[normal_side]++;
                on_death(battle, options, after, normal_side, normal_slot, ch, 0);
                cancel_entry(after, normal_side);
                changed[normal_side] = 1;
            }
        }
    }
    if (turn_end && (after->triggered[normal_side][normal_slot] & SAM_BOOST) && !skipped && !changed[normal_side] &&
        after->hp[normal_side][normal_slot] > 0) { /* Uncle Sam's boost: the card dies after attacking */
        after->hp[normal_side][normal_slot] = 0;
        after->active[normal_side]++;
        on_death(battle, options, after, normal_side, normal_slot, ch, 0);
        cancel_entry(after, normal_side);
        changed[normal_side] = 1;
    }
    if (turn_end && !changed[normal_side] && after->hp[normal_side][normal_slot] > 0) {
        const ce_fighter *f = nf;
        unsigned list[CE_MAX_FIGHTERS], n, q;
        if (!skipped && f->dutchman_swaps && after->timer[normal_side][normal_slot] < (uint32_t)f->dutchman_swaps &&
            !(after->triggered[normal_side][normal_slot] & DUTCH_AWAY)) {
            n = waiting(after, normal_side, list);
            if (n) { /* Flying Dutchman (user): swaps with a random ally, at most twice */
                unsigned p = list[pick(ch, n)], ally = after->order[normal_side][p];
                after->order[normal_side][after->active[normal_side]] = ally;
                after->order[normal_side][p] = normal_slot;
                after->timer[normal_side][normal_slot]++;
                after->triggered[normal_side][normal_slot] |= DUTCH_AWAY;
                after->triggered[normal_side][ally] |= DUTCH_ALLY;
                changed[normal_side] = 1;
            }
        } else if (after->triggered[normal_side][normal_slot] & DUTCH_ALLY) {
            after->triggered[normal_side][normal_slot] &= ~DUTCH_ALLY;
            for (q = after->active[normal_side] + 1; q < after->size[normal_side]; ++q) {
                unsigned dutch = after->order[normal_side][q];
                if ((after->triggered[normal_side][dutch] & DUTCH_AWAY) && after->hp[normal_side][dutch] > 0) {
                    after->triggered[normal_side][dutch] &= ~DUTCH_AWAY;
                    after->order[normal_side][after->active[normal_side]] = dutch;
                    after->order[normal_side][q] = normal_slot;
                    changed[normal_side] = 1;
                    break;
                }
            }
        }
    }
    if (turn_end && !skipped && (*nf).swap_enemies && after->hp[normal_side][normal_slot] > 0) {
        unsigned o = 1 - normal_side, living[CE_MAX_FIGHTERS], n = 0, p;
        for (p = after->active[o]; p < after->size[o]; ++p) if (after->hp[o][after->order[o][p]] > 0) living[n++] = p;
        if (n >= 2) { /* Marionette */
            unsigned i = pick(ch, n), j = pick(ch, n - 1), pi, pj, t;
            double m = (*nf).swap_stat_multiplier;
            j += j >= i;
            pi = living[i]; pj = living[j];
            t = after->order[o][pi]; after->order[o][pi] = after->order[o][pj]; after->order[o][pj] = t;
            scale(after, o, after->order[o][pi], m, m, options->stat_rounding);
            scale(after, o, after->order[o][pj], m, m, options->stat_rounding);
            if (after->active[o] == pi || after->active[o] == pj) changed[o] = 1;
        }
    }
    if (relocated) changed[side] = 1; /* Buddha moved to the back */
    {
        unsigned allowed = before->turn_actions_remaining ? before->turn_actions_remaining :
            after->extra[normal_side][normal_slot] + allowance(nf, turn_index, (int)after->faded[normal_side][normal_slot],
                      after->hp[normal_side][normal_slot] < after->max_hp[normal_side][normal_slot] *
                      (*nf).low_hp_actions_threshold);
        kill_extra = changed[other] && side == normal_side && !entry_context && !before->counter_pending && !attacker_died &&
            battle->fighters[side][attacker].kill_extra_action && !displaced_actor;
        turn_ends = !entry_context && !kill_extra && (changed[normal_side] || displaced_actor || allowed - 1 == 0);
    }
    if (turn_ends) {
        unsigned k;
        cleanse(battle, after);
        for (k = 0; k < 2; ++k) if (after->suppressed[k]) after->suppressed[k]--; /* Fuxi wears off */
        for (k = 0; k < 2; ++k) {
            /* Waiting cards: Loch Ness growth and Zombie Dragon's countdown. */
            unsigned ts = k == 0 ? normal_side : 1 - normal_side, q, n = after->size[ts], snapshot[CE_MAX_FIGHTERS];
            for (q = 0; q < n; ++q) snapshot[q] = after->order[ts][q];
            for (q = after->active[ts] + 1; q < n; ++q) {
                unsigned index = snapshot[q];
                const ce_fighter *card = &battle->fighters[ts][index];
                double before_ticks;
                if (after->hp[ts][index] <= 0) continue;
                if (ts == normal_side && card->bench_growth_multiplier != 1) {
                    /* Loch Ness (IMG_0303: x1.1 per allied turn while waiting), capped relative to base stats. */
                    double grow = card->bench_growth_multiplier;
                    double hp_factor = card->bench_growth_cap > 0 ? fmin(grow, card->bench_growth_cap * card->hp / after->max_hp[ts][index]) : grow;
                    double attack_factor = card->bench_growth_cap > 0 && after->attack[ts][index] != 0 ?
                        fmin(grow, card->bench_growth_cap * card->attack / after->attack[ts][index]) : grow;
                    if (hp_factor > 1 || attack_factor > 1)
                        scale(after, ts, index, fmax(1, hp_factor), fmax(1, attack_factor), options->stat_rounding);
                }
                before_ticks = after->hp[ts][index];
                if (after->undying[ts][index] && !--after->undying[ts][index]) after->hp[ts][index] = 0;
                if (after->death_timer[ts][index] && !--after->death_timer[ts][index]) after->hp[ts][index] = 0;
                if (after->hp[ts][index] <= 0 && before_ticks > 0) on_death(battle, options, after, ts, index, ch, 1);
            }
        }
        for (k = 0; k < 2; ++k) {
            unsigned ts = k == 0 ? normal_side : 1 - normal_side, slot;
            double hp_before_ticks;
            if (after->active[ts] >= after->size[ts]) continue;
            slot = front(after, ts);
            hp_before_ticks = after->hp[ts][slot];
            if (after->burn[ts][slot]) {
                after->burn[ts][slot]--;
                ability_hit(battle, after, ts, slot, rounded(after->max_hp[ts][slot] * BURN_MAX_HP_FRACTION, options->rounding), options->rounding);
            }
            if (after->bleed[ts][slot]) {
                after->bleed[ts][slot]--;
                ability_hit(battle, after, ts, slot, rounded(after->max_hp[ts][slot] * BLEED_MAX_HP_FRACTION, options->rounding), options->rounding);
            }
            if (after->frozen[ts][slot]) after->frozen[ts][slot]--;
            if (after->confused[ts][slot] && after->confused[ts][slot] < CONFUSION_UNTIL_SELF_HIT) after->confused[ts][slot]--;
            if (after->frostbite[ts][slot]) {
                after->frostbite[ts][slot]--;
                if (roll(ch, FROSTBITE_CHANCE)) {
                    ability_hit(battle, after, ts, slot, rounded(after->max_hp[ts][slot] * FROSTBITE_MAX_HP_FRACTION, options->rounding),
                                options->rounding);
                    after->slowed[ts][slot]++; /* Video 0290: the card loses its next turn. */
                }
            }
            if (after->undying[ts][slot] && !--after->undying[ts][slot])
                after->hp[ts][slot] = 0; /* Zombie Dragon: its turns at 0 HP are over */
            if (after->death_timer[ts][slot] && !--after->death_timer[ts][slot])
                after->hp[ts][slot] = 0; /* Velociraptor's timer ends */
            if (after->hp[ts][slot] <= 0 && after->hp[ts][slot] < hp_before_ticks) {
                after->active[ts]++;
                on_death(battle, options, after, ts, slot, ch, 0);
                cancel_entry(after, ts);
                changed[ts] = 1;
            }
        }
    }
    if (changed[0] || changed[1] || intercepted >= 0) after->pair_actions = 0;
    else if (!entry_context) {
        after->pair_actions++;
        if (options->repeat_cycles && after->pair_actions >= 2 * options->repeat_cycles) {
            unsigned k;
            for (k = 0; k < 2; ++k) { /* Sequential, as in the reference: side 0's death effects see side 1 alive. */
                unsigned dead = front(after, k);
                after->hp[k][dead] = 0;
                after->active[k]++;
                on_death(battle, options, after, k, dead, ch, 0);
            }
            changed[0] = changed[1] = 1;
            after->pair_actions = 0;
        }
    }
    for (team = 0; team < 2; ++team)
        /* Waiting cards killed in the deck were already handled; the next living card plays. */
        while (after->active[team] > before->active[team] && after->active[team] < after->size[team] &&
               after->hp[team][front(after, team)] <= 0) after->active[team]++;
    /* The hand-off uses the acting card as it acted: an ability stolen by this kill (Mother of Beasts) waits a turn. */
    ce_fighter normal_fighter = *nf;
    battle = ability_view(base_battle, after, &view_storage, failed);
    if (terminal(battle, after)) return 0;
    after->side = normal_side;
    if (!entry_context) {
        unsigned allowed = before->turn_actions_remaining ? before->turn_actions_remaining :
            after->extra[normal_side][normal_slot] + allowance(&normal_fighter, turn_index, (int)after->faded[normal_side][normal_slot],
                      after->hp[normal_side][normal_slot] < after->max_hp[normal_side][normal_slot] *
                      normal_fighter.low_hp_actions_threshold);
        if (!before->turn_actions_remaining) after->extra[normal_side][normal_slot] = 0;
        after->turn_actions_remaining = allowed - 1;
        if (kill_extra) after->turn_actions_remaining++; /* Immediately attacks again after a kill. */
        else if (changed[normal_side] || displaced_actor || !after->turn_actions_remaining) {
            after->side = 1-normal_side;
            after->turn_actions_remaining = 0;
        }
    } else if ((changed[normal_side] || displaced_actor) && after->turn_actions_remaining) {
        after->side = 1-normal_side;
        after->turn_actions_remaining = 0;
    }
    for (team = 0; team < 2; ++team) {
        if (changed[team]) {
            int returning = after->active[team] < after->size[team] && after->entered[team][front(after, team)];
            if (enter(battle, options, after, team, ch)) return 1;
            if (!returning && queued_entry(base_battle, after, team)) {
                if (after->entry_count >= 2) return 1;
                after->entry_queue[after->entry_count++] = team;
            }
        }
    }
    return 0;
}

static uint64_t hash_state(const state *s) {
    const unsigned char *bytes = (const unsigned char *)s;
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;
    for (i = 0; i < sizeof(*s); ++i) { hash ^= bytes[i]; hash *= UINT64_C(1099511628211); }
    return hash;
}

static int frontier_init(frontier *f, size_t capacity) {
    size_t slots = 1;
    memset(f, 0, sizeof(*f));
    if (capacity > SIZE_MAX / sizeof(weighted_state) || capacity > SIZE_MAX / 4) return 1;
    while (slots < capacity * 2) {
        if (slots > SIZE_MAX / 2) return 1;
        slots *= 2;
    }
    if (slots > SIZE_MAX / sizeof(size_t)) return 1;
    f->rows = (weighted_state *)calloc(capacity, sizeof(weighted_state));
    f->slots = (size_t *)calloc(slots, sizeof(size_t));
    if (!f->rows || !f->slots) { free(f->rows); free(f->slots); memset(f, 0, sizeof(*f)); return 1; }
    f->capacity = capacity;
    f->slot_count = slots;
    return 0;
}

static void frontier_free(frontier *f) { free(f->rows); free(f->slots); }
static void frontier_clear(frontier *f) { f->size = 0; memset(f->slots, 0, f->slot_count * sizeof(size_t)); }

/* Double capacity and rehash; chance rolls can give a parent more than three children. */
static int frontier_grow(frontier *f) {
    size_t capacity = f->capacity * 2, slots = f->slot_count * 2, i;
    weighted_state *rows;
    size_t *table;
    if (capacity > SIZE_MAX / sizeof(weighted_state) || slots > SIZE_MAX / sizeof(size_t)) return 2;
    rows = (weighted_state *)realloc(f->rows, capacity * sizeof(weighted_state));
    if (!rows) return 2;
    f->rows = rows;
    table = (size_t *)calloc(slots, sizeof(size_t));
    if (!table) return 2;
    free(f->slots);
    f->slots = table;
    f->slot_count = slots;
    f->capacity = capacity;
    for (i = 0; i < f->size; ++i) {
        size_t slot = (size_t)hash_state(&f->rows[i].value) & (slots - 1);
        while (table[slot]) slot = (slot + 1) & (slots - 1);
        table[slot] = i + 1;
    }
    return 0;
}

static int frontier_insert(frontier *f, const state *s, double p, ce_result *out) {
    size_t slot = (size_t)hash_state(s) & (f->slot_count - 1);
    while (f->slots[slot]) {
        weighted_state *existing = &f->rows[f->slots[slot] - 1];
        if (!memcmp(&existing->value, s, sizeof(*s))) {
            existing->probability += p;
            out->merged_states++;
            return 0;
        }
        slot = (slot + 1) & (f->slot_count - 1);
    }
    if (f->size >= f->capacity) {
        if (frontier_grow(f)) return 2;
        return frontier_insert(f, s, p, out);
    }
    memcpy(&f->rows[f->size].value, s, sizeof(*s));
    f->rows[f->size].probability = p;
    f->rows[f->size].insertion_order = f->size;
    f->slots[slot] = ++f->size;
    return 0;
}

static int compare_probability(const void *left, const void *right) {
    const weighted_state *a = (const weighted_state *)left;
    const weighted_state *b = (const weighted_state *)right;
    if (a->probability > b->probability) return -1;
    if (a->probability < b->probability) return 1;
    return a->insertion_order < b->insertion_order ? -1 : a->insertion_order > b->insertion_order;
}

/* User: bypassing defenses also defeats chance dodges. */
static int next_hit_is_followup(const ce_battle *battle, const state *s) {
    unsigned side = hit_actor(s);
    const ce_fighter *hitter = &battle->fighters[side][front(s, side)];
    return hitter->followup_multiplier > 0 && !s->counter_pending && !s->entry_count && s->hits_remaining == 1;
}

static double hit_critical_probability(const ce_battle *base_battle, const state *s) {
    ce_battle view_storage;
    const ce_battle *battle = ability_view(base_battle, s, &view_storage, NULL);
    unsigned side = hit_actor(s);
    return next_hit_is_followup(battle, s) ? 0 : battle->fighters[side][front(s, side)].critical_probability * (1 - fail_chance(base_battle, s, 1 - side));
}

static double hit_dodge_probability(const ce_battle *base_battle, const state *s) {
    ce_battle view_storage;
    const ce_battle *battle = ability_view(base_battle, s, &view_storage, NULL);
    unsigned side = hit_actor(s), other = 1 - side;
    const ce_fighter *hitter = &battle->fighters[side][front(s, side)];
    if (hitter->bypass_defenses || (hitter->entry_bypass_defenses && s->entry_count && !s->counter_pending) ||
        (hitter->followup_bypass && next_hit_is_followup(battle, s))) return 0;
    {
        const ce_fighter *d = &battle->fighters[other][front(s, other)];
        double p = d->dodge_probability;
        int li = hitter->domain ? -1 : luminant(battle, s, other, front(s, other));
        if (d->dodge_growth > 0) p = fmin(d->dodge_cap, p + d->dodge_growth * s->hits_taken[other][front(s, other)]);
        p *= 1 - fail_chance(base_battle, s, side); /* End Times: a chance ability can fail too */
        if (li >= 0 && s->evades[other][front(s, other)] < LUMINANT_MAX) { /* Eclipseborn Luminant */
            double q = fmax(LUMINANT_FLOOR, battle->fighters[other][li].luminant_evasion - LUMINANT_STEP * s->evades[other][front(s, other)]);
            p = 1 - (1 - p) * (1 - q);
        }
        return p;
    }
}

/* One sampled continuation from s; returns the terminal outcome or 0 if unfinished. */
static int playout(const ce_battle *battle, const ce_options *options, const state *start, uint64_t *rng,
                   uint32_t steps, int *code) {
    state current, next;
    uint32_t step;
    chance_ctx ch;
    memcpy(&current, start, sizeof(current));
    for (step = 0; step < steps; ++step) {
        double dodge = hit_dodge_probability(battle, &current);
        double critical_chance = hit_critical_probability(battle, &current);
        int dodged = dodge == 1 || (dodge > 0 && uniform_random(rng) < dodge);
        int critical = !dodged && (critical_chance == 1 || (critical_chance > 0 && uniform_random(rng) < critical_chance));
        int outcome;
        memset(&ch, 0, sizeof(ch));
        ch.rng = rng;
        *code = advance(battle, options, &current, &next, dodged, critical, &ch);
        if (*code) return 0;
        if (ch.overflow) { *code = 1; return 0; }
        outcome = terminal(battle, &next);
        if (outcome) return outcome;
        memcpy(&current, &next, sizeof(current));
    }
    return 0;
}

#define MIN_ROLLOUTS 8

/* Playouts for one step's pruned rows [first, size), as the reference's _rollout: each starts from a row drawn in
 * proportion to its mass; a fixed count, or with rollout_error e at least MIN_ROLLOUTS and at most `rollouts`,
 * stopping once M * max_k p_k(1-p_k)/n <= e^2 (p_k = (c_k+1)/(n+2), M = the pool's mass). */
static int rollout(const ce_battle *battle, const ce_options *options, const frontier *f, size_t first, ce_result *out,
                   uint32_t steps_left, uint64_t *playouts) {
    uint32_t counts[4] = {0, 0, 0, 0}, n = 0, k;
    double total = 0;
    size_t i;
    for (i = first; i < f->size; ++i) total += f->rows[i].probability;
    while (n < options->rollouts) {
        uint64_t rng = options->seed ^ (UINT64_C(0x9E3779B97F4A7C15) * (*playouts + n + 1));
        double target, acc = 0;
        const state *start = &f->rows[f->size - 1].value;
        int code = 0, outcome;
        if (!rng) rng = UINT64_C(0x9e3779b97f4a7c15);
        target = uniform_random(&rng) * total;
        for (i = first; i < f->size; ++i) {
            acc += f->rows[i].probability;
            if (target < acc) { start = &f->rows[i].value; break; }
        }
        outcome = playout(battle, options, start, &rng, steps_left, &code);
        if (code) return code;
        counts[outcome ? outcome - 1 : 3]++;
        ++n;
        if (options->rollout_error > 0 && n >= MIN_ROLLOUTS) {
            double spread = 0;
            for (k = 0; k < 4; ++k) {
                double p = (counts[k] + 1.0) / (n + 2.0);
                spread = fmax(spread, p * (1.0 - p));
            }
            if (total * spread / n <= options->rollout_error * options->rollout_error) break;
        }
    }
    *playouts += n;
    out->estimated += total;
    out->p_a += total * counts[0] / n;
    out->p_b += total * counts[1] / n;
    out->tie += total * counts[2] / n;
    out->unresolved += total * counts[3] / n;
    return 0;
}

/* Keep the most probable states (a prefix after sorting by mass); the rest, or all of them once the node budget
 * is spent, are resolved by pooled Monte Carlo playouts when enabled, in the reference's order. */
static int apply_budget(const ce_battle *battle, frontier *f, const ce_options *options, ce_result *out,
                        uint32_t steps_left, uint64_t *playouts, int spent) {
    size_t i, kept = 0;
    qsort(f->rows, f->size, sizeof(weighted_state), compare_probability);
    while (!spent && kept < f->size && f->rows[kept].probability >= options->prune_probability && kept < options->max_frontier) ++kept;
    if (kept < f->size) {
        if (!options->rollouts) {
            for (i = kept; i < f->size; ++i) out->unresolved += f->rows[i].probability;
        } else {
            int code = rollout(battle, options, f, kept, out, steps_left, playouts);
            if (code) return code;
        }
    }
    f->size = kept;
    /* Hash slots are no longer used: the next iteration only reads these rows. */
    return 0;
}

static int sample(const ce_battle *battle, const ce_options *options, ce_result *out, uint64_t *rng) {
    state current, next;
    uint32_t step;
    chance_ctx ch;
    memset(&ch, 0, sizeof(ch));
    ch.rng = rng;
    if (initialize(battle, options, &current, &ch) || ch.overflow) return 1;
    for (step = 0; step < options->max_steps; ++step) {
        double dodge = hit_dodge_probability(battle, &current);
        double critical_chance = hit_critical_probability(battle, &current);
        int dodged = dodge == 1 || (dodge > 0 && uniform_random(rng) < dodge);
        int critical = !dodged && (critical_chance == 1 || (critical_chance > 0 && uniform_random(rng) < critical_chance));
        int outcome, code;
        out->expanded_states++;
        memset(&ch, 0, sizeof(ch));
        ch.rng = rng;
        code = advance(battle, options, &current, &next, dodged, critical, &ch);
        if (code) return code;
        if (ch.overflow) return 1;
        outcome = terminal(battle, &next);
#ifdef CE_TRACE
        {
            unsigned ts, ti;
            fprintf(stderr, "step %u active %u %u side %u |", step, next.active[0], next.active[1], next.side);
            for (ts = 0; ts < 2; ++ts) for (ti = 0; ti < battle->counts[ts]; ++ti)
                fprintf(stderr, " %g/%g/%g:%d", next.hp[ts][ti], next.max_hp[ts][ti], next.attack[ts][ti], next.ability[ts][ti]);
            fprintf(stderr, "\n");
        }
#endif
        if (outcome) { add_terminal(out, outcome, 1); return 0; }
        memcpy(&current, &next, sizeof(current));
    }
    out->unresolved = 1;
    return 0;
}

/* Enumerate chance rolls depth first, unrolled outcome (0) before rolled (1),
 * matching reference._expand. parent == NULL expands the initial state. */
/* Sampled (collapsed) chance rolls in branch mode, as the reference's _expand `collapse`. */
typedef struct {
    uint64_t rng;
} collapse_ctx;
#define COLLAPSE_SEED UINT64_C(0xC0FFEE1234567890)

static int expand(const ce_battle *battle, const ce_options *options, const state *parent, int dodged, int critical,
                  uint16_t *forced, int forced_count, double weight, frontier *next, ce_result *out,
                  collapse_ctx *cc, int drawn) {
    chance_ctx ch;
    state child;
    int code;
    memset(&ch, 0, sizeof(ch));
    ch.forced = forced;
    ch.forced_count = forced_count;
    code = parent ? advance(battle, options, parent, &child, dodged, critical, &ch) : initialize(battle, options, &child, &ch);
    if (code) return code;
    if (ch.overflow) return 1;
    if (ch.count > forced_count) {
        double p = ch.p[forced_count];
        double smallest = p < 0 ? weight / (unsigned)-p : weight * fmin(p, 1 - p);
        if (cc != NULL && smallest < options->sample_below) {
            /* User: simulate the small branches; one draw keeps the whole weight (unbiased). */
            double u = uniform_random(&cc->rng);
            if (p < 0) {
                unsigned n = (unsigned)-p, k = (unsigned)(u * n);
                forced[forced_count] = (uint16_t)(k < n - 1 ? k : n - 1);
            } else {
                forced[forced_count] = u < p;
            }
            if (!drawn) out->estimated += weight;
            return expand(battle, options, parent, dodged, critical, forced, forced_count + 1, weight, next, out, cc, 1);
        }
        if (p < 0) { /* an n-way pick, outcome 0 first */
            unsigned n = (unsigned)-p, k;
            for (k = 0; k < n; ++k) {
                forced[forced_count] = (uint16_t)k;
                code = expand(battle, options, parent, dodged, critical, forced, forced_count + 1, weight / n, next, out, cc, drawn);
                if (code) return code;
            }
            return 0;
        }
        forced[forced_count] = 0;
        code = expand(battle, options, parent, dodged, critical, forced, forced_count + 1, weight * (1 - p), next, out, cc, drawn);
        if (code) return code;
        forced[forced_count] = 1;
        return expand(battle, options, parent, dodged, critical, forced, forced_count + 1, weight * p, next, out, cc, drawn);
    }
    if (parent) {
        int outcome = terminal(battle, &child);
        if (outcome) { add_terminal(out, outcome, weight); return 0; }
    }
    return frontier_insert(next, &child, weight, out);
}

static int branch(const ce_battle *battle, const ce_options *options, ce_result *out) {
    frontier current, next;
    uint16_t forced[MAX_CHANCE_ROLLS];
    uint32_t step;
    uint64_t playouts = 0;
    size_t i;
    int code;
    /* Rows grow on demand (states are large with 16-card teams). */
    size_t initial = options->max_frontier < 1024 ? (size_t)options->max_frontier * 3 : 3072;
    if (frontier_init(&current, initial)) return 2;
    if (frontier_init(&next, initial)) { frontier_free(&current); return 2; }
    collapse_ctx collapse, *cc = options->sample_below > 0 ? &collapse : NULL;
    collapse.rng = options->seed ^ COLLAPSE_SEED;
    if (!collapse.rng) collapse.rng = UINT64_C(0x9E3779B97F4A7C15);
    code = expand(battle, options, NULL, 0, 0, forced, 0, 1, &current, out, cc, 0);
    if (code) { frontier_free(&current); frontier_free(&next); return code; }
    for (step = 0; step < options->max_steps && current.size; ++step) {
        frontier_clear(&next);
        for (i = 0; i < current.size; ++i) {
            weighted_state *parent = &current.rows[i];
            double dodge = hit_dodge_probability(battle, &parent->value);
            double critical_chance = hit_critical_probability(battle, &parent->value);
            double probabilities[3] = {dodge, (1-dodge)*critical_chance, (1-dodge)*(1-critical_chance)};
            int choice;
            out->expanded_states++;
            /* Deterministic order: dodge, critical hit, ordinary hit. */
            for (choice = 0; choice < 3; ++choice) {
                double p = parent->probability * probabilities[choice];
                if (p == 0) continue;
                code = expand(battle, options, &parent->value, choice == 0, choice == 1, forced, 0, p, &next, out, cc, 0);
                if (code) { frontier_free(&current); frontier_free(&next); return code; }
            }
        }
        code = apply_budget(battle, &next, options, out, options->max_steps - step - 1, &playouts,
                            options->node_budget && out->expanded_states >= options->node_budget);
        if (code) { frontier_free(&current); frontier_free(&next); return code; }
        { frontier temp = current; current = next; next = temp; }
    }
    for (i = 0; i < current.size; ++i) out->unresolved += current.rows[i].probability;
    out->estimated = fmin(out->estimated, 1); /* a sampled path counts at every later step; 0 = exact */
    frontier_free(&current);
    frontier_free(&next);
    return 0;
}

static int batch5_valid(const ce_fighter *f) {
    const double values[] = {f->fade_threshold, f->fade_outgoing_multiplier, f->fade_incoming_multiplier, f->fade_stat_multiplier, f->survival_damage_max_hp_fraction, f->entry_next_card_stat_multiplier, f->next_card_stat_multiplier, f->next_card_gift_chance, f->entry_enemy_stat_subtract_fraction, f->entry_steal_attack_fraction, f->entry_steal_fraction, f->turn_start_hp_steal_fraction, f->turn_start_attack_chance, f->turn_start_attack_multiplier, f->turn_start_full_heal_chance, f->damaged_attack_loss_fraction, f->target_current_hp_damage_fraction, f->action_end_self_hp_loss_fraction, f->debuff_attacker_multiplier, f->first_hit_per_enemy_multiplier, f->field_damage_multiplier, f->action_end_ally_stat_multiplier, f->kill_ally_stat_multiplier, f->normal_attack_dodge_cost, f->action_end_incoming_multiplier, f->card_rarity, f->border_rarity, f->border_advantage_multiplier, f->border_advantage_incoming_multiplier, f->lower_rarity_incoming_multiplier, f->younger_target_multiplier, f->class_damage_multiplier, f->class_incoming_multiplier, f->stored_damage_to_next_attack_fraction, f->damage_taken_to_party_attack_fraction};
    const double positive[] = {f->fade_outgoing_multiplier, f->fade_incoming_multiplier, f->fade_stat_multiplier, f->entry_next_card_stat_multiplier, f->next_card_stat_multiplier, f->turn_start_attack_multiplier, f->debuff_attacker_multiplier, f->field_damage_multiplier, f->action_end_ally_stat_multiplier, f->kill_ally_stat_multiplier, f->action_end_incoming_multiplier};
    const double chances[] = {f->next_card_gift_chance, f->turn_start_attack_chance, f->turn_start_full_heal_chance};
    const int32_t counts[] = {f->alternate_rest, f->periodic_block_period, f->fade_extra_actions, f->kill_extra_action, f->never_attacks, f->debuff_attacker_count, f->next_card_freeze_turns};
    const int32_t masks[] = {f->class_mask, f->pack_index, f->bonus_vs_class_mask, f->guard_vs_class_mask};
    if (!isfinite(f->recoil_damage_fraction) || f->recoil_damage_fraction < 0) return 0;
    {
        const double b7[] = {f->death_damage_max_hp_fraction, f->death_damage_chance, f->death_ally_stat_multiplier, f->periodic_attack_multiplier,
                             f->periodic_attack_heal_max_hp_fraction, f->followup_multiplier, f->frozen_target_multiplier, f->low_hp_actions_threshold};
        const int32_t b7i[] = {f->periodic_attack_period, f->periodic_attack_hits, f->periodic_enemy_slow_turns, f->followup_bypass,
                               f->followup_on_critical, f->encounter_freeze_turns, f->encounter_confusion_turns, f->low_hp_extra_actions};
        size_t j;
        for (j = 0; j < sizeof(b7) / sizeof(b7[0]); ++j) if (!isfinite(b7[j]) || b7[j] < 0) return 0;
        for (j = 0; j < sizeof(b7i) / sizeof(b7i[0]); ++j) if (b7i[j] < 0 || b7i[j] > 16) return 0;
        const int32_t b8[] = {f->entry_poison_turns, f->entry_poison_all, f->hit_poison_turns, f->attacked_poison_turns, f->lethal_dodge_swap};
        for (j = 0; j < sizeof(b8) / sizeof(b8[0]); ++j) if (b8[j] < 0 || b8[j] > 16) return 0;
        if (!isfinite(f->poison_fraction) || f->poison_fraction < 0) return 0;
        {
            const double b9[] = {f->hit_weakness_multiplier, f->low_hp_heal_fraction, f->low_hp_trigger_threshold, f->poison_target_max_hp_fraction,
                                 f->delayed_strike_multiplier, f->bench_growth_multiplier, f->bench_growth_cap, f->block_max_hp_fraction};
            const int32_t b9i[] = {f->zero_hp_survival_turns, f->encounter_doom_turns, f->delayed_strike_turns, f->entry_blind,
                                   f->low_hp_stun_turns, f->stat_gamble, f->block_max, f->poison_permanent,
                                   f->action_end_self_slow_turns, f->dinosaur, f->death_fossils, f->counter_min_fossils,
                                   f->fossil_death_timer, f->fossil_deck_hits, f->entry_self_slow_turns, f->entry_disable_enemy,
                                   f->hit_disable, f->cancels_all_abilities, f->entry_suppress_enemy_turns, f->steal_first_attacked,
                                   f->kill_steal_ability, f->copy_fallen_ally, f->copies_enemy_in_play, f->entry_hit_next_card};
            size_t m;
            for (m = 0; m < sizeof(b9) / sizeof(b9[0]); ++m) if (!isfinite(b9[m]) || b9[m] < 0) return 0;
            for (m = 0; m < sizeof(b9i) / sizeof(b9i[0]); ++m) if (b9i[m] < 0 || b9i[m] > 16) return 0;
            if (f->disable_enemy_class_mask < 0 || f->disable_enemy_class_mask > 127 || f->entry_disable_enemy_class_mask < 0 ||
                f->entry_disable_enemy_class_mask > 127 || !isfinite(f->turn_start_steal_fraction) || f->turn_start_steal_fraction < 0 ||
                !isfinite(f->hit_attack_drain_fraction) || f->hit_attack_drain_fraction < 0 ||
                !isfinite(f->turn_start_stat_growth) || f->turn_start_stat_growth < 0 ||
                !isfinite(f->turn_start_incoming_reduction) || f->turn_start_incoming_reduction < 0 ||
                !isfinite(f->entry_steal_attack_chance) || f->entry_steal_attack_chance < 0 || f->entry_steal_attack_chance > 1 ||
                !isfinite(f->incoming_reduction_cap) || f->incoming_reduction_cap < 0 || f->incoming_reduction_cap >= 1 ||
                (f->turn_start_incoming_reduction > 0 && f->incoming_reduction_cap <= 0)) return 0;
            if (f->bench_growth_multiplier <= 0 || !isfinite(f->entry_fossil_attack_multiplier) || f->entry_fossil_attack_multiplier < 0) return 0;
        }
        if (f->death_damage_chance > 1 || f->death_ally_stat_multiplier <= 0 || f->encounter_class_mask < 0 || f->encounter_class_mask > 127) return 0;
    }
    if (!isfinite(f->splash_damage_fraction) || f->splash_damage_fraction < 0 || f->entry_hit_all_enemies < 0 || f->entry_hit_all_enemies > 1 ||
        f->death_reflects < 0 || f->death_reflects > 16) return 0;
    size_t i;
    for (i = 0; i < sizeof(values) / sizeof(values[0]); ++i) if (!isfinite(values[i]) || values[i] < 0) return 0;
    for (i = 0; i < sizeof(positive) / sizeof(positive[0]); ++i) if (positive[i] <= 0) return 0;
    for (i = 0; i < sizeof(chances) / sizeof(chances[0]); ++i) if (chances[i] > 1) return 0;
    for (i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) if (counts[i] < 0 || counts[i] > 16) return 0;
    for (i = 0; i < sizeof(masks) / sizeof(masks[0]); ++i) if (masks[i] < 0 || masks[i] > 127) return 0;
    return 1;
}

static int batch4_valid(const ce_fighter *f) {
    const double values[] = {f->frostbite_incoming_multiplier, f->hit_freeze_chance, f->skip_enemy_turn_chance, f->chance_survival_probability, f->chance_survival_hp_fraction, f->gamble_chance, f->gamble_multiplier, f->gamble_fail_heal_fraction, f->dodge_attack_multiplier, f->dodge_bonus_multiplier, f->dodge_growth, f->dodge_cap, f->entry_borderless_reset_chance, f->borderless_hp, f->borderless_attack, f->guardian_chance, f->confusion_fail_attack_ratio};
    const int32_t counts[] = {f->hit_frostbite_turns, f->death_frostbite_turns, f->entry_random_debuff_turns, f->hit_freeze_turns, f->entry_confusion_turns, f->random_multiplier_max, f->ratio_kill, f->hit_chance_squared, f->chance_survival_freeze_turns};
    const double chances[] = {f->hit_freeze_chance, f->skip_enemy_turn_chance, f->chance_survival_probability, f->gamble_chance,
                              f->entry_borderless_reset_chance, f->guardian_chance, f->dodge_cap};
    size_t i;
    for (i = 0; i < sizeof(values) / sizeof(values[0]); ++i) if (!isfinite(values[i]) || values[i] < 0) return 0;
    for (i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) if (counts[i] < 0 || counts[i] > 16) return 0;
    for (i = 0; i < sizeof(chances) / sizeof(chances[0]); ++i) if (chances[i] > 1) return 0;
    return 1;
}

static int batch2_valid(const ce_fighter *f) {
    const double values[] = {f->charged_attack_multiplier, f->alternate_dodge_heal_fraction, f->lethal_survival_hp_fraction, f->lethal_dodge_heal_max_hp_fraction, f->lethal_dodge_attack_multiplier, f->first_attack_multiplier, f->hit_growth_multiplier, f->weak_target_multiplier, f->weak_target_threshold, f->execute_below_max_hp, f->execute_after_below_max_hp, f->mutual_destruction_below_max_hp, f->lost_hp_damage_fraction, f->max_hp_damage_fraction, f->target_attack_damage_fraction, f->advantage_damage_multiplier, f->disadvantage_incoming_multiplier, f->lost_hp_damage_bonus, f->lost_hp_lifesteal, f->self_turn_start_hp_loss_fraction, f->self_turn_start_max_hp_loss_fraction, f->entry_fallen_attack_fraction, f->entry_fallen_stat_bonus, f->entry_party_attack_bonus, f->low_hp_incoming_multiplier, f->low_hp_incoming_threshold, f->action_end_stat_multiplier, f->periodic_stat_multiplier, f->action_end_attack_add_base};
    const int32_t counts[] = {f->max_hp_damage_costs, f->recharge_turns, f->first_turn_actions, f->actions_growth_per_turn, f->alternate_dodge, f->restore_lowered_attack, f->own_lethal_dodges, f->bypass_defenses, f->entry_bypass_defenses, f->entry_hit_requires_fallen, f->action_end_stat_turns, f->periodic_stat_period};
    size_t i;
    for (i = 0; i < sizeof(values) / sizeof(values[0]); ++i) if (!isfinite(values[i]) || values[i] < 0) return 0;
    for (i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) if (counts[i] < 0 || counts[i] > 16) return 0;
    if (f->survival_attack_multiplier <= 0 || !isfinite(f->survival_attack_multiplier)) return 0;
    for (i = 0; i < 7; ++i) {
        const int32_t statuses[] = {f->burn_on_attacked_turns, f->bleed_on_attacked_turns, f->hit_burn_turns, f->hit_bleed_turns, f->entry_burn_turns, f->entry_freeze_turns, f->entry_slow_turns};
        if (statuses[i] < 0 || statuses[i] > 16) return 0;
    }
    return f->lethal_dodge_attack_multiplier > 0 && f->action_end_stat_multiplier > 0 && f->periodic_stat_multiplier > 0;
}

static int batch1_valid(const ce_fighter *f) {
    const double values[] = {f->action_end_hp_multiplier, f->action_start_heal_attack_fraction, f->kill_heal_max_hp_fraction, f->kill_attack_multiplier, f->kill_hp_multiplier, f->kill_steal_fraction, f->hit_attack_gain_fraction, f->hit_target_attack_reduction_fraction, f->hit_target_max_hp_reduction_fraction, f->hit_target_attack_multiplier, f->hit_steal_fraction, f->damaged_attack_multiplier, f->damaged_stat_multiplier, f->damaged_heal_max_hp_fraction, f->attacked_steal_fraction, f->thorns_attack_fraction, f->reflect_damage_fraction, f->counter_multiplier, f->entry_enemy_hp_multiplier, f->entry_enemy_stat_multiplier, f->enemy_turn_start_attack_multiplier, f->enemy_turn_start_stat_multiplier, f->enemy_turn_start_hp_loss_fraction, f->enemy_turn_start_max_hp_loss_fraction, f->next_card_stat_add_fraction, f->next_card_max_hp_add_fraction, f->hit_hp_gain_fraction};
    const double positive[] = {f->action_end_hp_multiplier, f->kill_hp_multiplier, f->damaged_stat_multiplier,
                               f->entry_enemy_hp_multiplier, f->entry_enemy_stat_multiplier, f->enemy_turn_start_stat_multiplier};
    size_t i;
    for (i = 0; i < sizeof(values) / sizeof(values[0]); ++i) if (!isfinite(values[i]) || values[i] < 0) return 0;
    for (i = 0; i < sizeof(positive) / sizeof(positive[0]); ++i) if (positive[i] <= 0) return 0;
    return 1;
}

static int batch12_valid(const ce_fighter *f) {
    const double values[] = {f->aura_incoming_multiplier, f->chaos_hit_chance, f->lost_hp_incoming_bonus, f->turn_start_max_hp_multiplier, f->turn_start_low_hp_heal_fraction, f->turn_start_low_hp_threshold, f->entry_same_card_stat_bonus, f->evade_punish_multiplier, f->revive_on_ally_death_hp, f->revive_on_ally_death_chance, f->entry_absorb_behind, f->low_hp_transform_threshold, f->transform_stat_multiplier, f->growth_add_base, f->kill_stat_add_base, f->ally_redirect_fraction, f->enemy_turn_end_attack_drain, f->ability_reflect_fraction, f->ability_reflect_cap, f->summon_copy_fraction, f->piccolo_factor_override, f->luminant_evasion, f->enemy_death_revive_fraction, f->entry_ally_stat_multiplier, f->entry_global_steal, f->composer_chance_step, f->swap_attack_chance, f->swap_attack_multiplier, f->aura_negate_chance, f->rake_fraction, f->next_card_boost_multiplier, f->confused_self_hit_stat_multiplier, f->ally_class_stat_multiplier, f->swap_stat_multiplier, f->parry_chance, f->parry_reflect_fraction, f->hit_stun_chance,
                             f->nullify_max_hp_threshold, f->nullify_threshold_step, f->hit_burn_chance, f->counter_chance, f->followup_chance,
                             f->ability_fail_chance, f->damaged_toy_stat_gain, f->entry_fallen_toy_stat_bonus};
    const double chances[] = {f->chaos_hit_chance, f->revive_on_ally_death_chance, f->swap_attack_chance, f->aura_negate_chance, f->luminant_evasion,
                              f->parry_chance, f->hit_stun_chance, f->hit_burn_chance, f->counter_chance, f->followup_chance, f->ability_fail_chance};
    const double ones[] = {f->aura_incoming_multiplier, f->turn_start_max_hp_multiplier, f->transform_stat_multiplier, f->entry_ally_stat_multiplier,
                           f->swap_attack_multiplier, f->next_card_boost_multiplier, f->confused_self_hit_stat_multiplier,
                           f->ally_class_stat_multiplier, f->swap_stat_multiplier};
    size_t i;
    for (i = 0; i < sizeof(values) / sizeof(values[0]); ++i) if (!isfinite(values[i]) || values[i] < 0) return 0;
    for (i = 0; i < sizeof(chances) / sizeof(chances[0]); ++i) if (chances[i] > 1) return 0;
    for (i = 0; i < sizeof(ones) / sizeof(ones[0]); ++i) if (ones[i] <= 0) return 0;
    if (f->card_id < 0 || f->card_id > 1023 ||
        f->spare < 0 || f->spare > 16 ||
        f->status_dodge_max < 0 || f->status_dodge_max > 16 ||
        f->random_abilities < 0 || f->random_abilities > 16 ||
        f->pool_exclude_mask < 0 || f->pool_exclude_mask > 127 ||
        f->pool_flags < 0 || f->pool_flags > 16 ||
        f->pool_self < 0 || f->pool_self > 1023 ||
        f->entry_ability_swap_random < 0 || f->entry_ability_swap_random > 16 ||
        f->domain < 0 || f->domain > 16 ||
        f->random_hits_min < 0 || f->random_hits_min > 16 ||
        f->random_hits_max < 0 || f->random_hits_max > 16 ||
        f->chaos_targets < 0 || f->chaos_targets > 16 ||
        f->form_first < 0 || f->form_first > 1023 ||
        f->form_count < 0 || f->form_count > 16 ||
        f->team_status_immunity < 0 || f->team_status_immunity > 16 ||
        f->recruit_defeated < 0 || f->recruit_defeated > 16 ||
        f->awaken_on_death < 0 || f->awaken_on_death > 16 ||
        f->transform_form < 0 || f->transform_form > 1023 ||
        f->reflect_even_turns < 0 || f->reflect_even_turns > 16 ||
        f->persistent_confusion < 0 || f->persistent_confusion > 16 ||
        f->encounter_confusion_eternal < 0 || f->encounter_confusion_eternal > 16 ||
        f->growth_turns < 0 || f->growth_turns > 16 ||
        f->awaken_turn < 0 || f->awaken_turn > 16 ||
        f->doctor_period < 0 || f->doctor_period > 16 ||
        f->summon_random < 0 || f->summon_random > 16 ||
        f->random_deck_target < 0 || f->random_deck_target > 16 ||
        f->overkill_carry < 0 || f->overkill_carry > 16 ||
        f->ability_death_immune < 0 || f->ability_death_immune > 16 ||
        f->entry_reset_enemy_stats < 0 || f->entry_reset_enemy_stats > 16 ||
        f->sack < 0 || f->sack > 16 ||
        f->entry_shuffle_enemy < 0 || f->entry_shuffle_enemy > 16 ||
        f->entry_hit_count_max < 0 || f->entry_hit_count_max > 16 ||
        f->death_curse_turns < 0 || f->death_curse_turns > 16 ||
        f->dad_card_id < 0 || f->dad_card_id > 1023 ||
        f->split_on_death < 0 || f->split_on_death > 16 ||
        f->ally_class_mask < 0 || f->ally_class_mask > 127 ||
        f->team_class_shields < 0 || f->team_class_shields > 16 ||
        f->entry_shields < 0 || f->entry_shields > 16 ||
        f->link_first_two < 0 || f->link_first_two > 16 ||
        f->revive_party_on_death < 0 || f->revive_party_on_death > 16 ||
        f->dodge_extra_action < 0 || f->dodge_extra_action > 16 ||
        f->kill_summon_form < 0 || f->kill_summon_form > 1023 ||
        f->swap_enemies < 0 || f->swap_enemies > 16 ||
        f->curse_killer < 0 || f->curse_killer > 16 ||
        f->kill_extends_undying < 0 || f->kill_extends_undying > 16 ||
        f->dutchman_swaps < 0 || f->dutchman_swaps > 16 ||
        f->perish_turns < 0 || f->perish_turns > 16 ||
        f->turn_start_random_ability < 0 || f->turn_start_random_ability > 16 ||
        f->summon_copies < 0 || f->summon_copies > 16 ||
        f->shield_period < 0 || f->shield_period > 16 ||
        f->entry_hit_also_next < 0 || f->entry_hit_also_next > 16 ||
        f->form_pair_last < 0 || f->form_pair_last > 16 ||
        f->transform_heal < 0 || f->transform_heal > 16 ||
        f->buddha < 0 || f->buddha > 16 ||
        f->enemy_entry_confusion_turns < 0 || f->enemy_entry_confusion_turns > 16 ||
        f->awakened_toy < 0 || f->awakened_toy > 16) return 0;
    return 1;
}

static int validate_fighter(const ce_fighter *f) {
            if (!isfinite(f->hp) || f->hp <= 0 || !isfinite(f->attack) || f->attack < 0 ||
                !isfinite(f->dodge_probability) || f->dodge_probability < 0 || f->dodge_probability > 1 ||
                !isfinite(f->outgoing_multiplier) || f->outgoing_multiplier < 0 ||
                !isfinite(f->incoming_multiplier) || f->incoming_multiplier < 0 ||
                !isfinite(f->entry_self_multiplier) || f->entry_self_multiplier <= 0 ||
                !isfinite(f->entry_hp_multiplier) || f->entry_hp_multiplier <= 0 ||
                !isfinite(f->entry_attack_multiplier) || f->entry_attack_multiplier < 0 ||
                !isfinite(f->damage_reduction_max_hp) || f->damage_reduction_max_hp < 0 ||
                !isfinite(f->damage_cap_max_hp) || f->damage_cap_max_hp < 0 ||
                !isfinite(f->dodge_below_max_hp) || f->dodge_below_max_hp < 0 ||
                !isfinite(f->nullify_below_attack) || f->nullify_below_attack < 0 ||
                !isfinite(f->critical_probability) || f->critical_probability < 0 || f->critical_probability > 1 ||
                !isfinite(f->critical_multiplier) || f->critical_multiplier < 1 ||
                !isfinite(f->entry_hit_multiplier) || f->entry_hit_multiplier < 0 ||
                !isfinite(f->action_end_attack_multiplier) || f->action_end_attack_multiplier < 0 ||
                !isfinite(f->action_start_heal_max_hp_fraction) || f->action_start_heal_max_hp_fraction < 0 ||
                !isfinite(f->action_end_heal_max_hp_fraction) || f->action_end_heal_max_hp_fraction < 0 ||
                !isfinite(f->heal_damage_dealt_fraction) || f->heal_damage_dealt_fraction < 0 ||
                !isfinite(f->heal_damage_taken_fraction) || f->heal_damage_taken_fraction < 0 ||
                !isfinite(f->after_attack_heal_max_hp_fraction) || f->after_attack_heal_max_hp_fraction < 0 ||
                !isfinite(f->intercept_lethal_multiplier) || f->intercept_lethal_multiplier < 0 || !batch1_valid(f) || !batch2_valid(f) || !batch4_valid(f) || !batch5_valid(f) ||
                f->heal_on_kill < 0 || f->heal_on_kill > 1 ||
                f->counter_on_damage < 0 || f->counter_on_damage > 1 ||
                f->lethal_survivals < 0 || f->lethal_survivals > 16 ||
                f->invincible < 0 || f->invincible > 1 || f->lifetime_actions < 0 ||
                f->actions_per_turn < 1 || f->actions_per_turn > 16 ||
                f->next_card_dodges < 0 || f->next_card_dodges > 16 ||
                !isfinite(f->entry_enemy_attack_multiplier) || f->entry_enemy_attack_multiplier < 0 ||
                !isfinite(f->enemy_entry_steal_fraction) || f->enemy_entry_steal_fraction < 0 || f->enemy_entry_steal_fraction >= 1 ||
                f->block_mode < 0 || f->block_mode > 2 ||
                f->attacks_per_action < 1 || f->attacks_per_action > CE_MAX_ATTACKS_PER_ACTION || !batch12_valid(f)) return 0;
    return 1;
}

static int validate(const ce_battle *battle) {
    unsigned side, i;
    if (battle->first_side > 1 || (battle->pool_count && !battle->pool) || battle->pool_count > 1023 - POOL_CODE) return 0;
    if (offsetof(ce_fighter, block_mode) % sizeof(double)) return 0;
    for (side = 0; side < 2; ++side) {
        if (!battle->counts[side] || battle->counts[side] > CE_MAX_FIGHTERS) return 0;
        if (!battle->lineup[side] || battle->lineup[side] > battle->counts[side]) return 0;
        for (i = 0; i < battle->counts[side]; ++i)
            if (!validate_fighter(&battle->fighters[side][i])) return 0;
    }
    return 1;
}

int ce_simulate_batch(const ce_battle *battles, size_t count, const ce_options *options,
                      ce_result *results, char *error, size_t error_cap) {
    size_t i;
    uint64_t rng;
    if (error && error_cap) error[0] = '\0';
    if (!options || (count && (!battles || !results))) return fail(error, error_cap, 1, "Null input pointer");
    if ((options->mode != CE_SAMPLE && options->mode != CE_BRANCH) || options->rounding < 0 || options->rounding > 3 ||
        options->stat_rounding < 0 || options->stat_rounding > 3 ||
        !options->max_steps || options->max_steps > INT32_MAX || !options->max_frontier || options->max_frontier > INT32_MAX ||
        !options->repeat_cycles || options->repeat_cycles > UINT32_MAX / 2 || options->rollouts > INT32_MAX ||
        !isfinite(options->prune_probability) || options->prune_probability < 0 || options->prune_probability > 1 ||
        !isfinite(options->rollout_error) || options->rollout_error < 0 || options->rollout_error >= 1 || options->node_budget > INT32_MAX ||
        !isfinite(options->sample_below) || options->sample_below < 0 || options->sample_below > 1)
        return fail(error, error_cap, 1, "Invalid simulation options");
    {
        const ce_fighter *checked = NULL;
        for (i = 0; i < count; ++i) {
            size_t k;
            if (!validate(&battles[i])) return fail(error, error_cap, 1, "Invalid battle/fighter fields");
            if (battles[i].pool_count && battles[i].pool != checked) {
                for (k = 0; k < battles[i].pool_count; ++k)
                    if (!validate_fighter(&battles[i].pool[k])) return fail(error, error_cap, 1, "Invalid ability pool");
                checked = battles[i].pool;
            }
        }
    }
    for (i = 0; i < count; ++i) {
        int code;
        rng = options->seed + (uint64_t)i;
        if (!rng) rng = UINT64_C(0x9e3779b97f4a7c15);
        memset(&results[i], 0, sizeof(results[i]));
        code = options->mode == CE_SAMPLE ? sample(&battles[i], options, &results[i], &rng) : branch(&battles[i], options, &results[i]);
        if (code) return fail(error, error_cap, code, code == 2 ? "Frontier allocation failed" :
                              "Unsupported lethal entry effect or non-finite battle arithmetic");
        if (!isfinite(results[i].p_a + results[i].p_b + results[i].tie + results[i].unresolved) ||
            fabs(results[i].p_a + results[i].p_b + results[i].tie + results[i].unresolved - 1) > 1e-10)
            return fail(error, error_cap, 1, "Probability mass accounting failed");
    }
    return 0;
}
