#ifndef CARD_ENGINE_SIM_H
#define CARD_ENGINE_SIM_H

#include <stddef.h>
#include <stdint.h>

/* Provisional, deliberately limited battle semantics. No card-specific logic. */
#define CE_MAX_FIGHTERS 16 /* lineup plus spare slots; ability codes use side * 16 + index */
#define CE_MAX_ATTACKS_PER_ACTION 16
#define CE_SAMPLE 0
#define CE_BRANCH 1
#define CE_FLOAT 0
#define CE_CEIL 1
#define CE_FLOOR 2
#define CE_NEAREST_HALF_UP 3

typedef struct {
    double hp;
    double attack;
    double dodge_probability;
    double outgoing_multiplier;
    double incoming_multiplier;
    double entry_self_multiplier;
    double entry_enemy_attack_multiplier;
    double enemy_entry_steal_fraction;
    double entry_hp_multiplier;
    double entry_attack_multiplier;
    double damage_reduction_max_hp;
    double damage_cap_max_hp; /* 0 disables cap. */
    double dodge_below_max_hp;
    double nullify_below_attack;
    double critical_probability;
    double critical_multiplier;
    double entry_hit_multiplier; /* 0 disables the one-time entry attack. */
    double action_end_attack_multiplier;
    double action_start_heal_max_hp_fraction;
    double action_end_heal_max_hp_fraction;
    double heal_damage_dealt_fraction;
    double heal_damage_taken_fraction;
    double after_attack_heal_max_hp_fraction;
    double intercept_lethal_multiplier; /* 0 disables; swap in from directly behind. */
    /* Batch-1 extrapolated primitives; see reference.py for semantics. */
    double action_end_hp_multiplier;
    double action_start_heal_attack_fraction;
    double kill_heal_max_hp_fraction;
    double kill_attack_multiplier;
    double kill_hp_multiplier;
    double kill_steal_fraction;
    double hit_attack_gain_fraction;
    double hit_target_attack_reduction_fraction;
    double hit_target_max_hp_reduction_fraction;
    double hit_target_attack_multiplier;
    double hit_steal_fraction;
    double damaged_attack_multiplier;
    double damaged_stat_multiplier;
    double damaged_heal_max_hp_fraction;
    double attacked_steal_fraction;
    double thorns_attack_fraction;
    double reflect_damage_fraction;
    double counter_multiplier;
    double entry_enemy_hp_multiplier;
    double entry_enemy_stat_multiplier;
    double enemy_turn_start_attack_multiplier;
    double enemy_turn_start_stat_multiplier;
    double enemy_turn_start_hp_loss_fraction;
    double enemy_turn_start_max_hp_loss_fraction;
    double next_card_stat_add_fraction;
    double next_card_max_hp_add_fraction;
    double hit_hp_gain_fraction;
    /* Batch-2 extrapolated primitives. */
    double charged_attack_multiplier;
    double alternate_dodge_heal_fraction;
    double lethal_survival_hp_fraction;
    double lethal_dodge_heal_max_hp_fraction;
    double lethal_dodge_attack_multiplier;
    double first_attack_multiplier;
    double hit_growth_multiplier;
    double weak_target_multiplier;
    double weak_target_threshold;
    double execute_below_max_hp;
    double execute_after_below_max_hp;
    double mutual_destruction_below_max_hp;
    double lost_hp_damage_fraction;
    double max_hp_damage_fraction;
    double target_attack_damage_fraction;
    double advantage_damage_multiplier;
    double disadvantage_incoming_multiplier;
    double lost_hp_damage_bonus;
    double lost_hp_lifesteal;
    double self_turn_start_hp_loss_fraction;
    double self_turn_start_max_hp_loss_fraction;
    double entry_fallen_attack_fraction;
    double entry_fallen_stat_bonus;
    double entry_party_attack_bonus;
    double low_hp_incoming_multiplier;
    double low_hp_incoming_threshold;
    double action_end_stat_multiplier;
    double periodic_stat_multiplier;
    double action_end_attack_add_base;
    double survival_attack_multiplier;
    double frostbite_incoming_multiplier;
    double hit_freeze_chance;
    double skip_enemy_turn_chance;
    double chance_survival_probability;
    double chance_survival_hp_fraction;
    double gamble_chance;
    double gamble_multiplier;
    double gamble_fail_heal_fraction;
    double dodge_attack_multiplier;
    double dodge_bonus_multiplier;
    double dodge_growth;
    double dodge_cap;
    double entry_borderless_reset_chance;
    double borderless_hp;
    double borderless_attack;
    double guardian_chance;
    double confusion_fail_attack_ratio;
    double fade_threshold;
    double fade_outgoing_multiplier;
    double fade_incoming_multiplier;
    double fade_stat_multiplier;
    double survival_damage_max_hp_fraction;
    double entry_next_card_stat_multiplier;
    double next_card_stat_multiplier;
    double next_card_gift_chance;
    double entry_enemy_stat_subtract_fraction;
    double entry_steal_attack_fraction;
    double entry_steal_fraction;
    double turn_start_hp_steal_fraction;
    double turn_start_attack_chance;
    double turn_start_attack_multiplier;
    double turn_start_full_heal_chance;
    double damaged_attack_loss_fraction;
    double target_current_hp_damage_fraction;
    double action_end_self_hp_loss_fraction;
    double debuff_attacker_multiplier;
    double first_hit_per_enemy_multiplier;
    double field_damage_multiplier;
    double action_end_ally_stat_multiplier;
    double kill_ally_stat_multiplier;
    double normal_attack_dodge_cost;
    double action_end_incoming_multiplier;
    double card_rarity;
    double border_rarity;
    double border_advantage_multiplier;
    double border_advantage_incoming_multiplier;
    double lower_rarity_incoming_multiplier;
    double younger_target_multiplier;
    double class_damage_multiplier;
    double class_incoming_multiplier;
    double stored_damage_to_next_attack_fraction;
    double damage_taken_to_party_attack_fraction;
    double splash_damage_fraction;
    double recoil_damage_fraction;
    double death_damage_max_hp_fraction;
    double death_damage_chance;
    double death_ally_stat_multiplier;
    double periodic_attack_multiplier;
    double periodic_attack_heal_max_hp_fraction;
    double followup_multiplier;
    double frozen_target_multiplier;
    double low_hp_actions_threshold;
    double poison_fraction; /* poison tick = poisoner ATK x this, ignoring reductions */
    double hit_weakness_multiplier;
    double low_hp_heal_fraction;
    double low_hp_trigger_threshold;
    double poison_target_max_hp_fraction;
    double delayed_strike_multiplier;
    double bench_growth_multiplier;
    double bench_growth_cap;
    double block_max_hp_fraction;
    double entry_fossil_attack_multiplier; /* Tyrannodon: ATK x this per fossil on entry */
    double turn_start_steal_fraction; /* Mother of Beasts */
    double hit_attack_drain_fraction; /* Vampire Lord: ATK moved from the target, at most all of it */
    double turn_start_stat_growth; /* Limitless Rivals: fixed gain of this share of pre-growth stats per turn */
    double turn_start_incoming_reduction; /* Limitless Rivals: damage reduction grows by this fixed step per turn */
    double entry_steal_attack_chance; /* Witch: chance of the entry ATK steal (0: always) */
    double incoming_reduction_cap; /* Limitless Rivals: at most this damage reduction */
    /* Batch 12 (see reference.py). */
    double aura_incoming_multiplier;
    double chaos_hit_chance;
    double lost_hp_incoming_bonus;
    double turn_start_max_hp_multiplier;
    double turn_start_low_hp_heal_fraction;
    double turn_start_low_hp_threshold;
    double entry_same_card_stat_bonus;
    double evade_punish_multiplier;
    double revive_on_ally_death_hp;
    double revive_on_ally_death_chance;
    double entry_absorb_behind;
    double low_hp_transform_threshold;
    double transform_stat_multiplier;
    double growth_add_base;
    double kill_stat_add_base;
    double ally_redirect_fraction;
    double enemy_turn_end_attack_drain;
    double ability_reflect_fraction;
    double ability_reflect_cap;
    double summon_copy_fraction;
    double piccolo_factor_override;
    double luminant_evasion;
    double enemy_death_revive_fraction;
    double entry_ally_stat_multiplier;
    double entry_global_steal;
    double composer_chance_step;
    double swap_attack_chance;
    double swap_attack_multiplier;
    double aura_negate_chance;
    double rake_fraction;
    double next_card_boost_multiplier;
    double confused_self_hit_stat_multiplier;
    double ally_class_stat_multiplier;
    double swap_stat_multiplier;
    double parry_chance; /* Vajra Short Sword */
    double parry_reflect_fraction;
    double hit_stun_chance; /* Staff of Perfect Enlightenment */
    double nullify_max_hp_threshold; /* Astraeus: nullify damage below this share of max HP... */
    double nullify_threshold_step; /* ...dropping by this per nullification */
    double hit_burn_chance; /* Flame Wizard support */
    double counter_chance; /* Berserker support */
    double followup_chance; /* Storm Spirit support */
    double ability_fail_chance; /* End Times support: an enemy ability fails (transient roll) */
    double damaged_toy_stat_gain; /* awakened Toy Nutcracker */
    double entry_fallen_toy_stat_bonus; /* awakened Toy Bear */
    int32_t block_mode; /* 0 none; 1 first hit; 2 first nonlethal hit */
    int32_t attacks_per_action; /* 1..CE_MAX_ATTACKS_PER_ACTION */
    int32_t heal_on_kill;
    int32_t counter_on_damage;
    int32_t lethal_survivals;
    int32_t invincible;
    int32_t lifetime_actions; /* 0 disables own-action expiration. */
    int32_t actions_per_turn; /* Separate complete actions before yielding. */
    int32_t next_card_dodges; /* next-hit dodges granted to the next card on death */
    int32_t max_hp_damage_costs;
    int32_t recharge_turns;
    int32_t first_turn_actions;
    int32_t actions_growth_per_turn;
    int32_t alternate_dodge;
    int32_t restore_lowered_attack;
    int32_t own_lethal_dodges;
    int32_t bypass_defenses;
    int32_t entry_bypass_defenses;
    int32_t entry_hit_requires_fallen;
    int32_t action_end_stat_turns;
    int32_t periodic_stat_period;
    int32_t burn_on_attacked_turns;
    int32_t bleed_on_attacked_turns;
    int32_t hit_burn_turns;
    int32_t hit_bleed_turns;
    int32_t entry_burn_turns;
    int32_t entry_freeze_turns;
    int32_t entry_slow_turns;
    int32_t hit_frostbite_turns;
    int32_t death_frostbite_turns;
    int32_t entry_random_debuff_turns;
    int32_t hit_freeze_turns;
    int32_t entry_confusion_turns;
    int32_t random_multiplier_max;
    int32_t ratio_kill;
    int32_t hit_chance_squared;
    int32_t chance_survival_freeze_turns;
    int32_t alternate_rest;
    int32_t periodic_block_period;
    int32_t fade_extra_actions;
    int32_t kill_extra_action;
    int32_t never_attacks;
    int32_t debuff_attacker_count;
    int32_t next_card_freeze_turns;
    int32_t class_mask;
    int32_t pack_index;
    int32_t bonus_vs_class_mask;
    int32_t guard_vs_class_mask;
    int32_t entry_hit_all_enemies;
    int32_t death_reflects; /* Parallax: lethal hit negated, attacker dies */
    int32_t periodic_attack_period;
    int32_t periodic_attack_hits;
    int32_t periodic_enemy_slow_turns;
    int32_t followup_bypass;
    int32_t followup_on_critical;
    int32_t encounter_freeze_turns;
    int32_t encounter_confusion_turns;
    int32_t encounter_class_mask;
    int32_t low_hp_extra_actions;
    int32_t entry_poison_turns;
    int32_t entry_poison_all;
    int32_t hit_poison_turns;
    int32_t attacked_poison_turns;
    int32_t lethal_dodge_swap;
    int32_t zero_hp_survival_turns;
    int32_t encounter_doom_turns;
    int32_t delayed_strike_turns;
    int32_t entry_blind;
    int32_t low_hp_stun_turns;
    int32_t stat_gamble;
    int32_t block_max;
    int32_t poison_permanent; /* poison never wears off (Dilophosaurus) */
    int32_t action_end_self_slow_turns; /* Failed Mage (user): skips its next turn after attacking */
    int32_t dinosaur; /* +1 team fossil on death */
    int32_t death_fossils;
    int32_t counter_min_fossils;
    int32_t fossil_death_timer;
    int32_t fossil_deck_hits;
    int32_t entry_self_slow_turns;
    int32_t entry_disable_enemy; /* Hell's Army */
    int32_t hit_disable; /* Set */
    int32_t cancels_all_abilities; /* Samurai */
    int32_t entry_suppress_enemy_turns; /* Fuxi */
    int32_t steal_first_attacked; /* Hecate */
    int32_t kill_steal_ability; /* Mother of Beasts */
    int32_t copy_fallen_ally; /* Hades, Legends */
    int32_t disable_enemy_class_mask; /* Black Cat */
    int32_t entry_disable_enemy_class_mask; /* Night Witch */
    int32_t copies_enemy_in_play; /* Sable */
    int32_t entry_hit_next_card; /* Ra */
    int32_t card_id;
    int32_t spare;
    int32_t status_dodge_max;
    int32_t random_abilities;
    int32_t pool_exclude_mask;
    int32_t pool_flags;
    int32_t pool_self;
    int32_t entry_ability_swap_random;
    int32_t domain;
    int32_t random_hits_min;
    int32_t random_hits_max;
    int32_t chaos_targets;
    int32_t form_first;
    int32_t form_count;
    int32_t team_status_immunity;
    int32_t recruit_defeated;
    int32_t awaken_on_death;
    int32_t transform_form;
    int32_t reflect_even_turns;
    int32_t persistent_confusion;
    int32_t encounter_confusion_eternal;
    int32_t growth_turns;
    int32_t awaken_turn;
    int32_t doctor_period;
    int32_t summon_random;
    int32_t random_deck_target;
    int32_t overkill_carry;
    int32_t ability_death_immune;
    int32_t entry_reset_enemy_stats;
    int32_t sack;
    int32_t entry_shuffle_enemy;
    int32_t entry_hit_count_max;
    int32_t death_curse_turns;
    int32_t dad_card_id;
    int32_t split_on_death;
    int32_t ally_class_mask;
    int32_t team_class_shields;
    int32_t entry_shields;
    int32_t link_first_two;
    int32_t revive_party_on_death;
    int32_t dodge_extra_action;
    int32_t kill_summon_form;
    int32_t swap_enemies;
    int32_t curse_killer;
    int32_t kill_extends_undying;
    int32_t dutchman_swaps;
    int32_t perish_turns;
    int32_t turn_start_random_ability;
    int32_t summon_copies;
    int32_t shield_period;
    int32_t entry_hit_also_next;
    int32_t form_pair_last;
    int32_t transform_heal;
    int32_t buddha;
    int32_t enemy_entry_confusion_turns; /* awakened Toy Jack-in-the-Box */
    int32_t awakened_toy; /* Magical Elf support */
} ce_fighter;

typedef struct {
    ce_fighter fighters[2][CE_MAX_FIGHTERS];
    uint32_t counts[2]; /* card definitions per side: the lineup, then spare slots for summons */
    uint32_t first_side;
    ce_fighter blanks[2][CE_MAX_FIGHTERS]; /* each card with no ability (identity fields only) */
    uint32_t lineup[2]; /* cards in play at the start */
    uint32_t pool_count;
    const ce_fighter *pool; /* abilities for random, copied or transformed cards (codes 33 + i) */
    ce_fighter defaults; /* a card with every field at its default (merging two abilities) */
} ce_battle;

typedef struct {
    int32_t mode;
    int32_t rounding;
    int32_t stat_rounding;
    uint32_t repeat_cycles; /* positive; forced simultaneous active deaths */
    uint32_t max_steps; /* hit resolutions per branch, including dodges/blocks */
    uint32_t max_frontier;
    double prune_probability;
    uint64_t seed;
    uint32_t rollouts; /* branch mode: Monte Carlo playouts per pruned state (0 = count as unresolved) */
    double rollout_error; /* > 0: rollouts is the maximum; stop once the state's variance share is small enough */
    uint32_t node_budget; /* branch mode: after this many expansions every frontier state is played out (0 = none) */
    double sample_below; /* branch mode: a roll whose smallest branch is below this is sampled, not enumerated */
} ce_options;

typedef struct {
    double p_a;
    double p_b;
    double tie;
    double unresolved;
    uint64_t expanded_states;
    uint64_t merged_states;
    double estimated; /* probability assigned by playouts rather than exact branching */
} ce_result;

/* One options struct applies to the batch. Match i uses seed+i modulo 2^64.
 * 0 = success, 1 = invalid input or arithmetic, 2 = allocation failure.
 * On failure results must not be used; error holds a human-readable reason.
 */
int ce_simulate_batch(const ce_battle *battles, size_t count,
                      const ce_options *options, ce_result *results,
                      char *error, size_t error_cap);

const char *ce_version(void);

#endif
