"""Small Python oracle for the first portable C primitive subset.

Every state contains the ordered teams, active indices, block consumption,
initiative, pending hits and repeat counter. Probabilities merge only for identical complete
states. These explicit semantics are provisional until checked against the game.
"""

from dataclasses import dataclass, field, replace
import math


MAX_ATTACKS_PER_ACTION = 16
_BATCH1_FIELDS = ('action_end_hp_multiplier', 'action_start_heal_attack_fraction', 'kill_heal_max_hp_fraction', 'kill_attack_multiplier', 'kill_hp_multiplier', 'kill_steal_fraction', 'hit_attack_gain_fraction', 'hit_target_attack_reduction_fraction', 'hit_target_max_hp_reduction_fraction', 'hit_target_attack_multiplier', 'hit_steal_fraction', 'damaged_attack_multiplier', 'damaged_stat_multiplier', 'damaged_heal_max_hp_fraction', 'attacked_steal_fraction', 'thorns_attack_fraction', 'reflect_damage_fraction', 'counter_multiplier', 'entry_enemy_hp_multiplier', 'entry_enemy_stat_multiplier', 'enemy_turn_start_attack_multiplier', 'enemy_turn_start_stat_multiplier', 'enemy_turn_start_hp_loss_fraction', 'enemy_turn_start_max_hp_loss_fraction', 'next_card_stat_add_fraction', 'next_card_max_hp_add_fraction', 'hit_hp_gain_fraction')
_BATCH2_INTS = ('max_hp_damage_costs', 'recharge_turns', 'first_turn_actions', 'actions_growth_per_turn', 'alternate_dodge', 'restore_lowered_attack', 'own_lethal_dodges', 'bypass_defenses', 'entry_bypass_defenses', 'entry_hit_requires_fallen', 'action_end_stat_turns', 'periodic_stat_period')
_BATCH2_FLOATS = ('charged_attack_multiplier', 'alternate_dodge_heal_fraction', 'lethal_survival_hp_fraction', 'lethal_dodge_heal_max_hp_fraction', 'lethal_dodge_attack_multiplier', 'first_attack_multiplier', 'hit_growth_multiplier', 'weak_target_multiplier', 'weak_target_threshold', 'execute_below_max_hp', 'execute_after_below_max_hp', 'mutual_destruction_below_max_hp', 'lost_hp_damage_fraction', 'max_hp_damage_fraction', 'target_attack_damage_fraction', 'advantage_damage_multiplier', 'disadvantage_incoming_multiplier', 'lost_hp_damage_bonus', 'lost_hp_lifesteal', 'self_turn_start_hp_loss_fraction', 'self_turn_start_max_hp_loss_fraction', 'entry_fallen_attack_fraction', 'entry_fallen_stat_bonus', 'entry_party_attack_bonus', 'low_hp_incoming_multiplier', 'low_hp_incoming_threshold', 'action_end_stat_multiplier', 'periodic_stat_multiplier', 'action_end_attack_add_base')
# Statuses (user-observed): durations tick at the end of every turn, both sides';
# re-applying refreshes a status, different statuses stack. Tick sizes are fits:
# bleed killed a Ruby 100 Men on the 6th tick (15.5-18.5%), burn about 10%.
BURN_MAX_HP_FRACTION = 0.10
BLEED_MAX_HP_FRACTION = 0.16
# User video 0290: frostbite rolls once per tick; on success 20% max HP damage and
# the card loses its next turn. User: confusion self-attacks at about 50%.
FROSTBITE_CHANCE = 0.5
FROSTBITE_MAX_HP_FRACTION = 0.20
CONFUSION_CHANCE = 0.5
MAX_CHANCE_ROLLS = 32
MAX_FIGHTERS = 16  # Lineup plus spare slots for summons; ability codes use side * 16 + index.
_BATCH4_INTS = ('hit_frostbite_turns', 'death_frostbite_turns', 'entry_random_debuff_turns', 'hit_freeze_turns', 'entry_confusion_turns', 'random_multiplier_max', 'ratio_kill', 'hit_chance_squared', 'chance_survival_freeze_turns')
_BATCH4_FLOATS = ('frostbite_incoming_multiplier', 'hit_freeze_chance', 'skip_enemy_turn_chance', 'chance_survival_probability', 'chance_survival_hp_fraction', 'gamble_chance', 'gamble_multiplier', 'gamble_fail_heal_fraction', 'dodge_attack_multiplier', 'dodge_bonus_multiplier', 'dodge_growth', 'dodge_cap', 'entry_borderless_reset_chance', 'borderless_hp', 'borderless_attack', 'guardian_chance', 'confusion_fail_attack_ratio')
_BATCH5_INTS = ('alternate_rest', 'periodic_block_period', 'fade_extra_actions', 'kill_extra_action', 'never_attacks', 'debuff_attacker_count', 'next_card_freeze_turns', 'class_mask', 'pack_index', 'bonus_vs_class_mask', 'guard_vs_class_mask')
_BATCH5_FLOATS = ('fade_threshold', 'fade_outgoing_multiplier', 'fade_incoming_multiplier', 'fade_stat_multiplier', 'survival_damage_max_hp_fraction', 'entry_next_card_stat_multiplier', 'next_card_stat_multiplier', 'next_card_gift_chance', 'entry_enemy_stat_subtract_fraction', 'entry_steal_attack_fraction', 'entry_steal_fraction', 'turn_start_hp_steal_fraction', 'turn_start_attack_chance', 'turn_start_attack_multiplier', 'turn_start_full_heal_chance', 'damaged_attack_loss_fraction', 'target_current_hp_damage_fraction', 'action_end_self_hp_loss_fraction', 'debuff_attacker_multiplier', 'first_hit_per_enemy_multiplier', 'field_damage_multiplier', 'action_end_ally_stat_multiplier', 'kill_ally_stat_multiplier', 'normal_attack_dodge_cost', 'action_end_incoming_multiplier', 'card_rarity', 'border_rarity', 'border_advantage_multiplier', 'border_advantage_incoming_multiplier', 'lower_rarity_incoming_multiplier', 'younger_target_multiplier', 'class_damage_multiplier', 'class_incoming_multiplier', 'stored_damage_to_next_attack_fraction', 'damage_taken_to_party_attack_fraction')
_BATCH5_CHANCES = ('next_card_gift_chance', 'turn_start_attack_chance', 'turn_start_full_heal_chance')
_BATCH6_INTS = ('entry_hit_all_enemies', 'death_reflects')
_BATCH6_FLOATS = ('splash_damage_fraction', 'recoil_damage_fraction')
_BATCH7_FLOATS = ('death_damage_max_hp_fraction', 'death_damage_chance', 'death_ally_stat_multiplier', 'periodic_attack_multiplier',
                  'periodic_attack_heal_max_hp_fraction', 'followup_multiplier', 'frozen_target_multiplier', 'low_hp_actions_threshold')
_BATCH9_FLOATS = ('hit_weakness_multiplier', 'low_hp_heal_fraction', 'low_hp_trigger_threshold', 'poison_target_max_hp_fraction',
                  'delayed_strike_multiplier', 'bench_growth_multiplier', 'bench_growth_cap', 'block_max_hp_fraction')
_BATCH9_INTS = ('zero_hp_survival_turns', 'encounter_doom_turns', 'delayed_strike_turns', 'entry_blind', 'low_hp_stun_turns',
                'stat_gamble', 'block_max', 'poison_permanent', 'action_end_self_slow_turns')
# Poison that lasts the whole battle (user: Dilophosaurus in IMG_0303).
POISON_PERMANENT = 1 << 20
# Gambler (user): after attacking, ATK changes by one of -10%, +15%, +30% (IMG_0303 showed x1.3);
# equally likely (provisional).
STAT_GAMBLE_FACTORS = (0.9, 1.15, 1.3)
_BATCH11_INTS = ('entry_disable_enemy', 'hit_disable', 'cancels_all_abilities', 'entry_suppress_enemy_turns', 'steal_first_attacked',
                 'kill_steal_ability', 'copy_fallen_ally', 'disable_enemy_class_mask', 'entry_disable_enemy_class_mask',
                 'copies_enemy_in_play', 'entry_hit_next_card')
# A card keeps these when its ability is removed, stolen or copied; everything else is its ability.
IDENTITY_FIELDS = ('hp', 'attack', 'borderless_hp', 'borderless_attack', 'card_rarity', 'border_rarity', 'pack_index',
                   'class_mask', 'dinosaur', 'card_id', 'spare')
ABILITY_NONE = -1  # Ability state: 0 own, -1 removed, 1 + side * 16 + index: that fighter's ability; 33 + i: pool entry i.
POOL_CODE = 33
# Batch 12: the remaining cards (user answers 2026-09-30/10-01, IMG_0304/0325/0327-0341). Provisional extrapolations.
_BATCH12_FLOATS = ('aura_incoming_multiplier', 'chaos_hit_chance', 'lost_hp_incoming_bonus', 'turn_start_max_hp_multiplier',
                   'turn_start_low_hp_heal_fraction', 'turn_start_low_hp_threshold', 'entry_same_card_stat_bonus', 'evade_punish_multiplier',
                   'revive_on_ally_death_hp', 'revive_on_ally_death_chance', 'entry_absorb_behind', 'low_hp_transform_threshold',
                   'transform_stat_multiplier', 'growth_add_base', 'kill_stat_add_base', 'ally_redirect_fraction',
                   'enemy_turn_end_attack_drain', 'ability_reflect_fraction', 'ability_reflect_cap', 'summon_copy_fraction',
                   'piccolo_factor_override', 'luminant_evasion', 'enemy_death_revive_fraction', 'entry_ally_stat_multiplier',
                   'entry_global_steal', 'composer_chance_step', 'swap_attack_chance', 'swap_attack_multiplier', 'aura_negate_chance',
                   'rake_fraction', 'next_card_boost_multiplier', 'confused_self_hit_stat_multiplier', 'ally_class_stat_multiplier',
                   'swap_stat_multiplier', 'parry_chance', 'parry_reflect_fraction', 'hit_stun_chance',
                   'nullify_max_hp_threshold', 'nullify_threshold_step', 'hit_burn_chance', 'counter_chance', 'followup_chance',
                   'ability_fail_chance', 'damaged_toy_stat_gain', 'entry_fallen_toy_stat_bonus')
_BATCH12_ONES = ('aura_incoming_multiplier', 'turn_start_max_hp_multiplier', 'transform_stat_multiplier', 'entry_ally_stat_multiplier',
                 'swap_attack_multiplier', 'next_card_boost_multiplier', 'confused_self_hit_stat_multiplier', 'ally_class_stat_multiplier',
                 'swap_stat_multiplier')
_BATCH12_INTS = ('card_id', 'spare', 'status_dodge_max', 'random_abilities', 'pool_exclude_mask', 'pool_flags', 'pool_self',
                 'entry_ability_swap_random', 'domain', 'random_hits_min', 'random_hits_max', 'chaos_targets', 'form_first', 'form_count',
                 'team_status_immunity', 'recruit_defeated', 'awaken_on_death', 'transform_form', 'reflect_even_turns',
                 'persistent_confusion', 'encounter_confusion_eternal', 'growth_turns', 'awaken_turn', 'doctor_period', 'summon_random',
                 'random_deck_target', 'overkill_carry', 'ability_death_immune', 'entry_reset_enemy_stats', 'sack', 'entry_shuffle_enemy',
                 'entry_hit_count_max', 'death_curse_turns', 'dad_card_id', 'split_on_death', 'ally_class_mask', 'team_class_shields',
                 'entry_shields', 'link_first_two', 'revive_party_on_death', 'dodge_extra_action', 'kill_summon_form', 'swap_enemies',
                 'curse_killer', 'kill_extends_undying', 'dutchman_swaps', 'perish_turns', 'turn_start_random_ability', 'summon_copies',
                 'shield_period', 'entry_hit_also_next', 'form_pair_last', 'transform_heal', 'buddha',
                 'enemy_entry_confusion_turns', 'awakened_toy')
_BATCH12_WIDE = ('card_id', 'pool_self', 'form_first', 'transform_form', 'dad_card_id', 'kill_summon_form')  # 0..1023
POOL_LIMITED, POOL_GLAMOUR, POOL_FORM = 1, 2, 4  # pool_flags; forms are never drawn at random.
CURSE_MAX_HP_FRACTION = 0.25  # The Hanged Man, Naga's curse (user): 25% max HP at the start of each turn.
CURSE_PERMANENT = 1 << 20
CONFUSION_UNTIL_SELF_HIT = 1 << 20  # Kuchisake-onna: confusion persists until the card hits itself.
CONFUSION_ETERNAL = 1 << 21  # Cthulu.
SANTA_DECAY = 0.9  # Santa Claws: boosted allies lose 10% stats at the end of each of their turns.
SPLIT_HP, SPLIT_ATTACK = 0.5, 0.6  # The Broken One: Joy and Sorrow.
PARTY_REVIVED, SPENT, SAM_BOOST, SANTA, DUTCH_ALLY, DUTCH_AWAY, TRANSFORMED, AWAKENED = (1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14,
                                                                                 1 << 15, 1 << 16, 1 << 17)
SHIFTED = 1 << 18  # Glamour has changed this turn (its new form's entry attack comes first).
_EXT_CARD = ('ability2', 'shields', 'timer', 'bonus_hp', 'bonus_attack', 'evades', 'extra', 'link', 'curse')

_BATCH10_INTS = ('dinosaur', 'death_fossils', 'counter_min_fossils', 'fossil_death_timer', 'fossil_deck_hits', 'entry_self_slow_turns')
# Velociraptor: death in 5 - fossil count turns, the fossil count capped at 2 (card text).
FOSSIL_DEATH_TIMER_CAP = 2
_BATCH8_INTS = ('entry_poison_turns', 'entry_poison_all', 'hit_poison_turns', 'attacked_poison_turns', 'lethal_dodge_swap')
_BATCH7_INTS = ('periodic_attack_period', 'periodic_attack_hits', 'periodic_enemy_slow_turns', 'followup_bypass', 'followup_on_critical',
                'encounter_freeze_turns', 'encounter_confusion_turns', 'encounter_class_mask', 'low_hp_extra_actions')
_BATCH3_INTS = ('burn_on_attacked_turns', 'bleed_on_attacked_turns', 'hit_burn_turns', 'hit_bleed_turns', 'entry_burn_turns', 'entry_freeze_turns', 'entry_slow_turns')
_POSITIVE_FIELDS = ('survival_attack_multiplier',) + ('fade_outgoing_multiplier', 'fade_incoming_multiplier', 'fade_stat_multiplier', 'entry_next_card_stat_multiplier', 'next_card_stat_multiplier', 'turn_start_attack_multiplier', 'debuff_attacker_multiplier', 'field_damage_multiplier', 'action_end_ally_stat_multiplier', 'kill_ally_stat_multiplier', 'action_end_incoming_multiplier') + ('lethal_dodge_attack_multiplier', 'action_end_stat_multiplier', 'periodic_stat_multiplier') + ("action_end_hp_multiplier", "kill_hp_multiplier", "damaged_stat_multiplier",
                    "entry_enemy_hp_multiplier", "entry_enemy_stat_multiplier", "enemy_turn_start_stat_multiplier",
                    "death_ally_stat_multiplier")


@dataclass(frozen=True, slots=True)
class Fighter:
    hp: float
    attack: float
    dodge_probability: float = 0.0
    outgoing_multiplier: float = 1.0
    incoming_multiplier: float = 1.0
    entry_self_multiplier: float = 1.0
    entry_enemy_attack_multiplier: float = 1.0
    enemy_entry_steal_fraction: float = 0.0
    block_mode: int = 0  # 0 none, 1 first hit, 2 first nonlethal hit
    attacks_per_action: int = 1
    entry_hp_multiplier: float = 1.0
    entry_attack_multiplier: float = 1.0
    damage_reduction_max_hp: float = 0.0
    damage_cap_max_hp: float = 0.0  # 0 disables the cap.
    dodge_below_max_hp: float = 0.0
    nullify_below_attack: float = 0.0
    critical_probability: float = 0.0
    critical_multiplier: float = 2.0
    heal_on_kill: int = 0
    counter_on_damage: int = 0
    lethal_survivals: int = 0
    entry_hit_multiplier: float = 0.0  # 0 disables the one-time entry attack.
    invincible: int = 0
    lifetime_actions: int = 0  # 0 disables expiration after own completed actions.
    actions_per_turn: int = 1
    action_end_attack_multiplier: float = 1.0
    action_start_heal_max_hp_fraction: float = 0.0
    action_end_heal_max_hp_fraction: float = 0.0
    heal_damage_dealt_fraction: float = 0.0
    heal_damage_taken_fraction: float = 0.0
    after_attack_heal_max_hp_fraction: float = 0.0
    # Piccolo: from directly behind, swap in for the active card's lethal hit,
    # block it, and multiply own stats. 0 disables.
    intercept_lethal_multiplier: float = 0.0
    # True Prophet: on death, the next card gains this many dodges. The card text
    # says "lethal", but the user saw the next hit dodged even when not lethal.
    next_card_dodges: int = 0
    # Batch-1 primitives (card-text extrapolations, not game-verified).
    # hit_*: attacker effects after a damaging hit; damaged_*/attacked_*/thorns/
    # reflect: defender reactions; enemy_turn_start_*: applied to the opposing
    # active card when its normal action starts; kill_*: after a kill.
    action_end_hp_multiplier: float = 1.0
    action_start_heal_attack_fraction: float = 0.0
    kill_heal_max_hp_fraction: float = 0.0
    kill_attack_multiplier: float = 1.0
    kill_hp_multiplier: float = 1.0
    kill_steal_fraction: float = 0.0
    hit_attack_gain_fraction: float = 0.0
    hit_target_attack_reduction_fraction: float = 0.0
    hit_target_max_hp_reduction_fraction: float = 0.0
    hit_target_attack_multiplier: float = 1.0
    hit_steal_fraction: float = 0.0
    damaged_attack_multiplier: float = 1.0
    damaged_stat_multiplier: float = 1.0
    damaged_heal_max_hp_fraction: float = 0.0
    attacked_steal_fraction: float = 0.0
    thorns_attack_fraction: float = 0.0
    reflect_damage_fraction: float = 0.0
    counter_multiplier: float = 1.0
    entry_enemy_hp_multiplier: float = 1.0
    entry_enemy_stat_multiplier: float = 1.0
    enemy_turn_start_attack_multiplier: float = 1.0
    enemy_turn_start_stat_multiplier: float = 1.0
    enemy_turn_start_hp_loss_fraction: float = 0.0
    enemy_turn_start_max_hp_loss_fraction: float = 0.0
    next_card_stat_add_fraction: float = 0.0
    next_card_max_hp_add_fraction: float = 0.0
    hit_hp_gain_fraction: float = 0.0  # User: #91's stolen HP can exceed max HP.
    # Batch-3 status sources (turn counts); freeze blocks every action including
    # counters (user: Ice Queen froze Raze's counter), slow skips own normal turns.
    burn_on_attacked_turns: int = 0
    bleed_on_attacked_turns: int = 0
    hit_burn_turns: int = 0
    hit_bleed_turns: int = 0
    entry_burn_turns: int = 0
    entry_freeze_turns: int = 0
    entry_slow_turns: int = 0
    survival_attack_multiplier: float = 1.0
    # Batch-4 chance mechanics (frostbite, procs, revives, confusion, gambles).
    # borderless_hp/attack: base stats without border (0 = use hp/attack).
    hit_frostbite_turns: int = 0
    death_frostbite_turns: int = 0
    entry_random_debuff_turns: int = 0
    hit_freeze_turns: int = 0
    entry_confusion_turns: int = 0
    random_multiplier_max: int = 0
    ratio_kill: int = 0
    hit_chance_squared: int = 0
    chance_survival_freeze_turns: int = 0
    frostbite_incoming_multiplier: float = 1.0
    hit_freeze_chance: float = 1.0
    skip_enemy_turn_chance: float = 0.0
    chance_survival_probability: float = 0.0
    chance_survival_hp_fraction: float = 0.0
    gamble_chance: float = 0.0
    gamble_multiplier: float = 1.0
    gamble_fail_heal_fraction: float = 0.0
    dodge_attack_multiplier: float = 1.0
    dodge_bonus_multiplier: float = 1.0
    dodge_growth: float = 0.0
    dodge_cap: float = 1.0
    entry_borderless_reset_chance: float = 0.0
    borderless_hp: float = 0.0
    borderless_attack: float = 0.0
    guardian_chance: float = 0.0
    confusion_fail_attack_ratio: float = 0.0
    # Batch-5: fades, alternating rests/shields, next-card and entry effects, field and
    # party effects, rarity/pack/class matchups. class_mask bits: undead 1, demon 2,
    # dragon 4, avian 8, rng 16, toy 32, friendship 64 (inferred classes).
    alternate_rest: int = 0
    periodic_block_period: int = 0
    fade_extra_actions: int = 0
    kill_extra_action: int = 0
    never_attacks: int = 0
    debuff_attacker_count: int = 0
    next_card_freeze_turns: int = 0
    class_mask: int = 0
    pack_index: int = 0
    bonus_vs_class_mask: int = 0
    guard_vs_class_mask: int = 0
    fade_threshold: float = 0.0
    fade_outgoing_multiplier: float = 1.0
    fade_incoming_multiplier: float = 1.0
    fade_stat_multiplier: float = 1.0
    survival_damage_max_hp_fraction: float = 0.0
    entry_next_card_stat_multiplier: float = 1.0
    next_card_stat_multiplier: float = 1.0
    next_card_gift_chance: float = 1.0
    entry_enemy_stat_subtract_fraction: float = 0.0
    entry_steal_attack_fraction: float = 0.0
    entry_steal_fraction: float = 0.0
    turn_start_hp_steal_fraction: float = 0.0
    turn_start_attack_chance: float = 0.0
    turn_start_attack_multiplier: float = 1.0
    turn_start_full_heal_chance: float = 0.0
    damaged_attack_loss_fraction: float = 0.0
    target_current_hp_damage_fraction: float = 0.0
    action_end_self_hp_loss_fraction: float = 0.0
    debuff_attacker_multiplier: float = 1.0
    first_hit_per_enemy_multiplier: float = 1.0
    field_damage_multiplier: float = 1.0
    action_end_ally_stat_multiplier: float = 1.0
    kill_ally_stat_multiplier: float = 1.0
    normal_attack_dodge_cost: float = 0.0
    action_end_incoming_multiplier: float = 1.0
    card_rarity: float = 0.0
    border_rarity: float = 1.0
    border_advantage_multiplier: float = 1.0
    border_advantage_incoming_multiplier: float = 1.0
    lower_rarity_incoming_multiplier: float = 1.0
    younger_target_multiplier: float = 1.0
    class_damage_multiplier: float = 1.0
    class_incoming_multiplier: float = 1.0
    stored_damage_to_next_attack_fraction: float = 0.0
    damage_taken_to_party_attack_fraction: float = 0.0
    # Batch-6: damage to waiting (bench) cards. User-observed: Academy Student's
    # splash kills the bench even when Arthur survives the hit at 1 HP (it uses
    # the hit's damage); Seraphim's entry attack hits every enemy, and bench
    # cards can dodge it or die, with on-death effects.
    entry_hit_all_enemies: int = 0
    # an all-enemy entry (Seraphim) (user tests, entry damage >= 655,360): the entry kills the
    # first card through any protection. Set by compile_battle.
    # Parallax: charges of "if self would die, the enemy dies instead" (lethal hits only;
    # the hit is negated and the attacking card dies). Uses the survival counter.
    death_reflects: int = 0
    splash_damage_fraction: float = 0.0
    # User (Academy Student): the attacker loses HP equal to 30% of damage dealt
    # (card text says enemies lose 40%). Basis: HP actually removed (provisional).
    recoil_damage_fraction: float = 0.0
    # Batch 7 (extrapolated from card text).
    death_damage_max_hp_fraction: float = 0.0  # On death, the enemy front card loses this share of own max HP.
    death_damage_chance: float = 1.0
    death_ally_stat_multiplier: float = 1.0  # On death, living allies' stats are scaled.
    periodic_attack_multiplier: float = 1.0  # Every `periodic_attack_period`-th own turn.
    periodic_attack_heal_max_hp_fraction: float = 0.0
    followup_multiplier: float = 0.0  # Extra last hit per action at this share of the hit's damage.
    frozen_target_multiplier: float = 1.0
    low_hp_actions_threshold: float = 0.0
    periodic_attack_period: int = 0
    periodic_attack_hits: int = 0  # Hits in the periodic action (0: attacks_per_action).
    periodic_enemy_slow_turns: int = 0  # The enemy loses turns ("rests").
    followup_bypass: int = 0
    followup_on_critical: int = 0  # The follow-up only happens after a critical hit.
    encounter_freeze_turns: int = 0  # Applied to each enemy card met while this card is active.
    encounter_confusion_turns: int = 0
    encounter_class_mask: int = 0  # 0: every enemy.
    low_hp_extra_actions: int = 0  # Extra actions per turn below low_hp_actions_threshold of max HP.
    # Poison (IMG_0297, Black Plague vs Platinum Jamiy): each tick takes the poisoner's ATK x
    # poison_fraction off the target directly, ignoring damage reduction, at every turn end,
    # waiting cards included.
    poison_fraction: float = 1.0
    entry_poison_turns: int = 0
    entry_poison_all: int = 0  # Poison every living enemy card, not only the front one.
    hit_poison_turns: int = 0
    attacked_poison_turns: int = 0  # Poisons the attacker when damaged (Pestilence).
    lethal_dodge_swap: int = 0  # After its own lethal dodge, swaps with the next living ally (Dilophosaurus).
    # Batch 9 (user videos IMG_0299-0304 and card text; provisional where noted in the catalog).
    hit_weakness_multiplier: float = 1.0  # Ghoul: its target takes this much more damage from then on.
    low_hp_heal_fraction: float = 0.0  # Zombie Nurse: once, heal this share of max HP on dropping below the threshold.
    low_hp_trigger_threshold: float = 0.0  # Share of max HP for the low-HP heal and stun.
    poison_target_max_hp_fraction: float = 0.0  # Zombie Dragon: poison ticks for a share of the target's max HP.
    delayed_strike_multiplier: float = 0.0  # Onmyoji: ATK x this, dealt after delayed_strike_turns, even after death.
    bench_growth_multiplier: float = 1.0  # Loch Ness: stats x this after each allied turn while waiting.
    bench_growth_cap: float = 0.0  # ... up to this multiple of the base stats.
    block_max_hp_fraction: float = 0.0  # Steve: each block absorbs this share of max HP.
    zero_hp_survival_turns: int = 0  # Zombie Dragon: once, survives a lethal hit, untouchable, then dies after these turns.
    encounter_doom_turns: int = 0  # Kira: every enemy card met dies after these turns, even if Kira died.
    delayed_strike_turns: int = 0
    entry_blind: int = 0  # Rudolph: the enemy's attacks always miss (IMG_0299).
    low_hp_stun_turns: int = 0  # Banshee: the attacker that drops it below the threshold cannot attack for these turns.
    stat_gamble: int = 0  # Gambler: ATK x a STAT_GAMBLE_FACTORS draw after each attack.
    block_max: int = 0
    poison_permanent: int = 0  # Its poison never wears off (Dilophosaurus).
    action_end_self_slow_turns: int = 0  # Failed Mage (user): skips its next turn after attacking.
    # Fossils (user, IMG_0314/0315): a team counter; every dying dinosaur (Prehistoric card) adds one.
    dinosaur: int = 0
    death_fossils: int = 0  # Extra fossils on death (Ancient Egg 3, Meteosaurus 2).
    entry_fossil_attack_multiplier: float = 1.0  # Tyrannodon: ATK x this per fossil on entry.
    counter_min_fossils: int = 0  # T-Rex: counterattacks only with at least this many fossils.
    fossil_death_timer: int = 0  # Velociraptor: its hit dooms the target in this many turns minus fossils (capped).
    fossil_deck_hits: int = 0  # Cyberdon: also hits fossils + 1 waiting enemies, at most this many.
    entry_self_slow_turns: int = 0  # Cyberdon: charges (skips) its first turn.
    # Ability removal, theft and copying (user battles 5-9, IMG_0316/0317/0319/0324/0325).
    turn_start_steal_fraction: float = 0.0  # Mother of Beasts: takes this share of the enemy's stats each turn.
    hit_attack_drain_fraction: float = 0.0  # Vampire Lord (IMG_0336): moves ATK equal to damage dealt from the target, at most all of it.
    turn_start_stat_growth: float = 0.0  # Limitless Rivals: a fixed gain of this share of its pre-growth stats each turn.
    turn_start_incoming_reduction: float = 0.0  # Limitless Rivals: damage reduction grows by this fixed step per turn.
    incoming_reduction_cap: float = 0.0  # Limitless Rivals (user: it caps; IMG_0337): at most this reduction.
    entry_steal_attack_chance: float = 0.0  # Witch (user): the entry ATK steal happens with this chance (0: always).
    entry_disable_enemy: int = 0  # Hell's Army: the enemy card in play loses its ability.
    hit_disable: int = 0  # Set: a card it damages loses its ability.
    cancels_all_abilities: int = 0  # Samurai: every other card's ability is cancelled while it is in play.
    entry_suppress_enemy_turns: int = 0  # Fuxi: enemy cards cannot use abilities for these turns.
    steal_first_attacked: int = 0  # Hecate: takes the ability of the first card it damages.
    kill_steal_ability: int = 0  # Mother of Beasts: takes its first victim's ability.
    copy_fallen_ally: int = 0  # Hades, Legends: copy the most recently fallen ally's ability on entry.
    disable_enemy_class_mask: int = 0  # Black Cat: enemy cards of these classes cannot use abilities while it is in play.
    entry_disable_enemy_class_mask: int = 0  # Night Witch: hexes the enemy in play if it has these classes.
    copies_enemy_in_play: int = 0  # Sable: uses the ability of the enemy card in play.
    entry_hit_next_card: int = 0  # Ra: on entry, hits the enemy card behind the one in play, ignoring its on-death ability.
    # Batch-2 primitives (card-text extrapolations): turn counting, rests,
    # alternating/own lethal dodges, conditional damage, executes, fallen-ally scaling.
    recharge_turns: int = 0  # User: recharge turns attack normally, without charged_attack_multiplier.
    charged_attack_multiplier: float = 1.0
    max_hp_damage_costs: int = 0  # User (#161): converted max HP is lost, decaying exponentially.
    first_turn_actions: int = 0
    actions_growth_per_turn: int = 0
    alternate_dodge: int = 0  # 1: odd incoming hits (Deus Ex); 2: even hits (Amaterasu). User-observed.
    restore_lowered_attack: int = 0
    own_lethal_dodges: int = 0
    bypass_defenses: int = 0
    entry_bypass_defenses: int = 0
    entry_hit_requires_fallen: int = 0
    action_end_stat_turns: int = 0
    periodic_stat_period: int = 0
    alternate_dodge_heal_fraction: float = 0.0
    lethal_survival_hp_fraction: float = 0.0
    lethal_dodge_heal_max_hp_fraction: float = 0.0
    lethal_dodge_attack_multiplier: float = 1.0
    first_attack_multiplier: float = 1.0
    hit_growth_multiplier: float = 1.0
    weak_target_multiplier: float = 1.0
    weak_target_threshold: float = 0.0
    execute_below_max_hp: float = 0.0
    execute_after_below_max_hp: float = 0.0
    mutual_destruction_below_max_hp: float = 0.0
    lost_hp_damage_fraction: float = 0.0
    max_hp_damage_fraction: float = 0.0
    target_attack_damage_fraction: float = 0.0
    advantage_damage_multiplier: float = 1.0
    disadvantage_incoming_multiplier: float = 1.0
    lost_hp_damage_bonus: float = 0.0
    lost_hp_lifesteal: float = 0.0
    self_turn_start_hp_loss_fraction: float = 0.0
    self_turn_start_max_hp_loss_fraction: float = 0.0
    entry_fallen_attack_fraction: float = 0.0
    entry_fallen_stat_bonus: float = 0.0
    entry_party_attack_bonus: float = 0.0
    low_hp_incoming_multiplier: float = 1.0
    low_hp_incoming_threshold: float = 0.0
    action_end_stat_multiplier: float = 1.0
    periodic_stat_multiplier: float = 1.0
    action_end_attack_add_base: float = 0.0
    aura_incoming_multiplier: float = 1.0
    chaos_hit_chance: float = 0.0
    lost_hp_incoming_bonus: float = 0.0
    turn_start_max_hp_multiplier: float = 1.0
    turn_start_low_hp_heal_fraction: float = 0.0
    turn_start_low_hp_threshold: float = 0.0
    entry_same_card_stat_bonus: float = 0.0
    evade_punish_multiplier: float = 0.0
    revive_on_ally_death_hp: float = 0.0
    revive_on_ally_death_chance: float = 0.0
    entry_absorb_behind: float = 0.0
    low_hp_transform_threshold: float = 0.0
    transform_stat_multiplier: float = 1.0
    growth_add_base: float = 0.0
    kill_stat_add_base: float = 0.0
    ally_redirect_fraction: float = 0.0
    enemy_turn_end_attack_drain: float = 0.0
    ability_reflect_fraction: float = 0.0
    ability_reflect_cap: float = 0.0
    summon_copy_fraction: float = 0.0
    piccolo_factor_override: float = 0.0
    luminant_evasion: float = 0.0
    enemy_death_revive_fraction: float = 0.0
    entry_ally_stat_multiplier: float = 1.0
    entry_global_steal: float = 0.0
    composer_chance_step: float = 0.0
    swap_attack_chance: float = 0.0
    swap_attack_multiplier: float = 1.0
    aura_negate_chance: float = 0.0
    rake_fraction: float = 0.0
    next_card_boost_multiplier: float = 1.0
    confused_self_hit_stat_multiplier: float = 1.0
    ally_class_stat_multiplier: float = 1.0
    swap_stat_multiplier: float = 1.0
    parry_chance: float = 0.0  # Vajra Short Sword (IMG_0342): parries an attack, no damage.
    parry_reflect_fraction: float = 0.0  # ...reflecting 75% of it, at most 75% of the attacker's current HP.
    hit_stun_chance: float = 0.0  # Staff of Perfect Enlightenment (IMG_0342): 25% to stun the card it hits.
    nullify_max_hp_threshold: float = 0.0  # Astraeus (user): nullify damage below 100% of max HP...
    nullify_threshold_step: float = 0.0  # ...the threshold dropping 15% per nullification.
    hit_burn_chance: float = 0.0  # Flame Wizard support: chance to burn the card it hits for 2 turns.
    counter_chance: float = 0.0  # Berserker support: chance to counterattack when damaged.
    followup_chance: float = 0.0  # Storm Spirit support: the follow-up hit happens only with this chance.
    ability_fail_chance: float = 0.0  # End Times support (user): a transient roll for an enemy ability to fail.
    damaged_toy_stat_gain: float = 0.0  # Awakened Toy Nutcracker (IMG_0355): every awakened Toy gains 10% stats when it is damaged.
    entry_fallen_toy_stat_bonus: float = 0.0  # Awakened Toy Bear (user): +100% stats on entry per fallen awakened Toy.
    card_id: int = 0
    spare: int = 0
    status_dodge_max: int = 0
    random_abilities: int = 0
    pool_exclude_mask: int = 0
    pool_flags: int = 0
    pool_self: int = 0
    entry_ability_swap_random: int = 0
    domain: int = 0
    random_hits_min: int = 0
    random_hits_max: int = 0
    chaos_targets: int = 0
    form_first: int = 0
    form_count: int = 0
    team_status_immunity: int = 0
    recruit_defeated: int = 0
    awaken_on_death: int = 0
    transform_form: int = 0
    reflect_even_turns: int = 0
    persistent_confusion: int = 0
    encounter_confusion_eternal: int = 0
    growth_turns: int = 0
    awaken_turn: int = 0
    doctor_period: int = 0
    summon_random: int = 0
    random_deck_target: int = 0
    overkill_carry: int = 0
    ability_death_immune: int = 0
    entry_reset_enemy_stats: int = 0
    sack: int = 0
    entry_shuffle_enemy: int = 0
    entry_hit_count_max: int = 0
    death_curse_turns: int = 0
    dad_card_id: int = 0
    split_on_death: int = 0
    ally_class_mask: int = 0
    team_class_shields: int = 0
    entry_shields: int = 0
    link_first_two: int = 0
    revive_party_on_death: int = 0
    dodge_extra_action: int = 0
    kill_summon_form: int = 0
    swap_enemies: int = 0
    curse_killer: int = 0
    kill_extends_undying: int = 0
    dutchman_swaps: int = 0
    perish_turns: int = 0
    turn_start_random_ability: int = 0
    summon_copies: int = 0
    shield_period: int = 0
    entry_hit_also_next: int = 0
    form_pair_last: int = 0
    transform_heal: int = 0
    buddha: int = 0
    enemy_entry_confusion_turns: int = 0  # Awakened Toy Jack-in-the-Box (IMG_0355): confusion for each entering enemy.
    awakened_toy: int = 0  # Magical Elf support: a Toy awakened by it.

    def validate(self):
        for name in ("hp", "attack", "dodge_probability", "outgoing_multiplier", "incoming_multiplier",
                     "entry_self_multiplier", "entry_enemy_attack_multiplier", "enemy_entry_steal_fraction",
                     "entry_hp_multiplier", "entry_attack_multiplier", "damage_reduction_max_hp",
                     "damage_cap_max_hp", "dodge_below_max_hp", "nullify_below_attack", "critical_probability", "critical_multiplier", "entry_hit_multiplier",
                     "action_end_attack_multiplier", "action_start_heal_max_hp_fraction",
                     "action_end_heal_max_hp_fraction", "heal_damage_dealt_fraction", "heal_damage_taken_fraction",
                     "after_attack_heal_max_hp_fraction", "intercept_lethal_multiplier",
                     *_BATCH1_FIELDS, *_BATCH2_FLOATS, "survival_attack_multiplier", *_BATCH4_FLOATS, *_BATCH5_FLOATS, *_BATCH6_FLOATS, *_BATCH7_FLOATS, 'poison_fraction', *_BATCH9_FLOATS, 'entry_fossil_attack_multiplier', 'turn_start_steal_fraction', 'hit_attack_drain_fraction', 'turn_start_stat_growth', 'turn_start_incoming_reduction', 'entry_steal_attack_chance', 'incoming_reduction_cap', *_BATCH12_FLOATS):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid fighter {name}")
        if any(getattr(self, name) > 1 for name in ("hit_freeze_chance", "skip_enemy_turn_chance", "chance_survival_probability",
                                                    "gamble_chance", "entry_borderless_reset_chance", "guardian_chance", "dodge_cap",
                                                    'death_damage_chance', *_BATCH5_CHANCES)):
            raise ValueError("Chances must be at most 1")
        if any(getattr(self, name) > 1 for name in ('chaos_hit_chance', 'revive_on_ally_death_chance', 'swap_attack_chance',
                                                     'aura_negate_chance', 'luminant_evasion', 'parry_chance', 'hit_stun_chance',
                                                     'hit_burn_chance', 'counter_chance', 'followup_chance', 'ability_fail_chance')):
            raise ValueError("Chances must be at most 1")
        if any(getattr(self, name) <= 0 for name in _BATCH12_ONES):
            raise ValueError("Batch-12 multipliers must be positive")
        for name in _BATCH12_INTS:
            limit = 1023 if name in _BATCH12_WIDE else 127 if name.endswith("_mask") else 16
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= limit:
                raise ValueError(f"{name} must be an integer in 0..{limit}")
        if any(getattr(self, name) <= 0 for name in _POSITIVE_FIELDS):
            raise ValueError("Stat multipliers must be positive")
        if self.hp <= 0 or self.entry_self_multiplier <= 0 or self.entry_hp_multiplier <= 0 or self.dodge_probability > 1:
            raise ValueError("HP and entry self multiplier must be positive; dodge must be <=1")
        if self.enemy_entry_steal_fraction >= 1:
            raise ValueError("Enemy-entry steal fraction must be less than one")
        if self.critical_probability > 1 or self.critical_multiplier < 1:
            raise ValueError("Critical probability must be <=1 and multiplier >=1")
        if type(self.heal_on_kill) is not int or self.heal_on_kill not in (0, 1):
            raise ValueError("heal_on_kill must be 0 or 1")
        if type(self.counter_on_damage) is not int or self.counter_on_damage not in (0, 1):
            raise ValueError("counter_on_damage must be 0 or 1")
        if type(self.lethal_survivals) is not int or not 0 <= self.lethal_survivals <= 16:
            raise ValueError("lethal_survivals must be an integer in 0..16")
        if type(self.next_card_dodges) is not int or not 0 <= self.next_card_dodges <= 16:
            raise ValueError("next_card_dodges must be an integer in 0..16")
        for name in _BATCH2_INTS + _BATCH3_INTS + _BATCH4_INTS + _BATCH5_INTS + _BATCH6_INTS + _BATCH7_INTS + _BATCH8_INTS + _BATCH9_INTS + _BATCH10_INTS + _BATCH11_INTS:
            limit = 127 if name.endswith("_mask") or name == "pack_index" else 16
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= limit:
                raise ValueError(f"{name} must be an integer in 0..{limit}")
        if type(self.invincible) is not int or self.invincible not in (0, 1):
            raise ValueError("invincible must be 0 or 1")
        if type(self.lifetime_actions) is not int or not 0 <= self.lifetime_actions < 2**31:
            raise ValueError("lifetime_actions must be an integer in 0..2^31-1")
        if type(self.block_mode) is not int or self.block_mode not in (0, 1, 2):
            raise ValueError("Unknown block mode")
        if type(self.attacks_per_action) is not int or not 1 <= self.attacks_per_action <= MAX_ATTACKS_PER_ACTION:
            raise ValueError(f"attacks_per_action must be an integer in 1..{MAX_ATTACKS_PER_ACTION}")
        if type(self.actions_per_turn) is not int or not 1 <= self.actions_per_turn <= 16:
            raise ValueError("actions_per_turn must be an integer in 1..16")


_VALIDATED_POOLS = set()


@dataclass(frozen=True, slots=True)
class Battle:
    teams: tuple[tuple[Fighter, ...], tuple[Fighter, ...]]
    first_side: int = 0
    # Cards in play at the start (the rest of each team are spare slots for summons); () = every card.
    lineup: tuple[int, ...] = ()
    # Abilities that random, copied or transformed cards can take (ability codes POOL_CODE + i).
    pool: tuple[Fighter, ...] = ()

    def sizes(self):
        return self.lineup or tuple(len(team) for team in self.teams)

    def validate(self):
        if len(self.teams) != 2 or any(not 1 <= len(team) <= MAX_FIGHTERS for team in self.teams):
            raise ValueError(f"Expected two ordered teams with 1..{MAX_FIGHTERS} fighters each")
        if self.lineup and (len(self.lineup) != 2 or any(type(n) is not int or not 1 <= n <= len(team)
                                                         for n, team in zip(self.lineup, self.teams))):
            raise ValueError("lineup must give 1..len(team) starting cards per side")
        if type(self.first_side) is not int or self.first_side not in (0, 1):
            raise ValueError("first_side must be 0 or 1")
        if len(self.pool) > 1023 - POOL_CODE:
            raise ValueError("Ability pool too large")
        for team in self.teams:
            for fighter in team:
                fighter.validate()
        if self.pool and id(self.pool) not in _VALIDATED_POOLS:
            for fighter in self.pool:
                fighter.validate()
            _VALIDATED_POOLS.add(id(self.pool))


@dataclass(frozen=True, slots=True)
class Options:
    mode: str = "branch"
    rounding: str = "ceil"
    stat_rounding: str = "unrounded"
    repeat_cycles: int = 50  # Witch mirror: both active cards fall after 100 ordinary turns.
    max_steps: int = 1000  # Hit resolutions per branch, including dodged/blocked hits.
    max_frontier: int = 10000
    prune_probability: float = 0.0
    seed: int = 1
    # Branch mode: states pruned for low probability or frontier size are resolved by
    # this many Monte Carlo playouts (averaged) instead of counting as unresolved.
    # With rollout_error > 0 it is the maximum: playouts stop once the state's share of the
    # result's variance is small enough (see _rollout), so the total standard error stays <= rollout_error.
    rollouts: int = 0
    rollout_error: float = 0.0
    # Branch mode: once this many states have been expanded (0 = unlimited), every state left on the
    # frontier is resolved by playouts and the search stops (user: branch the largest probabilities first,
    # simulate the small branches when the depth is exhausted or the budget is spent).
    node_budget: int = 0
    # Branch mode: a chance roll whose smallest branch would carry less probability than this is sampled
    # (one seeded draw keeps the whole weight) instead of enumerated; 0 enumerates every roll.
    sample_below: float = 0.0

    def validate(self):
        if self.mode not in ("sample", "branch") or any(policy not in ("unrounded", "ceil", "floor", "nearest_half_up") for policy in (self.rounding, self.stat_rounding)):
            raise ValueError("Unknown mode or rounding policy")
        if type(self.rollouts) is not int or not 0 <= self.rollouts < 2**31:
            raise ValueError("rollouts must be an integer in 0..2^31-1")
        for name in ("repeat_cycles", "max_steps", "max_frontier"):
            if type(getattr(self, name)) is not int or not 0 < getattr(self, name) < 2**31:
                raise ValueError(f"{name} must be a positive integer below 2^31")
        if type(self.prune_probability) not in (float, int) or not math.isfinite(self.prune_probability) or not 0 <= self.prune_probability <= 1:
            raise ValueError("Invalid pruning probability")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("Seed must be an unsigned 64-bit integer")
        if type(self.rollout_error) not in (float, int) or not math.isfinite(self.rollout_error) or not 0 <= self.rollout_error < 1:
            raise ValueError("rollout_error must be in [0, 1)")
        if type(self.node_budget) is not int or not 0 <= self.node_budget < 2**31:
            raise ValueError("node_budget must be an integer in 0..2^31-1")
        if type(self.sample_below) not in (float, int) or not math.isfinite(self.sample_below) or not 0 <= self.sample_below <= 1:
            raise ValueError("sample_below must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Result:
    p_a: float
    p_b: float
    tie: float
    unresolved: float
    expanded_states: int
    merged_states: int
    rules_status: str = "experimental_subset"
    training_labels_allowed: bool = False
    trace: tuple[dict, ...] = field(default_factory=tuple)
    estimated: float = 0.0  # Probability touched by Monte Carlo (playouts or sampled rolls), capped at 1; 0 = exact.

    @property
    def accounted_probability(self):
        return self.p_a + self.p_b + self.tie + self.unresolved


@dataclass(frozen=True, slots=True)
class _State:
    hp: tuple[tuple[float, ...], ...]
    max_hp: tuple[tuple[float, ...], ...]
    attack: tuple[tuple[float, ...], ...]
    used_block: tuple[tuple[bool, ...], ...]
    survivals_used: tuple[tuple[int, ...], ...]
    completed_actions: tuple[tuple[int, ...], ...]
    order: tuple[tuple[int, ...], ...]  # Team position -> fighter index; stats are per fighter.
    entered: tuple[tuple[bool, ...], ...]
    granted_dodges: tuple[tuple[int, ...], ...]
    stolen_from: tuple[tuple[bool, ...], ...]
    turns: tuple[tuple[int, ...], ...]  # Own turns started; only counted for turn-dependent cards.
    hits_taken: tuple[tuple[int, ...], ...]  # Only counted for alternating dodgers.
    own_dodges_used: tuple[tuple[int, ...], ...]
    burn: tuple[tuple[int, ...], ...]
    bleed: tuple[tuple[int, ...], ...]
    frozen: tuple[tuple[int, ...], ...]
    slowed: tuple[tuple[int, ...], ...]
    frostbite: tuple[tuple[int, ...], ...]
    confused: tuple[tuple[int, ...], ...]
    dodge_charged: tuple[tuple[int, ...], ...]
    revived: tuple[tuple[int, ...], ...]
    guarded: tuple[tuple[int, ...], ...]
    faded: tuple[tuple[int, ...], ...]
    resting: tuple[tuple[int, ...], ...]
    debuffed: tuple[tuple[int, ...], ...]
    marked: tuple[tuple[int, ...], ...]
    incoming_scale: tuple[tuple[float, ...], ...]
    stored: tuple[tuple[float, ...], ...]
    poison: tuple[tuple[int, ...], ...]  # Turns left.
    poison_damage: tuple[tuple[float, ...], ...]  # HP lost per tick.
    weakness: tuple[tuple[float, ...], ...]  # Incoming damage multiplier from Ghoul.
    triggered: tuple[tuple[int, ...], ...]  # One-time abilities used: 1 low-HP heal, 2 zero-HP survival, 4 stun.
    undying: tuple[tuple[int, ...], ...]  # Zombie Dragon: turn ends left at 0 HP, untouchable; then it dies.
    doom: tuple[tuple[int, ...], ...]  # Kira: enemy normal attacks left before this card dies while in play.
    strike_timer: tuple[tuple[int, ...], ...]  # Onmyoji: allied normal attacks left before its strike.
    strike_damage: tuple[tuple[float, ...], ...]
    blind: tuple[tuple[int, ...], ...]  # Attacks always miss (Rudolph).
    block_pool: tuple[tuple[float, ...], ...]  # Damage the card's blocks can still absorb (Steve).
    death_timer: tuple[tuple[int, ...], ...]  # Velociraptor: turn ends until this card dies.
    fossils: tuple[int, int]  # Team fossil counters.
    ability: tuple[tuple[int, ...], ...]  # Whose ability each card uses (ABILITY_NONE: removed).
    suppressed: tuple[int, int]  # Fuxi: turn ends left during which a side cannot use abilities.
    ext: tuple  # Batch-12 per-card arrays (_EXT_CARD order), then each side's negated aura flag.
    active: tuple[int, int]
    actor: int
    pair_actions: int
    hits_remaining: int
    counter_pending: int
    counter_from_entry: int
    entry_queue: tuple[int, ...]
    turn_actions_remaining: int


class _RNG:
    def __init__(self, seed):
        self.state = seed or 0x9E3779B97F4A7C15

    def random(self):
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & ((1 << 64) - 1)
        x ^= x >> 27
        self.state = x & ((1 << 64) - 1)
        result = (self.state * 2685821657736338717) & ((1 << 64) - 1)
        return (result >> 11) * 2**-53


def _blank(fighter):
    """The card with no ability: identity fields only."""
    return Fighter(**{name: getattr(fighter, name) for name in IDENTITY_FIELDS})


def _source(base, ability, side, index):
    code = ability[side][index]
    if code == ABILITY_NONE:
        return None
    if code == 0:
        return base.teams[side][index]
    if code >= POOL_CODE:
        return base.pool[code - POOL_CODE]
    code -= 1
    return base.teams[code // 16][code % 16]


def _ability_code(ability, side, index):
    """The code another card stores when it takes this card's current ability."""
    code = ability[side][index]
    return 1 + side * 16 + index if code == 0 else code


_VIEWS = {}
_DEFAULT = Fighter(1.0, 0.0)
_FIGHTER_FIELDS = tuple(Fighter.__slots__)
# The Curse's domain (user Part A): enemy dodges, blocks, damage reduction and survivals stop working
# (not Parallax's reflection).
_DOMAIN_STRIP = {'dodge_probability': 0.0, 'block_mode': 0, 'damage_reduction_max_hp': 0.0, 'damage_cap_max_hp': 0.0,
                 'dodge_below_max_hp': 0.0, 'nullify_below_attack': 0.0, 'lethal_survivals': 0, 'invincible': 0,
                 'own_lethal_dodges': 0, 'alternate_dodge': 0, 'chance_survival_probability': 0.0, 'guardian_chance': 0.0,
                 'hit_chance_squared': 0, 'status_dodge_max': 0, 'zero_hp_survival_turns': 0, 'aura_incoming_multiplier': 1.0,
                 'lost_hp_incoming_bonus': 0.0}


def _merge(first, second):
    """Two abilities at once (Pandora, the Great Nirvana Sword): the second's non-default fields win."""
    return replace(first, **{name: getattr(second, name) for name in _FIGHTER_FIELDS
                             if getattr(second, name) != getattr(_DEFAULT, name)})


def _strip(fighter):
    values = {name: value for name, value in _DOMAIN_STRIP.items()}
    for name in ('incoming_multiplier', 'low_hp_incoming_multiplier', 'disadvantage_incoming_multiplier'):
        values[name] = max(1.0, getattr(fighter, name))
    return replace(fighter, **values)


def _fail_chance(base, ext, side):
    """End Times on `side`: the chance an enemy ability fails (none while Marrowclaw negates the support)."""
    negated = ext['negated'] if isinstance(ext, dict) else ext[-1]
    return 0.0 if negated[side] else max((f.ability_fail_chance for f in base.teams[side]), default=0.0)


def _enemy_entry(battle, hp, order, active, status, side):
    """Awakened Toy Jack-in-the-Box (IMG_0355): every enemy that enters is confused for a turn per unique Toy."""
    index, enemy = order[side][active[side]], 1 - side
    if active[enemy] >= len(order[enemy]) or hp[side][index] <= 0:
        return
    turns = battle.teams[enemy][order[enemy][active[enemy]]].enemy_entry_confusion_turns
    if turns and hp[enemy][order[enemy][active[enemy]]] > 0:
        status[5][side][index] = max(status[5][side][index], turns)


def _view(base, ability, suppressed, order, active, hp, ability2=None, failed=(0, 0)):
    """The battle as each card currently acts: abilities removed, stolen, copied, combined or cancelled
    (Samurai, Fuxi, Black Cat, Sable, Pandora, The Curse); identity fields always stay the card's own."""
    second = ability2 if ability2 is not None and any(code for row in ability2 for code in row) else None
    dynamic = any(f.cancels_all_abilities or f.disable_enemy_class_mask or f.copies_enemy_in_play or f.domain
                  for team in base.teams for f in team)
    if not dynamic and not any(suppressed) and not any(failed) and not any(code for row in ability for code in row) and second is None:
        return base
    front = tuple(order[s][active[s]] if active[s] < len(order[s]) and hp[s][order[s][active[s]]] > 0 else None for s in (0, 1))
    key = (id(base), tuple(map(tuple, ability)), tuple(suppressed), front,
           None if second is None else tuple(map(tuple, second)), tuple(failed))
    cached = _VIEWS.get(key)
    if cached is not None and cached[0] is base:
        return cached[1]
    sources = [[_source(base, ability, s, i) for i in range(len(base.teams[s]))] for s in (0, 1)]
    samurai = [s for s in (0, 1) if front[s] is not None and sources[s][front[s]] is not None
               and sources[s][front[s]].cancels_all_abilities]
    # Sable (user): takes the enemy card in play's ability and disables it on that card.
    sable = [s for s in (0, 1) if front[s] is not None and sources[s][front[s]] is not None
             and sources[s][front[s]].copies_enemy_in_play]
    teams = []
    for s in (0, 1):
        e = 1 - s
        enemy_front = sources[e][front[e]] if front[e] is not None else None
        team = []
        for i, fighter in enumerate(base.teams[s]):
            source = sources[s][i]
            cancelled = (suppressed[s] > 0 or any(not (t == s and front[t] == i) for t in samurai) or (failed[s] and front[s] == i)
                         or (1 - s in sable and front[s] == i))
            if enemy_front is not None and enemy_front.disable_enemy_class_mask & fighter.class_mask:
                cancelled = True  # Black Cat: RNG abilities always fail.
            if source is not None and source.copies_enemy_in_play and enemy_front is not None:
                source = enemy_front  # Sable (user): copies the enemy card in play.
            if not cancelled and source is not None and second is not None and second[s][i]:
                extra = _source(base, second, s, i)
                if extra is not None:
                    source = _merge(source, extra)
            if cancelled or source is None:
                card = _blank(fighter)
            elif source is fighter:
                card = fighter
            else:
                card = replace(source, **{name: getattr(fighter, name) for name in IDENTITY_FIELDS})
            if enemy_front is not None and enemy_front.domain:
                card = _strip(card)
            team.append(card)
        teams.append(tuple(team))
    result = Battle(tuple(teams), base.first_side, base.lineup, base.pool)
    if len(_VIEWS) > 4096:
        _VIEWS.clear()
    _VIEWS[key] = (base, result)
    return result


class _Chance:
    """Random rolls inside one step. Sample mode draws lazily from the RNG; branch
    mode replays `forced` outcomes and defaults later rolls to False, recording
    their probabilities so the caller can enumerate both outcomes."""

    def __init__(self, forced=(), rng=None):
        self.forced, self.rng, self.probabilities = forced, rng, []

    def roll(self, p):
        if p <= 0:
            return False
        if p >= 1:
            return True
        index = len(self.probabilities)
        if index >= MAX_CHANCE_ROLLS:
            raise ValueError("Too many chance rolls in one step")
        self.probabilities.append(p)
        if self.rng is not None:
            return self.rng.random() < p
        return bool(self.forced[index]) if index < len(self.forced) else False

    def pick(self, n):
        """A uniform choice among n outcomes; branch mode enumerates all of them."""
        if n <= 1:
            return 0
        index = len(self.probabilities)
        if index >= MAX_CHANCE_ROLLS:
            raise ValueError("Too many chance rolls in one step")
        self.probabilities.append(-n)
        if self.rng is not None:
            return min(n - 1, int(self.rng.random() * n))
        return self.forced[index] if index < len(self.forced) else 0


def _expand(run, weight, rng=None, collapse=None):
    """Yield (result, weight) over every combination of chance rolls in `run`.

    collapse = [threshold, rng, estimated]: a roll whose smallest branch would carry less than `threshold`
    is not enumerated; one outcome is drawn (seeded `rng`) and keeps the whole weight, an unbiased estimate
    (user: branch the large probabilities, simulate the small ones); the first such draw on a path adds
    its weight to `estimated`."""
    if rng is not None:
        yield run(_Chance(rng=rng)), weight
        return
    stack = [((), weight, False)]
    while stack:
        forced, w, drawn = stack.pop()
        chance = _Chance(forced)
        result = run(chance)
        if len(chance.probabilities) > len(forced):
            p = chance.probabilities[len(forced)]
            smallest = w / -p if p < 0 else w * min(p, 1 - p)
            if collapse is not None and smallest < collapse[0]:
                u = collapse[1].random()
                outcome = min(-p - 1, int(u * -p)) if p < 0 else u < p
                if not drawn:
                    collapse[2] += w
                stack.append((forced + (outcome,), w, True))
            elif p < 0:  # An n-way pick: outcome 0 is expanded first.
                for k in range(-p - 1, -1, -1):
                    stack.append((forced + (k,), w / -p, drawn))
            else:
                stack.append((forced + (True,), w * p, drawn))
                stack.append((forced + (False,), w * (1 - p), drawn))
        else:
            yield result, w


def _event(trace, phase, **fields):
    if trace is not None:
        trace.append({"phase": phase, **fields})


def _freeze(hp, max_hp, attack, used, survivals, completed, order, entered, dodges, stolen, turns, hits, own_dodges, burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked, inscale, stored, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, active, actor, actions, hits_remaining=0,
            counter_pending=0, counter_from_entry=0, entry_queue=(), turn_actions_remaining=0):
    for row in hp + max_hp + attack:
        if any(not math.isfinite(value) for value in row):
            raise ValueError("Battle arithmetic overflowed")
    return _State(tuple(map(tuple, hp)), tuple(map(tuple, max_hp)), tuple(map(tuple, attack)), tuple(map(tuple, used)), tuple(map(tuple, survivals)),
                  tuple(map(tuple, completed)), tuple(map(tuple, order)), tuple(map(tuple, entered)), tuple(map(tuple, dodges)), tuple(map(tuple, stolen)), tuple(map(tuple, turns)), tuple(map(tuple, hits)), tuple(map(tuple, own_dodges)), tuple(map(tuple, burn)), tuple(map(tuple, bleed)), tuple(map(tuple, frozen)), tuple(map(tuple, slowed)), tuple(map(tuple, frostbite)), tuple(map(tuple, confused)), tuple(map(tuple, charged)), tuple(map(tuple, revived)), tuple(map(tuple, guarded)), tuple(map(tuple, faded)), tuple(map(tuple, resting)), tuple(map(tuple, debuffed)), tuple(map(tuple, marked)), tuple(map(tuple, inscale)), tuple(map(tuple, stored)), tuple(map(tuple, poison)), tuple(map(tuple, poison_damage)), tuple(map(tuple, weakness)), tuple(map(tuple, triggered)), tuple(map(tuple, undying)), tuple(map(tuple, doom)), tuple(map(tuple, strike_timer)), tuple(map(tuple, strike_damage)), tuple(map(tuple, blind)), tuple(map(tuple, block_pool)), tuple(map(tuple, death_timer)), tuple(fossils), tuple(map(tuple, ability)), tuple(suppressed), _freeze_ext(ext), tuple(active), actor, actions, hits_remaining, counter_pending,
                  counter_from_entry, tuple(entry_queue), turn_actions_remaining)


def _new_ext(battle):
    ext = {key: [[0.0 if key in ('bonus_hp', 'bonus_attack') else 0 for _ in team] for team in battle.teams] for key in _EXT_CARD}
    ext['negated'] = [0, 0]
    return ext


def _freeze_ext(ext):
    return tuple(tuple(map(tuple, ext[key])) for key in _EXT_CARD) + (tuple(ext['negated']),)


def _thaw_ext(frozen):
    ext = {key: [list(row) for row in rows] for key, rows in zip(_EXT_CARD, frozen)}
    ext['negated'] = list(frozen[-1])
    return ext


def _hit_actor(state):
    if state.counter_pending:
        return state.counter_pending - 1
    return state.entry_queue[0] if state.entry_queue else state.actor


def _terminal(battle, active, order):
    dead = [active[side] >= len(order[side]) for side in (0, 1)]
    if all(dead):
        return 1 - battle.first_side  # Initiator needs a surviving card to win.
    return 1 if dead[0] else 0 if dead[1] else None


def _steal(battle, hp, max_hp, attack, active, order, stolen, side, rounding, trace):
    """Enemy-entry theft against the card now at the front of `side`.

    User-observed: Chronus steals when Piccolo swaps in, but never from the same
    card twice. A returning card never stolen from is assumed to be stolen from.
    """
    index = order[side][active[side]]
    enemy = 1 - side
    enemy_index = order[enemy][active[enemy]]
    fraction = battle.teams[enemy][enemy_index].enemy_entry_steal_fraction
    if not fraction or stolen[side][index] or hp[side][index] <= 0:  # Nothing to steal from a 0-HP card.
        return
    stolen[side][index] = True
    stolen_hp = _round(hp[side][index] * fraction, rounding)
    stolen_attack = _round(attack[side][index] * fraction, rounding)
    hp[side][index] = _round(hp[side][index] - stolen_hp, rounding)
    max_hp[side][index] = _round(max_hp[side][index] - stolen_hp, rounding)
    attack[side][index] = _round(attack[side][index] - stolen_attack, rounding)
    hp[enemy][enemy_index] = _round(hp[enemy][enemy_index] + stolen_hp, rounding)
    max_hp[enemy][enemy_index] = _round(max_hp[enemy][enemy_index] + stolen_hp, rounding)
    attack[enemy][enemy_index] = _round(attack[enemy][enemy_index] + stolen_attack, rounding)
    if hp[side][index] <= 0 or attack[side][index] < 0:
        raise ValueError("Lethal/negative-stat entry theft is not implemented in this subset")
    _event(trace, "ENEMY_ENTRY", side=enemy, entering_side=side, stolen_hp=stolen_hp, stolen_attack=stolen_attack)


def _entry(battle, hp, max_hp, attack, active, order, entered, stolen, status, side, rounding, trace, chance=None, body_only=False):
    """A card entering play; body_only re-applies the entry effects of a new form (Glamour, IMG_0330)."""
    index = order[side][active[side]]
    fighter = battle.teams[side][index]
    enemy = 1 - side
    enemy_index = order[enemy][active[enemy]]
    if entered[side][index] and not body_only:
        # A card displaced by an interception returns to the front later.
        # User-observed: Poseidon, Knightmare and Good Boy repeat no entry effect.
        _event(trace, "REENTRY", side=side, slot=active[side], hp=hp[side][index], attack=attack[side][index])
        _steal(battle, hp, max_hp, attack, active, order, stolen, side, rounding, trace)
        _enemy_entry(battle, hp, order, active, status, side)
        return False
    entered[side][index] = True
    arriving_hp = hp[side][index]  # May already be <= 0 if hit while waiting.
    hp[side][index] = _round(hp[side][index] * fighter.entry_self_multiplier * fighter.entry_hp_multiplier, rounding)
    max_hp[side][index] = _round(max_hp[side][index] * fighter.entry_self_multiplier * fighter.entry_hp_multiplier, rounding)
    attack[side][index] = _round(attack[side][index] * fighter.entry_self_multiplier * fighter.entry_attack_multiplier, rounding)
    if hp[side][index] <= 0 < arriving_hp:
        raise ValueError("Lethal entry self scaling is not implemented in this subset")
    fallen = active[side]  # Positions ahead are dead; displaced cards wait behind.
    if fighter.entry_fallen_stat_bonus and fallen:
        _scale(hp, max_hp, attack, side, index, 1 + fighter.entry_fallen_stat_bonus * fallen,
               1 + fighter.entry_fallen_stat_bonus * fallen, rounding)
    if fighter.entry_fallen_toy_stat_bonus:
        toys = sum(1 for i, f in enumerate(battle.teams[side]) if f.awakened_toy and hp[side][i] <= 0)
        if toys:
            factor = 1 + fighter.entry_fallen_toy_stat_bonus * toys
            _scale(hp, max_hp, attack, side, index, factor, factor, rounding)
    if fighter.entry_fallen_attack_fraction and fallen:
        attack[side][index] = _round(attack[side][index] + fighter.entry_fallen_attack_fraction
                                     * sum(attack[side][order[side][p]] for p in range(fallen)), rounding)
    if fighter.fade_stat_multiplier != 1:
        _scale(hp, max_hp, attack, side, index, fighter.fade_stat_multiplier, fighter.fade_stat_multiplier, rounding)
    if fighter.entry_next_card_stat_multiplier != 1 and fallen + 1 < len(order[side]):
        nxt = order[side][fallen + 1]
        _scale(hp, max_hp, attack, side, nxt, fighter.entry_next_card_stat_multiplier, fighter.entry_next_card_stat_multiplier, rounding)
    if fighter.entry_party_attack_bonus:
        attack[side][index] = _round(attack[side][index] * (1 + fighter.entry_party_attack_bonus
                                                             * len(order[side])), rounding)  # User: dead or alive.
    if fighter.entry_fossil_attack_multiplier != 1 and status[17][side]:
        # Tyrannodon (IMG_0314: 8,678 x 1.5^7 = 148,272 with 7 fossils).
        attack[side][index] = _round(attack[side][index] * fighter.entry_fossil_attack_multiplier ** status[17][side], rounding)
    if fighter.entry_self_slow_turns:
        status[3][side][index] = max(status[3][side][index], fighter.entry_self_slow_turns)
    attack[enemy][enemy_index] = _round(attack[enemy][enemy_index] * fighter.entry_enemy_attack_multiplier
                                        * fighter.entry_enemy_stat_multiplier, rounding)
    hp[enemy][enemy_index] = _round(hp[enemy][enemy_index] * fighter.entry_enemy_hp_multiplier
                                    * fighter.entry_enemy_stat_multiplier, rounding)
    max_hp[enemy][enemy_index] = _round(max_hp[enemy][enemy_index] * fighter.entry_enemy_stat_multiplier, rounding)
    for kind, turns_applied in enumerate((fighter.entry_burn_turns, 0, fighter.entry_freeze_turns, fighter.entry_slow_turns)):
        if turns_applied:
            status[kind][enemy][enemy_index] = max(status[kind][enemy][enemy_index], turns_applied)
    chance = chance or _Chance()
    enemy_fighter = battle.teams[enemy][enemy_index]
    if fighter.entry_enemy_stat_subtract_fraction:
        cut = fighter.entry_enemy_stat_subtract_fraction  # Clamped: entry effects cannot kill in this subset.
        hp[enemy][enemy_index] = max(1.0, _round(hp[enemy][enemy_index] - hp[side][index] * cut, rounding))
        max_hp[enemy][enemy_index] = max(hp[enemy][enemy_index], _round(max_hp[enemy][enemy_index] - hp[side][index] * cut, rounding))
        attack[enemy][enemy_index] = max(0.0, _round(attack[enemy][enemy_index] - attack[side][index] * cut, rounding))
    if fighter.entry_steal_attack_fraction and (not fighter.entry_steal_attack_chance or chance.roll(fighter.entry_steal_attack_chance)):
        take = _round(attack[enemy][enemy_index] * fighter.entry_steal_attack_fraction, rounding)
        attack[enemy][enemy_index] = _round(attack[enemy][enemy_index] - take, rounding)
        attack[side][index] = _round(attack[side][index] + take, rounding)
    if fighter.entry_steal_fraction:
        _transfer(hp, max_hp, attack, (enemy, enemy_index), (side, index), fighter.entry_steal_fraction, rounding, trace, "ENTRY_STEAL")
    if fighter.entry_borderless_reset_chance and chance.roll(fighter.entry_borderless_reset_chance):
        # Achyls: the enemy returns to its borderless base stats.
        hp[enemy][enemy_index] = max_hp[enemy][enemy_index] = float(enemy_fighter.borderless_hp or enemy_fighter.hp)
        attack[enemy][enemy_index] = float(enemy_fighter.borderless_attack or enemy_fighter.attack)
        _event(trace, "BORDERLESS_RESET", side=enemy)
    if fighter.entry_random_debuff_turns:
        # Old Man Winter: one of frostbite, slow or freeze, equally likely (extrapolated).
        kind = 4 if chance.roll(1 / 3) else 3 if chance.roll(.5) else 2
        status[kind][enemy][enemy_index] = max(status[kind][enemy][enemy_index], fighter.entry_random_debuff_turns)
    if fighter.entry_confusion_turns and not (fighter.confusion_fail_attack_ratio and
                                              attack[enemy][enemy_index] >= attack[side][index] * fighter.confusion_fail_attack_ratio):
        status[5][enemy][enemy_index] = max(status[5][enemy][enemy_index], fighter.entry_confusion_turns)
    # Ice King / Scarecrow: statuses for every enemy card met while this card is active.
    for src_side, src, dst_side, dst in ((side, index, enemy, enemy_index), (enemy, enemy_index, side, index)):
        source, met = battle.teams[src_side][src], battle.teams[dst_side][dst]
        if (hp[src_side][src] > 0 and hp[dst_side][dst] > 0
                and (not source.encounter_class_mask or source.encounter_class_mask & met.class_mask)):
            status[2][dst_side][dst] = max(status[2][dst_side][dst], source.encounter_freeze_turns)
            status[5][dst_side][dst] = max(status[5][dst_side][dst], source.encounter_confusion_turns)
            if source.encounter_confusion_eternal:
                status[5][dst_side][dst] = max(status[5][dst_side][dst], CONFUSION_ETERNAL)  # Cthulu.
            if source.encounter_doom_turns and not status[11][dst_side][dst]:
                status[11][dst_side][dst] = source.encounter_doom_turns  # Kira: an existing doom is not extended.
    if hp[enemy][enemy_index] > 0 and (fighter.entry_disable_enemy
                                       or fighter.entry_disable_enemy_class_mask & battle.teams[enemy][enemy_index].class_mask):
        status[18][enemy][enemy_index] = ABILITY_NONE  # Hell's Army; Night Witch's hex on RNG cards.
    if fighter.entry_suppress_enemy_turns:
        status[19][enemy] = max(status[19][enemy], fighter.entry_suppress_enemy_turns)  # Fuxi.
    if fighter.copy_fallen_ally:
        for position in range(active[side] - 1, -1, -1):  # Hades, Legends: the most recently fallen ally.
            ally = order[side][position]
            if hp[side][ally] <= 0:
                status[18][side][index] = _ability_code(status[18], side, ally)
                break
    if fighter.entry_blind and hp[enemy][enemy_index] > 0:
        status[14][enemy][enemy_index] = 1
    if fighter.delayed_strike_turns:
        status[12][side][index] = fighter.delayed_strike_turns
        status[13][side][index] = attack[side][index] * fighter.delayed_strike_multiplier
    if fighter.entry_poison_turns:
        amount = attack[side][index] * fighter.poison_fraction
        turns = POISON_PERMANENT if fighter.poison_permanent else fighter.entry_poison_turns
        targets = range(active[enemy], len(order[enemy])) if fighter.entry_poison_all else (active[enemy],)
        for q in targets:
            if hp[enemy][order[enemy][q]] > 0:
                _poison(status, enemy, order[enemy][q], turns, amount)
    _entry_effects(battle, hp, max_hp, attack, active, order, stolen, status, side, index, rounding, trace, chance)
    _event(trace, "ENTRY", side=side, slot=active[side], hp=hp[side][index], max_hp=max_hp[side][index], attack=attack[side][index],
           enemy_attack=attack[enemy][enemy_index])
    if not body_only:
        _steal(battle, hp, max_hp, attack, active, order, stolen, side, rounding, trace)
        _enemy_entry(battle, hp, order, active, status, side)
    return True


def _queues_entry_hit(base, status, order, active, hp, side):
    """Whether the card now in front has an entry attack, in its current form (Awakened One's War Scythe)."""
    view = _view(base, status[18], status[19], order, active, hp, status[20]['ability2'])
    return _has_entry_hit(view.teams[side][order[side][active[side]]], active[side])


def _entry_effects(battle, hp, max_hp, attack, active, order, stolen, status, side, index, rounding, trace, chance):
    """Batch-12 entry effects, in this order (the C kernel follows it)."""
    fighter = battle.teams[side][index]
    enemy = 1 - side
    enemy_index = order[enemy][active[enemy]] if active[enemy] < len(order[enemy]) else None
    ext, ability, inscale = status[20], status[18], status[22]
    alive = enemy_index is not None and hp[enemy][enemy_index] > 0
    if fighter.aura_negate_chance and chance.roll(fighter.aura_negate_chance):
        ext['negated'][enemy] = 1  # Marrowclaw: the enemy's blue support stops working.
    if fighter.persistent_confusion and alive:
        status[5][enemy][enemy_index] = max(status[5][enemy][enemy_index], CONFUSION_UNTIL_SELF_HIT)  # Kuchisake-onna
    if fighter.entry_reset_enemy_stats and alive:
        base = battle.teams[enemy][enemy_index]  # Poison Witch: the enemy's stat changes are cleared.
        hp[enemy][enemy_index] = _round(base.hp * hp[enemy][enemy_index] / max_hp[enemy][enemy_index], rounding)
        max_hp[enemy][enemy_index] = float(base.hp)
        attack[enemy][enemy_index] = float(base.attack)
    if fighter.random_abilities:  # Pandora (user): any abilities but its own, at most 2.
        first = _pool_pick(battle, chance, fighter.pool_exclude_mask, fighter.pool_self - 1)
        if first is not None:
            ability[side][index] = first
            if fighter.random_abilities > 1:
                second = _pool_pick(battle, chance, fighter.pool_exclude_mask, fighter.pool_self - 1, first - POOL_CODE)
                if second is not None:
                    ext['ability2'][side][index] = second
    if fighter.entry_ability_swap_random and alive:  # Loki: takes the enemy's ability and gives it a random one.
        taken = _ability_code(ability, enemy, enemy_index)
        given = _pool_pick(battle, chance, fighter.pool_exclude_mask, fighter.pool_self - 1)
        if given is not None:
            ability[enemy][enemy_index] = given
        ability[side][index] = taken
    sources = [fighter]
    if fighter.form_count:  # Astraeus's sign, The Awakened One's weapon (the last option: two weapons).
        k = chance.pick(fighter.form_count)
        ext['ability2'][side][index] = 0
        if fighter.form_pair_last and k == fighter.form_count - 1:
            a = chance.pick(fighter.form_count - 1)
            b = chance.pick(fighter.form_count - 2)
            b += b >= a
            ability[side][index] = POOL_CODE + fighter.form_first - 1 + a
            ext['ability2'][side][index] = POOL_CODE + fighter.form_first - 1 + b
        else:
            ability[side][index] = POOL_CODE + fighter.form_first - 1 + k
        sources += [battle.pool[code - POOL_CODE] for code in (ability[side][index], ext['ability2'][side][index]) if code >= POOL_CODE]
    for source in sources:
        if source.entry_same_card_stat_bonus:  # Gemini: +50% stats per Astraeus in the deck (IMG_0333: x3 with four).
            count = sum(1 for i in order[side] if battle.teams[side][i].card_id == fighter.card_id)
            factor = 1 + source.entry_same_card_stat_bonus * count
            _scale(hp, max_hp, attack, side, index, factor, factor, rounding)
        if source.entry_shields:
            ext['shields'][side][index] += source.entry_shields  # Virgo
    if fighter.sack:  # The Sack (user): three independent rolls, 1-50% damage, 1-50% max HP, 1-30% damage reduction.
        bonus_attack, bonus_hp, reduction = chance.pick(50) + 1, chance.pick(50) + 1, chance.pick(30) + 1
        _scale(hp, max_hp, attack, side, index, 1 + bonus_hp / 100, 1 + bonus_attack / 100, rounding)
        inscale[side][index] *= 1 - reduction / 100
    if fighter.entry_absorb_behind:  # Cosmic Pop Star: 75% of the card behind's stats; that card is banished.
        behind = next((p for p in range(active[side] + 1, len(order[side])) if hp[side][order[side][p]] > 0), None)
        if behind is not None:
            victim = order[side][behind]
            f = fighter.entry_absorb_behind
            hp[side][index] = _round(hp[side][index] + hp[side][victim] * f, rounding)
            max_hp[side][index] = _round(max_hp[side][index] + max_hp[side][victim] * f, rounding)
            attack[side][index] = _round(attack[side][index] + attack[side][victim] * f, rounding)
            del order[side][behind]
    for _ in range(fighter.summon_copies):  # Fresno: copies with 35% of its stats and no ability (user: they do not summon).
        _summon(battle, hp, max_hp, attack, order, status, side, _round(max_hp[side][index] * fighter.summon_copy_fraction, rounding),
                _round(attack[side][index] * fighter.summon_copy_fraction, rounding), 0)
    for _ in range(fighter.summon_random):  # Nuwa (user): a random card (not Limited, not Glamour) with its stats.
        code = _pool_pick(battle, chance, fighter.pool_exclude_mask)
        if code is not None:
            _summon(battle, hp, max_hp, attack, order, status, side, max_hp[side][index], attack[side][index], code)
    if fighter.entry_global_steal:  # Eonus: 5% of every other card's stats, allies included; decays after 3 turns.
        for s2 in (0, 1):
            for i in order[s2]:
                if (s2, i) != (side, index) and hp[s2][i] > 0:
                    take_hp = _round(hp[s2][i] * fighter.entry_global_steal, rounding)
                    take_attack = _round(attack[s2][i] * fighter.entry_global_steal, rounding)
                    hp[s2][i] -= take_hp
                    max_hp[s2][i] -= take_hp
                    attack[s2][i] -= take_attack
                    hp[side][index] += take_hp
                    max_hp[side][index] += take_hp
                    attack[side][index] += take_attack
                    ext['bonus_hp'][side][index] += take_hp
                    ext['bonus_attack'][side][index] += take_attack
        ext['timer'][side][index] = EONUS_TURNS
    if fighter.entry_ally_stat_multiplier != 1:  # Santa Claws: living allies +50%, then -10% per turn in play.
        for i in order[side]:
            if i != index and hp[side][i] > 0:
                _scale(hp, max_hp, attack, side, i, fighter.entry_ally_stat_multiplier, fighter.entry_ally_stat_multiplier, rounding)
                status[9][side][i] |= SANTA
    if fighter.link_first_two:  # Fate Seamstress: the first 2 living enemies share damage.
        linked = [order[enemy][p] for p in range(active[enemy], len(order[enemy])) if hp[enemy][order[enemy][p]] > 0][:2]
        if len(linked) == 2:
            ext['link'][enemy][linked[0]], ext['link'][enemy][linked[1]] = linked[1] + 1, linked[0] + 1
    if fighter.perish_turns:
        ext['timer'][side][index] = fighter.perish_turns + 1  # Sleep Paralysis: both die at the start of its 4th turn
    if fighter.entry_shuffle_enemy and alive:  # Jersey Devil (IMG_0341): the enemy lineup is shuffled.
        positions = [p for p in range(active[enemy], len(order[enemy])) if hp[enemy][order[enemy][p]] > 0]
        cards = [order[enemy][p] for p in positions]
        for i in range(len(cards) - 1, 0, -1):
            j = chance.pick(i + 1)
            cards[i], cards[j] = cards[j], cards[i]
        for p, card in zip(positions, cards):
            order[enemy][p] = card
        if order[enemy][active[enemy]] != enemy_index and not status[21][enemy][order[enemy][active[enemy]]]:
            _entry(battle, hp, max_hp, attack, active, order, status[21], stolen, status, enemy, rounding, trace, chance)
    _cleanse(battle, hp, order, status)


def _initial(battle, options, trace, chance=None):
    hp = [[float(f.hp) for f in team] for team in battle.teams]
    max_hp = [row[:] for row in hp]
    attack = [[float(f.attack) for f in team] for team in battle.teams]
    used = [[False for _ in team] for team in battle.teams]
    survivals = [[0 for _ in team] for team in battle.teams]
    completed = [[0 for _ in team] for team in battle.teams]
    order = [list(range(n)) for n in battle.sizes()]
    entered = [[False for _ in team] for team in battle.teams]
    dodges = [[0 for _ in team] for team in battle.teams]
    stolen = [[False for _ in team] for team in battle.teams]
    turns, hits, own_dodges, burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked = (
        [[0 for _ in team] for team in battle.teams] for _ in range(16))
    inscale = [[1.0 for _ in team] for team in battle.teams]
    stored = [[0.0 for _ in team] for team in battle.teams]
    poison = [[0 for _ in team] for team in battle.teams]
    poison_damage = [[0.0 for _ in team] for team in battle.teams]
    weakness = [[1.0 for _ in team] for team in battle.teams]
    triggered, undying, doom, strike_timer, blind = ([[0 for _ in team] for team in battle.teams] for _ in range(5))
    strike_damage = [[0.0 for _ in team] for team in battle.teams]
    block_pool = [[0.0 for _ in team] for team in battle.teams]
    death_timer = [[0 for _ in team] for team in battle.teams]
    fossils = [0, 0]
    ability = [[0 for _ in team] for team in battle.teams]
    suppressed = [0, 0]
    ext = _new_ext(battle)
    status = (burn, bleed, frozen, slowed, frostbite, confused, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, entered, inscale)
    active = [0, 0]
    for side in (0, 1):
        # Longmu's Mother of Dragons: dragons in its deck are shielded from attacks from the start.
        for index in order[side]:
            card = battle.teams[side][index]
            if card.team_class_shields:
                for ally in order[side]:
                    if ally != index and battle.teams[side][ally].class_mask & card.ally_class_mask:
                        ext['shields'][side][ally] += card.team_class_shields
    _entry(battle, hp, max_hp, attack, active, order, entered, stolen, status, 0, options.stat_rounding, trace, chance)
    _entry(battle, hp, max_hp, attack, active, order, entered, stolen, status, 1, options.stat_rounding, trace, chance)
    queue = tuple(side for side in (battle.first_side, 1-battle.first_side)
                  if _queues_entry_hit(battle, status, order, active, hp, side))
    return _freeze(hp, max_hp, attack, used, survivals, completed, order, entered, dodges, stolen, turns, hits, own_dodges, burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked, inscale, stored, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, active, battle.first_side, 0,
                   entry_queue=queue)


def _on_death(battle, hp, max_hp, attack, order, active, dodges, side, dead, rounding, trace, status=None, chance=None, stored=None,
              in_deck=False):
    """Call after advancing active past a dead fighter. True Prophet (user-stated
    on-death ability) grants its dodge to the card that comes in next; stat
    gifts add a fraction of the dead card's max HP/ATK (extrapolated)."""
    fighter = battle.teams[side][dead]
    enemy = 1 - side
    if status is not None and (fighter.dinosaur or fighter.death_fossils):
        status[17][side] += fighter.dinosaur + fighter.death_fossils
        _event(trace, "FOSSILS", side=side, count=status[17][side])
    if fighter.death_frostbite_turns and status is not None and active[enemy] < len(order[enemy]):
        killer = order[enemy][active[enemy]]
        if hp[enemy][killer] > 0:  # Frosty: the surviving enemy card is frostbitten.
            status[4][enemy][killer] = max(status[4][enemy][killer], fighter.death_frostbite_turns)
    if fighter.death_ally_stat_multiplier != 1:
        for q in range(active[side], len(order[side])):
            ally = order[side][q]
            if ally != dead and hp[side][ally] > 0:
                _scale(hp, max_hp, attack, side, ally, fighter.death_ally_stat_multiplier, fighter.death_ally_stat_multiplier, rounding)
    if status is not None:
        _death_effects(battle, hp, max_hp, attack, order, active, side, dead, rounding, trace, status, chance or _Chance(), in_deck)
    # User: a front card's effects go to the next card in the lineup, dead or alive (so they can
    # be wasted on a card killed in the deck). A card killed in the deck gives them to the card
    # still playing (user: Arthur used True Prophet's grant after Prophet died waiting).
    if in_deck:
        heir_position = next((q for q in range(active[side], len(order[side]))
                              if order[side][q] != dead and hp[side][order[side][q]] > 0), None)
    else:
        heir_position = active[side] if active[side] < len(order[side]) else None
    if heir_position is None:
        return
    heir = order[side][heir_position]
    passed = fighter.next_card_dodges + dodges[side][dead]  # Unused grants pass on too.
    dodges[side][dead] = 0
    if passed:
        dodges[side][heir] += passed
        _event(trace, "ON_DEATH", side=side, effect="grant_dodge", slot=heir_position, charges=passed)
    add_hp = max_hp[side][dead] * (fighter.next_card_stat_add_fraction + fighter.next_card_max_hp_add_fraction)
    add_attack = attack[side][dead] * fighter.next_card_stat_add_fraction
    if (add_hp or add_attack) and fighter.next_card_gift_chance < 1 and not (chance or _Chance()).roll(fighter.next_card_gift_chance):
        add_hp = add_attack = 0.0
    if stored is not None and stored[side][dead]:
        attack[side][heir] = _round(attack[side][heir] + stored[side][dead], rounding)
    if fighter.next_card_stat_multiplier != 1:
        _scale(hp, max_hp, attack, side, heir, fighter.next_card_stat_multiplier, fighter.next_card_stat_multiplier, rounding)
        _event(trace, "ON_DEATH", side=side, effect="scale_stats", slot=heir_position, hp=hp[side][heir], attack=attack[side][heir])
    if fighter.next_card_freeze_turns and status is not None:
        status[2][side][heir] = max(status[2][side][heir], fighter.next_card_freeze_turns)
    if fighter.next_card_boost_multiplier != 1 and status is not None:
        status[9][side][heir] |= SAM_BOOST  # Uncle Sam: the next card deals 5x and dies after attacking.
    if add_hp or add_attack:
        hp[side][heir] = _round(hp[side][heir] + add_hp, rounding)
        max_hp[side][heir] = _round(max_hp[side][heir] + add_hp, rounding)
        attack[side][heir] = _round(attack[side][heir] + add_attack, rounding)
        _event(trace, "ON_DEATH", side=side, effect="gift_stats", slot=heir_position, hp=add_hp, attack=add_attack)


def _field_multiplier(battle, order, active, hp):
    """Product of field damage boosts from every living card on both teams."""
    product = 1.0
    for side in (0, 1):
        for position in range(active[side], len(order[side])):
            index = order[side][position]
            if hp[side][index] > 0:
                product *= battle.teams[side][index].field_damage_multiplier
    return product


def _scale_allies(battle, hp, max_hp, attack, order, active, side, factor, rounding):
    """Scale every living ally behind the active card."""
    for position in range(active[side] + 1, len(order[side])):
        index = order[side][position]
        if hp[side][index] > 0:
            _scale(hp, max_hp, attack, side, index, factor, factor, rounding)


def _counts_turns(fighter):
    return bool(fighter.alternate_rest or fighter.periodic_block_period or fighter.recharge_turns or fighter.first_turn_actions or fighter.actions_growth_per_turn
                or fighter.first_attack_multiplier != 1 or fighter.action_end_stat_turns or fighter.periodic_stat_period
                or fighter.periodic_attack_period or fighter.turn_start_stat_growth or fighter.turn_start_incoming_reduction
                or fighter.growth_turns or fighter.awaken_turn or fighter.shield_period)


def _allowance(fighter, turn_index, faded=0, low=False):
    """Normal actions in the fighter's own turn `turn_index` (0-based)."""
    if fighter.low_hp_extra_actions and low:
        return min(16, _allowance(fighter, turn_index, faded) + fighter.low_hp_extra_actions)
    if fighter.fade_extra_actions and not faded:
        return min(16, _allowance(fighter, turn_index, 1) + fighter.fade_extra_actions)
    turn_index = max(0, turn_index)
    if fighter.first_turn_actions and turn_index == 0:
        return fighter.first_turn_actions
    return min(16, fighter.actions_per_turn + fighter.actions_growth_per_turn * turn_index)


def _periodic(fighter, turns_done):
    return bool(fighter.periodic_attack_period) and (turns_done + 1) % fighter.periodic_attack_period == 0


def _action_hits(fighter, turns_done):
    """Hits in a normal action: the periodic special's count, plus a follow-up hit."""
    hits = fighter.periodic_attack_hits if _periodic(fighter, turns_done) and fighter.periodic_attack_hits else fighter.attacks_per_action
    return hits + (1 if fighter.followup_multiplier else 0)


def _blast(battle, hp, max_hp, side, dead, victim_side, victim, chance, options, trace):
    """Savior / Frankenstein / Valentine's Specter: on death, the enemy front card loses a
    share of the dead card's max HP (direct loss, extrapolated)."""
    fighter = battle.teams[side][dead]
    if fighter.death_damage_max_hp_fraction and hp[victim_side][victim] > 0 and chance.roll(fighter.death_damage_chance):
        loss = _round(max_hp[side][dead] * fighter.death_damage_max_hp_fraction, options.rounding)
        hp[victim_side][victim] -= loss
        _event(trace, "DEATH_BLAST", side=side, damage=loss, hp=hp[victim_side][victim])


def _has_entry_hit(fighter, position):
    return bool(fighter.entry_hit_multiplier) and (not fighter.entry_hit_requires_fallen or position > 0)


def _bench_hits(battle, hp, max_hp, attack, order, active, dodges, survivals, side, amount, as_attack, options, trace,
                status, chance, stored, hits=None, limit=None, on_death_effects=True, positions=None):
    """Damage every waiting card of `side`. Attacks can be dodged, absorbed by Prophet
    grants, invincibility or lethal survival; plain HP loss cannot. Cards reduced to 0 HP
    die in the deck (user Seraphim tests); their on-death effects go to the next card to play. Without
    on_death_effects (Ra, IMG_0325) lethal survival, reflection and on-death effects are all ignored."""
    killed = []
    end = len(order[side]) if limit is None else min(len(order[side]), active[side] + 1 + limit)
    for position in (range(active[side] + 1, end) if positions is None else positions):
        index = order[side][position]
        card = battle.teams[side][index]
        if hp[side][index] <= 0:
            continue
        if as_attack:
            if chance.roll(card.dodge_probability) or card.invincible:
                continue
            if card.alternate_dodge and hits is not None:
                # User: Deus Ex always dodges the entry while waiting; deck hits count toward the alternation.
                hits[side][index] += 1
                if hits[side][index] % 2 == card.alternate_dodge % 2:
                    _event(trace, "BENCH_DODGE", side=side, slot=position)
                    continue
            if dodges[side][index]:
                dodges[side][index] -= 1
                continue
            damage = _round(amount * card.incoming_multiplier, options.rounding)
            if on_death_effects and damage >= hp[side][index] and survivals[side][index] < card.lethal_survivals:
                survivals[side][index] += 1
                hp[side][index] = (max(1.0, _round(max_hp[side][index] * card.lethal_survival_hp_fraction, options.stat_rounding))
                                   if card.lethal_survival_hp_fraction else 1.0)
                _event(trace, "LETHAL_REPLACEMENT", side=side, slot=position, effect="survive_at_one_hp", uses=survivals[side][index])
                continue
        else:
            damage = _round(amount, options.rounding)
        if as_attack and status is not None and status[20]['shields'][side][index]:
            status[20]['shields'][side][index] -= 1  # A shield blocks the whole attack.
            continue
        if (on_death_effects and 0 < hp[side][index] <= damage and card.death_reflects and survivals[side][index] >= card.lethal_survivals
                and survivals[side][index] - card.lethal_survivals < card.death_reflects):
            # User Seraphim tests: Parallax hit in the deck reflects its death onto the attacker.
            survivals[side][index] += 1
            killer_side = 1 - side
            killer = order[killer_side][active[killer_side]]
            hp[killer_side][killer] = min(hp[killer_side][killer], 0.0)
            _event(trace, "LETHAL_REPLACEMENT", side=side, slot=position, effect="death_reflect")
            continue
        before = hp[side][index]
        _ability_hit(battle, hp, attack, order, active, side, index, damage, options.rounding)
        _event(trace, "BENCH_HIT", side=side, slot=position, damage=damage, hp=hp[side][index])
        if hp[side][index] <= 0 < before:
            killed.append((position, index))
    for position, index in killed:
        _event(trace, "DEATH", side=side, slot=position, reason="bench")
        if on_death_effects:
            _blast(battle, hp, max_hp, side, index, 1 - side, order[1 - side][active[1 - side]], chance, options, trace)
            _on_death(battle, hp, max_hp, attack, order, active, dodges, side, index, options.stat_rounding, trace, status, chance,
                      stored, in_deck=True)


def _transfer(hp, max_hp, attack, source, target, fraction, rounding, trace, phase):
    """Move a fraction of the source card's current HP/max HP/ATK to the target."""
    (ss, si), (ts, ti) = source, target
    take_hp = _round(hp[ss][si] * fraction, rounding)
    take_attack = _round(attack[ss][si] * fraction, rounding)
    hp[ss][si] = _round(hp[ss][si] - take_hp, rounding)
    max_hp[ss][si] = _round(max_hp[ss][si] - take_hp, rounding)
    attack[ss][si] = _round(attack[ss][si] - take_attack, rounding)
    hp[ts][ti] = _round(hp[ts][ti] + take_hp, rounding)
    max_hp[ts][ti] = _round(max_hp[ts][ti] + take_hp, rounding)
    attack[ts][ti] = _round(attack[ts][ti] + take_attack, rounding)
    _event(trace, phase, side=ts, from_side=ss, hp=take_hp, attack=take_attack)


def _scale(hp, max_hp, attack, side, slot, hp_factor, attack_factor, rounding):
    hp[side][slot] = _round(hp[side][slot] * hp_factor, rounding)
    max_hp[side][slot] = _round(max_hp[side][slot] * hp_factor, rounding)
    attack[side][slot] = _round(attack[side][slot] * attack_factor, rounding)


def _poison(status, side, index, turns, amount):
    """Re-poisoning refreshes the duration and keeps the stronger tick (extrapolated)."""
    if turns and amount > 0:
        status[6][side][index] = max(status[6][side][index], turns)
        status[7][side][index] = max(status[7][side][index], amount)


def _hit_effects(hp, max_hp, attack, status, actor, a, enemy, d, attacker, target, loss, options, trace):
    """Extrapolated on-hit reactions after a damaging hit, before counters.
    Attacker effects first, then defender reactions, then retaliation damage."""
    r = options.stat_rounding
    # Statuses refresh (max of remaining and new duration); burn and bleed stack.
    burn, bleed = status[0], status[1]
    burn[enemy][d] = max(burn[enemy][d], attacker.hit_burn_turns)
    bleed[enemy][d] = max(bleed[enemy][d], attacker.hit_bleed_turns)
    burn[actor][a] = max(burn[actor][a], target.burn_on_attacked_turns)
    bleed[actor][a] = max(bleed[actor][a], target.bleed_on_attacked_turns)
    _poison(status, enemy, d, POISON_PERMANENT if attacker.poison_permanent and attacker.hit_poison_turns else attacker.hit_poison_turns,
            max_hp[enemy][d] * attacker.poison_target_max_hp_fraction if attacker.poison_target_max_hp_fraction
            else attack[actor][a] * attacker.poison_fraction)
    status[8][enemy][d] = max(status[8][enemy][d], attacker.hit_weakness_multiplier)
    if attacker.steal_first_attacked and not status[9][actor][a] & 8:
        status[9][actor][a] |= 8  # Hecate: takes the first damaged card's ability; the victim loses it.
        status[18][actor][a] = _ability_code(status[18], enemy, d)
        status[18][enemy][d] = ABILITY_NONE
    if attacker.hit_disable:
        status[18][enemy][d] = ABILITY_NONE  # Set.
    if attacker.fossil_death_timer and not status[16][enemy][d] and hp[enemy][d] > 0:
        status[16][enemy][d] = max(1, attacker.fossil_death_timer - min(status[17][actor], FOSSIL_DEATH_TIMER_CAP))
    threshold = max_hp[enemy][d] * target.low_hp_trigger_threshold
    if 0 < hp[enemy][d] < threshold <= hp[enemy][d] + loss:
        if target.low_hp_stun_turns and not status[9][enemy][d] & 4:
            status[9][enemy][d] |= 4  # Banshee (user): stuns only once.
            status[3][actor][a] = max(status[3][actor][a], target.low_hp_stun_turns)
    if target.low_hp_heal_fraction and not status[9][enemy][d] & 1 and 0 < hp[enemy][d] < threshold:
        status[9][enemy][d] |= 1
        _heal(hp, max_hp, enemy, d, max_hp[enemy][d] * target.low_hp_heal_fraction, options, trace, "LOW_HP_HEAL")
    _poison(status, actor, a, target.attacked_poison_turns, attack[enemy][d] * target.poison_fraction)
    if attacker.hit_attack_gain_fraction:
        attack[actor][a] = _round(attack[actor][a] + loss * attacker.hit_attack_gain_fraction, r)
    if attacker.hit_attack_drain_fraction:
        taken = min(attack[enemy][d], _round(loss * attacker.hit_attack_drain_fraction, r))
        attack[enemy][d] = _round(attack[enemy][d] - taken, r)
        attack[actor][a] = _round(attack[actor][a] + taken, r)
    if attacker.hit_hp_gain_fraction and hp[actor][a] > 0:
        hp[actor][a] = _round(hp[actor][a] + loss * attacker.hit_hp_gain_fraction, r)
        max_hp[actor][a] = max(max_hp[actor][a], hp[actor][a])
    if hp[enemy][d] > 0:
        if attacker.hit_target_attack_reduction_fraction:
            attack[enemy][d] = max(0.0, _round(attack[enemy][d] - loss * attacker.hit_target_attack_reduction_fraction, r))
        if attacker.hit_target_max_hp_reduction_fraction:
            max_hp[enemy][d] = max(hp[enemy][d], _round(max_hp[enemy][d] - loss * attacker.hit_target_max_hp_reduction_fraction, r))
        if attacker.hit_target_attack_multiplier != 1:
            attack[enemy][d] = _round(attack[enemy][d] * attacker.hit_target_attack_multiplier, r)
        if attacker.hit_steal_fraction:
            _transfer(hp, max_hp, attack, (enemy, d), (actor, a), attacker.hit_steal_fraction, r, trace, "HIT_STEAL")
        if target.damaged_attack_multiplier != 1:
            attack[enemy][d] = _round(attack[enemy][d] * target.damaged_attack_multiplier, r)
        if target.damaged_stat_multiplier != 1:
            _scale(hp, max_hp, attack, enemy, d, target.damaged_stat_multiplier, target.damaged_stat_multiplier, r)
        if target.damaged_heal_max_hp_fraction:
            _heal(hp, max_hp, enemy, d, max_hp[enemy][d] * target.damaged_heal_max_hp_fraction, options, trace, "DAMAGED_HEAL")
        if target.attacked_steal_fraction:
            _transfer(hp, max_hp, attack, (actor, a), (enemy, d), target.attacked_steal_fraction, r, trace, "ATTACKED_STEAL")
    retaliation = _round(attack[enemy][d] * target.thorns_attack_fraction + loss * target.reflect_damage_fraction,
                         options.rounding) if target.thorns_attack_fraction or target.reflect_damage_fraction else 0.0
    if retaliation:
        hp[actor][a] -= retaliation
        _event(trace, "RETALIATION", side=actor, damage=retaliation, hp=hp[actor][a])
    if attacker.recoil_damage_fraction:
        recoil = _round(loss * attacker.recoil_damage_fraction, options.rounding)
        hp[actor][a] -= recoil
        _event(trace, "RECOIL", side=actor, damage=recoil, hp=hp[actor][a])
    if 0 < hp[enemy][d] < max_hp[enemy][d] * attacker.execute_after_below_max_hp:
        hp[enemy][d] = 0.0
        _event(trace, "EXECUTE", side=enemy)
    if 0 < hp[enemy][d] < max_hp[enemy][d] * target.mutual_destruction_below_max_hp:
        hp[enemy][d] = hp[actor][a] = 0.0
        _event(trace, "MUTUAL_DESTRUCTION", side=enemy)


def _round(value, policy):
    if policy == "unrounded":
        return value
    if policy == "ceil":
        return float(math.ceil(value))
    if policy == "floor":
        return float(math.floor(value))
    return float(math.floor(value + .5))


def _heal(hp, max_hp, side, slot, raw_amount, options, trace, phase):
    """Capped healing; dead cards cannot receive healing or revive through it."""
    before = hp[side][slot]
    if before <= 0:
        return
    if not math.isfinite(raw_amount):
        raise ValueError("Healing calculation overflowed")
    amount = _round(raw_amount, options.rounding)
    after = before + amount
    if not math.isfinite(after):
        raise ValueError("Healing calculation overflowed")
    hp[side][slot] = max(before, min(after, max_hp[side][slot]))
    _event(trace, phase, side=side, slot=slot, before=before, amount=amount,
           healed=hp[side][slot]-before, hp=hp[side][slot], max_hp=max_hp[side][slot])


SAM_BOOST_MULTIPLIER = 5.0  # Uncle Sam (user Part A): the next card deals 5x and dies after attacking.
EONUS_TURNS = 3
LUMINANT_STEP, LUMINANT_FLOOR, LUMINANT_MAX, LUMINANT_GAIN, LUMINANT_CAP = 0.1, 0.2, 2, 0.1, 3.0
BUDDHA_FRACTION = 0.5


def _ability_hit(battle, hp, attack, order, active, side, index, amount, rounding):
    """Damage from an ability rather than an attack. Marrowclaw cannot be killed this way; Juggernoid
    reflects a share to the enemy card in play (at most ability_reflect_cap x its ATK)."""
    if amount <= 0:
        return
    card = battle.teams[side][index]
    before = hp[side][index]
    hp[side][index] = before - amount
    if card.ability_death_immune and before > 0 and hp[side][index] < 1:
        hp[side][index] = min(before, 1.0)
    if card.ability_reflect_fraction:
        enemy = 1 - side
        if active[enemy] < len(order[enemy]):
            other = order[enemy][active[enemy]]
            if hp[enemy][other] > 0:
                hp[enemy][other] -= _round(min(amount * card.ability_reflect_fraction,
                                               card.ability_reflect_cap * attack[side][index]), rounding)


def _has_status(status, side, index):
    return bool(status[0][side][index] or status[1][side][index] or status[2][side][index] or status[3][side][index]
                or status[4][side][index] or status[5][side][index] or status[6][side][index] or status[8][side][index] > 1
                or status[11][side][index] or status[14][side][index] or status[16][side][index]
                or status[20]['curse'][side][index])


def _cleanse(battle, hp, order, status):
    """Serket: while it is alive in the party, its whole side is immune to statuses (cleared as they land)."""
    for side in (0, 1):
        if not any(hp[side][i] > 0 and battle.teams[side][i].team_status_immunity for i in order[side]):
            continue
        for i in order[side]:
            for k in (0, 1, 2, 4, 5, 6, 11, 14, 16):
                status[k][side][i] = 0
            status[7][side][i] = 0.0
            status[8][side][i] = 1.0
            status[20]['curse'][side][i] = 0


def _waiting(hp, order, active, side):
    return [p for p in range(active[side] + 1, len(order[side])) if hp[side][order[side][p]] > 0]


def _pool_pick(battle, chance, exclude, skip=-1, other=-1):
    """A uniformly random ability from the pool (never a form; never `skip` or `other`)."""
    eligible = [i for i, f in enumerate(battle.pool) if not f.pool_flags & (exclude | POOL_FORM) and i != skip and i != other]
    if not eligible:
        return None
    return POOL_CODE + eligible[chance.pick(len(eligible))]


def _summon(battle, hp, max_hp, attack, order, status, side, new_hp, new_attack, code, position=None):
    """Put a free spare slot into the lineup (back, or at `position`); None when none is left."""
    triggered = status[9]
    index = next((i for i, card in enumerate(battle.teams[side])
                  if card.spare and i not in order[side] and not triggered[side][i] & SPENT), None)
    if index is None:
        return None
    triggered[side][index] |= SPENT
    hp[side][index] = max_hp[side][index] = float(new_hp)
    attack[side][index] = float(new_attack)
    status[18][side][index] = code
    status[21][side][index] = False
    if position is None:
        order[side].append(index)
    else:
        order[side].insert(position, index)
    return index


def _relocate(order, active, side, index):
    """Move a card to the back of its lineup (revivals, Buddha)."""
    position = order[side].index(index)
    del order[side][position]
    if position < active[side]:
        active[side] -= 1
    order[side].append(index)


def _restore(hp, status, side, index, new_hp):
    hp[side][index] = new_hp
    for k in (0, 1, 2, 3, 4, 5, 6, 10, 11, 14, 16):
        status[k][side][index] = 0
    status[7][side][index] = 0.0
    status[8][side][index] = 1.0
    status[20]['curse'][side][index] = 0


def _luminant(battle, hp, order, side, index):
    """Index of an Eclipseborn Luminant alive in the party (not the defending card itself)."""
    return next((i for i in order[side] if i != index and hp[side][i] > 0 and battle.teams[side][i].luminant_evasion), None)


def _buddha(battle, hp, max_hp, order, active, status, side, index, options, trace):
    """Buddha (user): instead of attacking, revive the dead allies one per turn in setup order (full HP, to the back)
    while staying in front; with none dead, heal the weakest other card by 50% of its max HP (never itself, user)
    and move to the back once a heal fully heals it (or every other living card is already full)."""
    dead = next((i for i in sorted(order[side]) if i != index and hp[side][i] <= 0), None)
    if dead is not None:
        _restore(hp, status, side, dead, max_hp[side][dead])  # User: Buddha revives at the back with full HP.
        _relocate(order, active, side, dead)
        _event(trace, "REVIVE", side=side, by="buddha")
        return False
    living = [i for i in order[side][active[side]:] if i != index and hp[side][i] > 0]
    weakest = None
    for i in living:
        if weakest is None or hp[side][i] / max_hp[side][i] < hp[side][weakest] / max_hp[side][weakest]:
            weakest = i
    healed_full = False
    if weakest is not None and hp[side][weakest] < max_hp[side][weakest]:
        _heal(hp, max_hp, side, weakest, max_hp[side][weakest] * BUDDHA_FRACTION, options, trace, "BUDDHA_HEAL")
        healed_full = hp[side][weakest] >= max_hp[side][weakest]
    if living and (healed_full or all(hp[side][i] >= max_hp[side][i] for i in living)):
        _relocate(order, active, side, index)
        return True
    return False


def _death_effects(battle, hp, max_hp, attack, order, active, side, dead, rounding, trace, status, chance, in_deck):
    """Batch-12 death triggers, in this order: The Broken One's split, Naga's curse, Anubis / Bloody Mary,
    Control Freak, Anubis & Hades, the Flying Dutchman's return, Time Lord Stryx."""
    fighter = battle.teams[side][dead]
    enemy = 1 - side
    ext, triggered, ability = status[20], status[9], status[18]
    if fighter.split_on_death:
        position = order[side].index(dead) + 1 if in_deck else active[side]
        for k in range(2):  # Joy then Sorrow, no abilities, 50% max HP / 60% ATK of the card when it died (user).
            if _summon(battle, hp, max_hp, attack, order, status, side, _round(max_hp[side][dead] * SPLIT_HP, rounding),
                       _round(attack[side][dead] * SPLIT_ATTACK, rounding), 0, position + k) is None:
                break
    if fighter.death_curse_turns and active[enemy] < len(order[enemy]):
        other = order[enemy][active[enemy]]
        if hp[enemy][other] > 0:
            ext['curse'][enemy][other] = max(ext['curse'][enemy][other], fighter.death_curse_turns)
    for index in list(order[side]):
        card = battle.teams[side][index]
        if (index != dead and card.revive_on_ally_death_hp and hp[side][index] <= 0
                and chance.roll(card.revive_on_ally_death_chance)):
            _restore(hp, status, side, index, max(1.0, _round(max_hp[side][index] * card.revive_on_ally_death_hp, rounding)))
            _relocate(order, active, side, index)  # User: Anubis comes back at the back.
            _event(trace, "REVIVE", side=side, by="ally_death")
    if any(hp[enemy][i] > 0 and battle.teams[enemy][i].recruit_defeated for i in order[enemy]):
        base = battle.teams[side][dead]  # Control Freak: the defeated card joins its deck with its own stats.
        _summon(battle, hp, max_hp, attack, order, status, enemy, base.hp, base.attack, _ability_code(ability, side, dead))
    for index in list(order[enemy]):
        card = battle.teams[enemy][index]
        if not card.enemy_death_revive_fraction:
            continue
        target = index if hp[enemy][index] <= 0 else next((i for i in order[enemy] if hp[enemy][i] <= 0), None)
        if target is None:
            continue
        fraction = card.enemy_death_revive_fraction  # Anubis & Hades (IMG_0311): itself first, with 75% of its stats.
        max_hp[enemy][target] = _round(max_hp[enemy][index] * fraction, rounding)
        attack[enemy][target] = _round(attack[enemy][index] * fraction, rounding)
        _restore(hp, status, enemy, target, max_hp[enemy][target])
        _relocate(order, active, enemy, target)
        _event(trace, "REVIVE", side=enemy, by="enemy_death")
    if triggered[side][dead] & DUTCH_ALLY and not in_deck:
        triggered[side][dead] &= ~DUTCH_ALLY
        for index in order[side]:
            if triggered[side][index] & DUTCH_AWAY and hp[side][index] > 0:
                triggered[side][index] &= ~DUTCH_AWAY  # User: the Dutchman swaps back when its ally dies.
                order[side].remove(index)
                order[side].insert(active[side], index)
                break
    if fighter.revive_party_on_death and not triggered[side][dead] & PARTY_REVIVED:
        triggered[side][dead] |= PARTY_REVIVED  # Time Lord Stryx (IMG_0310): the party returns in its initial order.
        lineup = battle.sizes()[side]
        first = [i for i in range(lineup) if i in order[side]]
        rest = [i for i in order[side] if i >= lineup]
        for i in first:
            if hp[side][i] <= 0:
                _restore(hp, status, side, i, max_hp[side][i])
        order[side][:] = first + rest
        active[side] = 0
        _event(trace, "REVIVE", side=side, by="party")


def _advance(battle, state, dodge, options, trace, critical=False, chance=None):
    (hp, max_hp, attack, used, survivals, completed, order, entered, dodges, stolen, turns, hits, own_dodges,
     burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked,
     inscale, stored, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, ability) = [list(map(list, data)) for data in (
        state.hp, state.max_hp, state.attack, state.used_block, state.survivals_used, state.completed_actions,
        state.order, state.entered, state.granted_dodges, state.stolen_from, state.turns, state.hits_taken,
        state.own_dodges_used, state.burn, state.bleed, state.frozen, state.slowed, state.frostbite, state.confused,
        state.dodge_charged, state.revived, state.guarded, state.faded, state.resting, state.debuffed, state.marked,
        state.incoming_scale, state.stored, state.poison, state.poison_damage, state.weakness, state.triggered, state.undying, state.doom, state.strike_timer, state.strike_damage, state.blind, state.block_pool, state.death_timer, state.ability)]
    fossils = list(state.fossils)
    suppressed = list(state.suppressed)
    ext = _thaw_ext(state.ext)
    status = (burn, bleed, frozen, slowed, frostbite, confused, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, entered, inscale)
    chance = chance or _Chance()
    base_battle = battle
    failed = [0, 0]
    for side in (0, 1):
        p = _fail_chance(base_battle, ext, 1 - side)
        if p and state.active[side] < len(state.order[side]) and state.hp[side][state.order[side][state.active[side]]] > 0:
            failed[side] = int(chance.roll(p))  # End Times: the enemy card's ability fails for this step.
    battle = _view(base_battle, ability, suppressed, state.order, state.active, state.hp, ext['ability2'], failed)
    active, normal_actor = list(state.active), state.actor
    is_counter = bool(state.counter_pending)
    is_entry = bool(state.entry_queue) and not is_counter
    entry_context = is_entry or (is_counter and state.counter_from_entry)
    queue = list(state.entry_queue)
    actor = _hit_actor(state)
    if is_entry:
        queue.pop(0)
    enemy = 1 - actor
    a, d = order[actor][active[actor]], order[enemy][active[enemy]]
    attacker = battle.teams[actor][a]
    target = battle.teams[enemy][d]
    # A card dies only when HP is actually lowered to <= 0: cards already at 0 HP
    # (hit while waiting) can act and survive hits that do not damage them.
    actor_hp_start, enemy_hp_start = hp[actor][a], hp[enemy][d]
    total_hits = 1 if is_counter or is_entry else _action_hits(attacker, turns[actor][a])
    if attacker.random_hits_max and not is_counter and not is_entry:
        # The Curse: 2-5 slashes, equally likely, decided on the first.
        total_hits = state.hits_remaining + 1 if state.hits_remaining else (
            attacker.random_hits_min + chance.pick(attacker.random_hits_max - attacker.random_hits_min + 1))
    remaining = 1 if is_counter or is_entry else state.hits_remaining or total_hits
    hit_index = 1 if is_counter or is_entry else total_hits - remaining + 1
    periodic = not is_counter and not is_entry and _periodic(attacker, turns[actor][a])
    followup_hit = bool(attacker.followup_multiplier) and not is_counter and not is_entry and hit_index == total_hits
    skipped = False
    relocated = []
    reflect = False
    if not is_counter and not is_entry and not state.hits_remaining and not state.turn_actions_remaining:
        own = base_battle.teams[actor][a]
        changed_ability = shifted = False
        if own.turn_start_random_ability and ability[actor][a] != ABILITY_NONE and not triggered[actor][a] & SHIFTED:
            code = _pool_pick(base_battle, chance, own.pool_exclude_mask, own.pool_self - 1)  # Glamour (user): changes each turn.
            if code is not None:
                ability[actor][a] = code
                ext['ability2'][actor][a] = 0
                changed_ability = shifted = True
        if attacker.awaken_turn and turns[actor][a] == attacker.awaken_turn - 1 and not triggered[actor][a] & AWAKENED:
            triggered[actor][a] |= AWAKENED  # Demon / Immortal Cultivator awaken on their 3rd turn (IMG_0327).
            ability[actor][a] = POOL_CODE + attacker.transform_form - 1
            changed_ability = True
        if changed_ability:
            battle = _view(base_battle, ability, suppressed, order, active, hp, ext['ability2'], failed)
            if shifted:
                # IMG_0330: each new form's entry effects apply to Glamour's current stats (Serpent Mist 7,680 -> 23,040,
                # Hard Claws -40% on Malik), and its entry attack comes first (Slum Dweller).
                _entry(battle, hp, max_hp, attack, active, order, entered, stolen, status, actor, options.stat_rounding, trace, chance,
                       body_only=True)
                battle = _view(base_battle, ability, suppressed, order, active, hp, ext['ability2'], failed)
                if _has_entry_hit(battle.teams[actor][a], active[actor]):
                    triggered[actor][a] |= SHIFTED
                    return _freeze(hp, max_hp, attack, used, survivals, completed, order, entered, dodges, stolen, turns, hits, own_dodges, burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked, inscale, stored, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, active,
                                   state.actor, state.pair_actions, entry_queue=queue + [actor], turn_actions_remaining=0), None
            attacker, target = battle.teams[actor][a], battle.teams[enemy][d]
    if not is_counter and not is_entry and not state.hits_remaining:
        r = options.stat_rounding
        _cleanse(battle, hp, order, status)
        if target.enemy_turn_start_attack_multiplier != 1 or target.enemy_turn_start_stat_multiplier != 1:
            _scale(hp, max_hp, attack, actor, a, target.enemy_turn_start_stat_multiplier,
                   target.enemy_turn_start_attack_multiplier * target.enemy_turn_start_stat_multiplier, r)
        drain = (hp[actor][a] * (target.enemy_turn_start_hp_loss_fraction + attacker.self_turn_start_hp_loss_fraction)
                 + max_hp[actor][a] * (target.enemy_turn_start_max_hp_loss_fraction + attacker.self_turn_start_max_hp_loss_fraction))
        if drain:
            _ability_hit(battle, hp, attack, order, active, actor, a, _round(drain, options.rounding), options.rounding)
            _event(trace, "TURN_START_DRAIN", side=actor, damage=_round(drain, options.rounding), hp=hp[actor][a])
        if not state.turn_actions_remaining:
            # Poison ticks at the start of the poisoned side's turn (IMG_0297, IMG_0300, IMG_0305),
            # for waiting cards too (Black Plague).
            for position, index in [(p, order[actor][p]) for p in range(active[actor] + 1, len(order[actor]))]:
                if poison[actor][index] and hp[actor][index] > 0:
                    poison[actor][index] -= 1
                    tick = _round(poison_damage[actor][index], options.rounding)
                    before = hp[actor][index]
                    _ability_hit(battle, hp, attack, order, active, actor, index, tick, options.rounding)
                    _event(trace, "STATUS_TICK", side=actor, slot=position, status="poison", damage=tick, hp=hp[actor][index])
                    if hp[actor][index] <= 0 < before:
                        _event(trace, "DEATH", side=actor, slot=position, reason="bench")
                        _on_death(battle, hp, max_hp, attack, order, active, dodges, actor, index, options.stat_rounding, trace,
                                  status, chance, stored, in_deck=True)
            if poison[actor][a]:
                poison[actor][a] -= 1
                tick = _round(poison_damage[actor][a], options.rounding)
                _ability_hit(battle, hp, attack, order, active, actor, a, tick, options.rounding)
                _event(trace, "STATUS_TICK", side=actor, status="poison", damage=tick, hp=hp[actor][a])
            if ext['curse'][actor][a]:  # The Hanged Man / Naga (user): 25% max HP at the start of each turn.
                if ext['curse'][actor][a] < CURSE_PERMANENT:
                    ext['curse'][actor][a] -= 1
                _ability_hit(battle, hp, attack, order, active, actor, a, _round(max_hp[actor][a] * CURSE_MAX_HP_FRACTION, options.rounding),
                             options.rounding)
            if attacker.dad_card_id:  # Milk (user): gives its stats to the Dad cards and dies; without Dad it fights.
                dads = [i for i in order[actor] if i != a and hp[actor][i] > 0 and battle.teams[actor][i].card_id == attacker.dad_card_id]
                for i in dads:
                    hp[actor][i] += hp[actor][a]
                    max_hp[actor][i] += max_hp[actor][a]
                    attack[actor][i] += attack[actor][a]
                if dads:
                    hp[actor][a] = 0.0
        if hp[actor][a] <= 0 and hp[actor][a] < actor_hp_start:
            # Extrapolated: the drained card dies and its side loses the turn.
            _event(trace, "DEATH", side=actor, slot=active[actor], reason="turn_start")
            active[actor] += 1
            _on_death(battle, hp, max_hp, attack, order, active, dodges, actor, a, r, trace, status, chance, stored)
            while active[actor] < len(order[actor]) and hp[actor][order[actor][active[actor]]] <= 0:
                active[actor] += 1  # Skip cards that died waiting (e.g. poisoned in the deck).
            queue = [side for side in queue if side != actor]
            outcome = _terminal(battle, active, order)
            if outcome is not None:
                _event(trace, "END_CHECK", outcome=("A", "B", "tie")[outcome])
                return None, outcome
            _event(trace, "TRANSITION", side=actor, next_slot=active[actor])
            if (_entry(battle, hp, max_hp, attack, active, order, entered, stolen, status, actor, r, trace, chance)
                    and _queues_entry_hit(base_battle, status, order, active, hp, actor)):
                queue.append(actor)
            return _freeze(hp, max_hp, attack, used, survivals, completed, order, entered, dodges, stolen, turns, hits, own_dodges, burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked, inscale, stored, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, active,
                           1 - actor, 0, entry_queue=queue), None
        if attacker.action_start_heal_max_hp_fraction:
            _heal(hp, max_hp, actor, a, max_hp[actor][a] * attacker.action_start_heal_max_hp_fraction,
                  options, trace, "TURN_START_HEAL")
        if attacker.action_start_heal_attack_fraction:
            _heal(hp, max_hp, actor, a, attack[actor][a] * attacker.action_start_heal_attack_fraction,
                  options, trace, "TURN_START_HEAL")
        if attacker.restore_lowered_attack and attack[actor][a] < attacker.attack:
            attack[actor][a] = float(attacker.attack)
        resting[actor][a] = 0  # Its own turn ends the rest's protection.
        if attacker.periodic_block_period and turns[actor][a] % attacker.periodic_block_period == 0:
            used[actor][a] = False  # A fresh shield (block) this turn.
        if attacker.turn_start_stat_growth:
            # Limitless Rivals (IMG_0336, before each of its attacks: 2,705 -> 2,976 -> 3,246 ATK): a fixed gain of this share of
            # its pre-growth stats; the max HP gain is also added to current HP. t = its completed turns.
            t, g = turns[actor][a], attacker.turn_start_stat_growth
            growth = (1 + g * (t + 1)) / (1 + g * t)
            gain = _round(max_hp[actor][a] * growth, options.stat_rounding) - max_hp[actor][a]
            attack[actor][a] = _round(attack[actor][a] * growth, options.stat_rounding)
            max_hp[actor][a] += gain
            hp[actor][a] += gain
        if attacker.turn_start_incoming_reduction:
            # Damage reduction in fixed steps up to a cap (IMG_0336: Hades' hits x0.93, 0.82, 0.69 after 1-3 steps;
            # IMG_0337: Nao's hits level off near x0.5 from the 5th step to the 19th).
            t, g, cap = turns[actor][a], attacker.turn_start_incoming_reduction, attacker.incoming_reduction_cap
            inscale[actor][a] *= (1 - min(cap, g * (t + 1))) / (1 - min(cap, g * t))
        if attacker.turn_start_steal_fraction and hp[enemy][d] > 0:
            # Mother of Beasts (IMG_0317: Inari 20,480 -> 17,409 -> 14,797 ATK while it gained 3,072, 2,611).
            _transfer(hp, max_hp, attack, (enemy, d), (actor, a), attacker.turn_start_steal_fraction, options.stat_rounding,
                      trace, "TURN_START_STEAL")
        if attacker.turn_start_hp_steal_fraction and hp[enemy][d] > 0:
            take = _round(hp[enemy][d] * attacker.turn_start_hp_steal_fraction, options.stat_rounding)
            hp[enemy][d] -= take
            hp[actor][a] += take
            max_hp[actor][a] = max(max_hp[actor][a], hp[actor][a])
        if chance.roll(attacker.turn_start_full_heal_chance):
            hp[actor][a] = max(hp[actor][a], max_hp[actor][a])
        if chance.roll(attacker.turn_start_attack_chance):
            attack[actor][a] = _round(attack[actor][a] * attacker.turn_start_attack_multiplier, options.stat_rounding)
        # Astraeus's Aquarius (user): heals 30% at or below half HP, otherwise +25% max HP.
        if attacker.turn_start_low_hp_heal_fraction and hp[actor][a] <= max_hp[actor][a] * attacker.turn_start_low_hp_threshold:
            _heal(hp, max_hp, actor, a, max_hp[actor][a] * attacker.turn_start_low_hp_heal_fraction, options, trace, "TURN_START_HEAL")
        elif attacker.turn_start_max_hp_multiplier != 1:
            gain = _round(max_hp[actor][a] * attacker.turn_start_max_hp_multiplier, options.stat_rounding) - max_hp[actor][a]
            max_hp[actor][a] += gain
            hp[actor][a] += gain
        if attacker.shield_period and turns[actor][a] % attacker.shield_period == 0:
            ext['shields'][actor][a] = max(ext['shields'][actor][a], 1)  # Shield of Ahimsa: a shield every other turn.
        if not state.turn_actions_remaining:
            for position in range(active[actor], len(order[actor])):
                c = order[actor][position]
                card = battle.teams[actor][c]
                if card.composer_chance_step and hp[actor][c] > 0:  # The Composer: 10%, +10% per turn, to confuse.
                    ext['timer'][actor][c] += 1
                    if chance.roll(min(1.0, card.composer_chance_step * ext['timer'][actor][c])) and hp[enemy][d] > 0:
                        confused[enemy][d] = max(confused[enemy][d], 1)
            if target.reflect_even_turns:  # Tsukuyomi (user): the opponent's even turns are reflected.
                ext['timer'][enemy][d] += 1
                reflect = ext['timer'][enemy][d] % 2 == 0
        if attacker.perish_turns and ext['timer'][actor][a] and not state.turn_actions_remaining:
            ext['timer'][actor][a] -= 1
            if not ext['timer'][actor][a]:
                # Sleep Paralysis (IMG_0345): at the start of its 4th turn both active cards die (deaths resolve below).
                _ability_hit(battle, hp, attack, order, active, enemy, d, hp[enemy][d], options.rounding)
                _ability_hit(battle, hp, attack, order, active, actor, a, hp[actor][a], options.rounding)
                skipped = True
        if not skipped and slowed[actor][a] and not frozen[actor][a]:
            slowed[actor][a] -= 1
            skipped = True
        if not skipped and not frozen[actor][a] and chance.roll(target.skip_enemy_turn_chance):
            skipped = True
        if not skipped and not frozen[actor][a] and confused[actor][a] and chance.roll(CONFUSION_CHANCE):
            # User: confusion makes the card attack itself (about 50%); raw own damage.
            self_hit = _round(attack[actor][a] * attacker.outgoing_multiplier, options.rounding)
            hp[actor][a] -= self_hit
            _event(trace, "CONFUSED_SELF_HIT", side=actor, damage=self_hit, hp=hp[actor][a])
            skipped = True
            if confused[actor][a] >= CONFUSION_ETERNAL:
                pass
            elif confused[actor][a] >= CONFUSION_UNTIL_SELF_HIT:  # Kuchisake-onna: -20% ATK, then the confusion ends.
                attack[actor][a] = _round(attack[actor][a] * 0.8, options.stat_rounding)
                confused[actor][a] = 0
            if target.confused_self_hit_stat_multiplier != 1 and hp[enemy][d] > 0:  # Cthulu: +50% stats per self-hit.
                _scale(hp, max_hp, attack, enemy, d, target.confused_self_hit_stat_multiplier, target.confused_self_hit_stat_multiplier,
                       options.stat_rounding)
        if not skipped and not frozen[actor][a] and reflect:
            # Tsukuyomi: the attack lands on the attacker, with the effects it carries.
            self_hit = _round(attack[actor][a] * attacker.outgoing_multiplier * _action_hits(attacker, turns[actor][a]), options.rounding)
            hp[actor][a] -= self_hit
            burn[actor][a] = max(burn[actor][a], attacker.hit_burn_turns)
            bleed[actor][a] = max(bleed[actor][a], attacker.hit_bleed_turns)
            frostbite[actor][a] = max(frostbite[actor][a], attacker.hit_frostbite_turns)
            _poison(status, actor, a, POISON_PERMANENT if attacker.poison_permanent and attacker.hit_poison_turns else attacker.hit_poison_turns,
                    attack[actor][a] * attacker.poison_fraction)
            weakness[actor][a] = max(weakness[actor][a], attacker.hit_weakness_multiplier)
            _event(trace, "REFLECTED", side=actor, damage=self_hit, hp=hp[actor][a])
            skipped = True
        if not skipped and attacker.buddha:
            if _buddha(battle, hp, max_hp, order, active, status, actor, a, options, trace):
                relocated.append(actor)
            skipped = True
        if not skipped and attacker.never_attacks:
            skipped = True
        if not skipped and attacker.alternate_rest and turns[actor][a] % 2 == 1:
            skipped = True
            resting[actor][a] = 1  # Rests every other turn, untouchable until its next turn.
    skipped = skipped or bool((not is_counter and not is_entry) and frozen[actor][a])  # User: entry hits ignore freeze.
    if not skipped and not is_counter and not is_entry and not state.hits_remaining:
        # User: Onmyoji and Kira count the normal attacks of the ally in play, just before each attack,
        # even after they died; they hit the enemy card in play.
        before_count = hp[enemy][d]
        for source in range(len(battle.teams[actor])):
            if strike_timer[actor][source]:
                strike_timer[actor][source] -= 1
                if not strike_timer[actor][source]:
                    strike = _round(strike_damage[actor][source], options.rounding)
                    _ability_hit(battle, hp, attack, order, active, enemy, d, strike, options.rounding)
                    _event(trace, "DELAYED_STRIKE", side=actor, damage=strike, hp=hp[enemy][d])
        if doom[enemy][d] and hp[enemy][d] > 0:
            doom[enemy][d] -= 1
            if not doom[enemy][d]:
                hp[enemy][d] = 0.0
                _event(trace, "DOOM", side=enemy)
        if hp[enemy][d] <= 0 < before_count:
            skipped = True  # The card in front died before the attack (provisional: the attack is spent).
    if skipped:
        _event(trace, "SKIP", side=actor, slot=active[actor], reason="frozen" if frozen[actor][a] else "slowed")
    elif is_counter:
        _event(trace, "COUNTER_DECLARE", side=actor, slot=active[actor], target_side=enemy, target_slot=active[enemy])
    elif is_entry:
        _event(trace, "ENTRY_ATTACK_DECLARE", side=actor, slot=active[actor], target_side=enemy,
               target_slot=active[enemy], multiplier=attacker.entry_hit_multiplier)
    elif not state.hits_remaining:
        _event(trace, "ATTACK_DECLARE", side=actor, slot=active[actor], target_side=enemy,
               target_slot=active[enemy], attacks=total_hits)
    if not skipped:
        _event(trace, "HIT_DECLARE", side=actor, slot=active[actor], target_side=enemy,
               target_slot=active[enemy], hit_index=hit_index, kind="counter" if is_counter else "entry" if is_entry else "normal")
    swap_boost, kraken, redirected, chosen, deck_target = 1.0, 1, False, [], None
    deck_hits = []  # (side, index, hp before): waiting cards hurt by redirects, links and overkill
    if not skipped and not is_counter and not is_entry and hit_index == 1:
        if attacker.swap_attack_chance and chance.roll(attacker.swap_attack_chance):
            waiting = _waiting(hp, order, active, enemy)  # Heavenly Demon: pulls a random waiting card forward, 3x.
            if waiting:
                p = waiting[chance.pick(len(waiting))]
                order[enemy][active[enemy]], order[enemy][p] = order[enemy][p], d
                d = order[enemy][active[enemy]]
                target = battle.teams[enemy][d]
                enemy_hp_start = hp[enemy][d]
                entered[enemy][d] = True
                swap_boost = attacker.swap_attack_multiplier
        if attacker.random_deck_target:
            waiting = _waiting(hp, order, active, enemy)  # Brachiosaurus: a random living waiting card instead.
            if waiting:
                deck_target = waiting[chance.pick(len(waiting))]
                redirected = True
        if attacker.chaos_targets:
            redirected = not chance.roll(attacker.chaos_hit_chance)  # Chaos (user): each enemy hit independently.
            chosen = [p for p in _waiting(hp, order, active, enemy)[:attacker.chaos_targets - 1] if chance.roll(attacker.chaos_hit_chance)]
    if not skipped and followup_hit and attacker.followup_chance and not chance.roll(attacker.followup_chance):
        redirected = True  # Storm Spirit: no second attack this time.
    if not skipped and is_entry and attacker.entry_hit_count_max:
        kraken = 1 + chance.pick(attacker.entry_hit_count_max)  # Kraken: 1-8 hits of 10% on the whole team.
    punish = 1.0
    if not skipped and dodge and attacker.evade_punish_multiplier:
        dodge, punish = False, attacker.evade_punish_multiplier  # Sagittarius: 2x to cards that would have evaded.
    if not skipped and periodic and hit_index == 1:
        if attacker.periodic_attack_heal_max_hp_fraction:
            _heal(hp, max_hp, actor, a, max_hp[actor][a] * attacker.periodic_attack_heal_max_hp_fraction, options, trace, "PERIODIC_HEAL")
        if attacker.periodic_enemy_slow_turns:
            slowed[enemy][d] = max(slowed[enemy][d], attacker.periodic_enemy_slow_turns)
    dealt = 0.0
    intercepted = None  # Side whose active card was displaced by an interception.
    alternate_dodge = False
    if not skipped and target.alternate_dodge:
        alternate_dodge = hits[enemy][d] % 2 == target.alternate_dodge - 1
        hits[enemy][d] += 1
    actor_lost = 1 - hp[actor][a] / max_hp[actor][a]
    if skipped:
        pass
    elif blind[actor][a]:
        _event(trace, "PRE_HIT", outcome="blind_miss", side=enemy, hit_index=hit_index)
    elif is_entry and attacker.entry_hit_next_card:
        pass  # Ra: the entry hit goes to the card behind instead (below).
    elif redirected:
        pass  # Brachiosaurus / a missed Chaos hit on the card in play.
    elif dodge:
        _event(trace, "PRE_HIT", outcome="dodge", side=enemy, hit_index=hit_index)
        li = None if attacker.domain else _luminant(battle, hp, order, enemy, d)
        if li is not None and ext['evades'][enemy][d] < LUMINANT_MAX:
            ext['evades'][enemy][d] += 1  # Eclipseborn Luminant gains 10% of the prevented damage (200% cap).
            attack[enemy][li] = min(_round(attack[enemy][li] + attack[actor][a] * attacker.outgoing_multiplier * LUMINANT_GAIN,
                                           options.stat_rounding), max(attack[enemy][li], LUMINANT_CAP * battle.teams[enemy][li].attack))
        if target.dodge_attack_multiplier != 1:
            attack[enemy][d] = _round(attack[enemy][d] * target.dodge_attack_multiplier, options.stat_rounding)
        if target.dodge_bonus_multiplier != 1:
            charged[enemy][d] = 1
    elif target.invincible or resting[enemy][d] or undying[enemy][d]:
        _event(trace, "PRE_HIT", outcome="invincible", side=enemy, hit_index=hit_index)
    else:
        raw = ((attack[actor][a] + attacker.target_attack_damage_fraction * attack[enemy][d]
                + attacker.lost_hp_damage_fraction * (max_hp[actor][a] - hp[actor][a])
                + attacker.max_hp_damage_fraction * max_hp[actor][a]) * attacker.outgoing_multiplier
               + attacker.target_current_hp_damage_fraction * hp[enemy][d])
        if is_entry:
            raw *= attacker.entry_hit_multiplier * kraken
        raw *= swap_boost * punish
        if not is_counter and not is_entry and triggered[actor][a] & SAM_BOOST:
            raw *= SAM_BOOST_MULTIPLIER
        if critical:
            raw *= attacker.critical_multiplier
        if is_counter:
            raw *= attacker.counter_multiplier
        if not is_counter and not is_entry:
            if hit_index == 1 and turns[actor][a] == 0:
                raw *= attacker.first_attack_multiplier
            if attacker.recharge_turns and turns[actor][a] % (1 + attacker.recharge_turns) == 0:
                raw *= attacker.charged_attack_multiplier
            raw *= attacker.hit_growth_multiplier ** (hit_index - 1)
        gamble_failed = False
        if not is_counter and not is_entry:
            if charged[actor][a]:
                raw *= attacker.dodge_bonus_multiplier
                charged[actor][a] = 0
            if attacker.random_multiplier_max > 1:
                factor = attacker.random_multiplier_max
                for k in range(1, attacker.random_multiplier_max):
                    if chance.roll(1 / (attacker.random_multiplier_max - k + 1)):
                        factor = k
                        break
                raw *= factor
            if attacker.gamble_chance:
                if chance.roll(attacker.gamble_chance):
                    raw *= attacker.gamble_multiplier
                else:
                    gamble_failed = True
        if frostbite[actor][a]:
            raw *= target.frostbite_incoming_multiplier  # Yeti: half damage from frostbitten enemies.
        if not faded[actor][a]:
            raw *= attacker.fade_outgoing_multiplier
        raw *= _field_multiplier(battle, order, active, hp)
        if attacker.border_rarity > target.border_rarity:
            raw *= attacker.border_advantage_multiplier
        if target.pack_index > attacker.pack_index:
            raw *= attacker.younger_target_multiplier  # Later packs are younger (extrapolated).
        if attacker.bonus_vs_class_mask & target.class_mask:
            raw *= attacker.class_damage_multiplier
        first_hit_bypass = False
        if not is_counter and not is_entry and attacker.first_hit_per_enemy_multiplier != 1 and not marked[enemy][d]:
            raw *= attacker.first_hit_per_enemy_multiplier
            marked[enemy][d] = 1
            first_hit_bypass = True
        if hp[enemy][d] <= max_hp[enemy][d] * attacker.weak_target_threshold:
            raw *= attacker.weak_target_multiplier
        if hp[actor][a] > hp[enemy][d]:
            raw *= attacker.advantage_damage_multiplier
        raw *= 1 + attacker.lost_hp_damage_bonus * actor_lost
        if periodic:
            raw *= attacker.periodic_attack_multiplier
        if followup_hit:
            raw *= attacker.followup_multiplier
        if frozen[enemy][d]:
            raw *= attacker.frozen_target_multiplier
        if attacker.max_hp_damage_costs and attacker.max_hp_damage_fraction:
            keep = 1 - attacker.max_hp_damage_fraction
            _scale(hp, max_hp, attack, actor, a, keep, 1, options.stat_rounding)
        if not math.isfinite(raw):
            raise ValueError("Damage calculation overflowed")
        bypass = bool(attacker.bypass_defenses or (is_entry and attacker.entry_bypass_defenses) or first_hit_bypass
                      or (followup_hit and attacker.followup_bypass))
        incoming = target.incoming_multiplier * inscale[enemy][d]
        incoming *= weakness[enemy][d]
        if not faded[enemy][d]:
            incoming *= target.fade_incoming_multiplier
        if target.border_rarity > attacker.border_rarity:
            incoming *= target.border_advantage_incoming_multiplier
        if attacker.card_rarity * attacker.border_rarity < target.card_rarity * target.border_rarity:
            incoming *= target.lower_rarity_incoming_multiplier
        if target.guard_vs_class_mask & attacker.class_mask:
            incoming *= target.class_incoming_multiplier
        if hp[enemy][d] < hp[actor][a]:
            incoming *= target.disadvantage_incoming_multiplier
        if hp[enemy][d] < max_hp[enemy][d] * target.low_hp_incoming_threshold:
            incoming *= target.low_hp_incoming_multiplier
        if target.lost_hp_incoming_bonus:  # Taurus: less damage taken as HP lowers (up to 2.5x).
            incoming /= 1 + target.lost_hp_incoming_bonus * (1 - hp[enemy][d] / max_hp[enemy][d])
        if not ext['negated'][enemy]:
            incoming *= target.aura_incoming_multiplier  # The blue support (Marrowclaw can negate it).
        # Thresholds inspect incoming attack strength before defensive changes.
        # Extrapolated: bypass ignores thresholds, reductions, caps and blocks,
        # keeping damage amplifiers; dodges still apply.
        dodge_threshold = 0.0 if bypass else max_hp[enemy][d] * target.dodge_below_max_hp
        nullify_threshold = 0.0 if bypass else attack[enemy][d] * target.nullify_below_attack
        reduction = 0.0 if bypass else max_hp[enemy][d] * target.damage_reduction_max_hp
        cap = max_hp[enemy][d] * target.damage_cap_max_hp
        if bypass:
            incoming = max(1.0, incoming)
        if any(not math.isfinite(value) for value in (dodge_threshold, nullify_threshold, reduction, cap)):
            raise ValueError("Defense calculation overflowed")
        threshold_dodge = raw < dodge_threshold
        nullified = raw < nullify_threshold
        sign_nullified = bool(not bypass and target.nullify_max_hp_threshold and raw < max_hp[enemy][d] * max(
            0.0, target.nullify_max_hp_threshold - target.nullify_threshold_step * ext['timer'][enemy][d]))
        reduced = max(0.0, raw - reduction)
        raw = reduced * incoming
        if not math.isfinite(raw):
            raise ValueError("Damage calculation overflowed")
        if target.damage_cap_max_hp and not bypass:
            raw = min(raw, cap)
        damage = _round(raw, options.rounding)
        if attacker.execute_below_max_hp and hp[enemy][d] < max_hp[enemy][d] * attacker.execute_below_max_hp:
            damage = max(damage, hp[enemy][d])
        if block_pool[enemy][d] > 0 and not bypass:
            absorbed = min(block_pool[enemy][d], damage)  # Steve's blocks soak damage first.
            block_pool[enemy][d] -= absorbed
            damage -= absorbed
            _event(trace, "BLOCK_ABSORB", side=enemy, absorbed=absorbed, remaining=block_pool[enemy][d])
        blocked = not bypass and not used[enemy][d] and (
            target.block_mode == 1 or (target.block_mode == 2 and damage < hp[enemy][d]))
        behind = active[enemy] + 1
        # User-observed Piccolo: intercepts a lethal Raze counter. The card's own
        # lethal survival is provisionally checked first (the hit is not lethal).
        interceptor = order[enemy][behind] if (
            damage >= hp[enemy][d] and survivals[enemy][d] >= target.lethal_survivals
            and behind < len(order[enemy]) and hp[enemy][order[enemy][behind]] > 0
            and battle.teams[enemy][order[enemy][behind]].intercept_lethal_multiplier) else None
        evaded = bool(target.hit_chance_squared) and not chance.roll(min(1.0, (damage / max_hp[enemy][d]) ** 2))
        if gamble_failed:
            _heal(hp, max_hp, enemy, d, max_hp[enemy][d] * attacker.gamble_fail_heal_fraction, options, trace, "GAMBLE_HEAL")
        elif not is_counter and not is_entry and target.normal_attack_dodge_cost:
            cost = _round(hp[enemy][d] * target.normal_attack_dodge_cost, options.rounding)
            hp[enemy][d] -= cost  # Dodges normal attacks by paying HP (extrapolated: rounded up).
            _event(trace, "PRE_HIT", outcome="paid_dodge", side=enemy, hit_index=hit_index, cost=cost)
        elif evaded:
            _event(trace, "PRE_HIT", outcome="evaded", side=enemy, hit_index=hit_index)
        elif threshold_dodge or nullified:
            _event(trace, "PRE_HIT", outcome="threshold_dodge" if threshold_dodge else "nullify", side=enemy, hit_index=hit_index)
        elif sign_nullified:
            ext['timer'][enemy][d] += 1  # Astraeus: the threshold drops 15% per nullification.
            _event(trace, "PRE_HIT", outcome="nullify", side=enemy, hit_index=hit_index)
        elif (target.status_dodge_max and own_dodges[enemy][d] < target.status_dodge_max
              and _has_status(status, actor, a)):
            own_dodges[enemy][d] += 1  # Mist Spirit: cards with a status cannot hit it (5 dodges).
            _event(trace, "PRE_HIT", outcome="status_dodge", side=enemy, hit_index=hit_index)
        elif alternate_dodge:
            _event(trace, "PRE_HIT", outcome="alternate_dodge", side=enemy, hit_index=hit_index)
            if target.dodge_extra_action:
                ext['extra'][enemy][d] += 1  # Gingerbread Man: one more attack next turn.
            if target.alternate_dodge_heal_fraction:
                _heal(hp, max_hp, enemy, d, damage * target.alternate_dodge_heal_fraction, options, trace, "DODGE_HEAL")
        elif blocked:
            used[enemy][d] = True
            _event(trace, "PRE_HIT", outcome="block", side=enemy, hit_index=hit_index)
        elif ext['shields'][enemy][d] and not bypass and not attacker.domain:
            ext['shields'][enemy][d] -= 1
            _event(trace, "PRE_HIT", outcome="shield", side=enemy, hit_index=hit_index)
        elif target.parry_chance and not bypass and chance.roll(target.parry_chance):
            # Vajra Short Sword (IMG_0342): no damage; 75% reflected, at most 75% of the attacker's current HP.
            if hp[actor][a] > 0:
                hp[actor][a] -= _round(min(damage, hp[actor][a]) * target.parry_reflect_fraction, options.rounding)
            _event(trace, "PRE_HIT", outcome="parry", side=enemy, hit_index=hit_index)
        elif dodges[enemy][d]:
            # User-observed True Prophet/Shu (Raze and mirror): the next hit is dodged,
            # lethal or not, and Shu still recovers from the dodged damage, capped at max HP.
            dodges[enemy][d] -= 1
            _event(trace, "PRE_HIT", outcome="prophet_dodge", side=enemy, hit_index=hit_index)
            if target.heal_damage_taken_fraction:
                _heal(hp, max_hp, enemy, d, min(hp[enemy][d], damage) * target.heal_damage_taken_fraction,
                      options, trace, "DAMAGE_RECOVERY")
        elif damage >= hp[enemy][d] and own_dodges[enemy][d] < target.own_lethal_dodges:
            own_dodges[enemy][d] += 1
            _event(trace, "PRE_HIT", outcome="own_lethal_dodge", side=enemy, hit_index=hit_index)
            if target.lethal_dodge_heal_max_hp_fraction:
                _heal(hp, max_hp, enemy, d, max_hp[enemy][d] * target.lethal_dodge_heal_max_hp_fraction, options, trace, "DODGE_HEAL")
            attack[enemy][d] = _round(attack[enemy][d] * target.lethal_dodge_attack_multiplier, options.stat_rounding)
            nxt = active[enemy] + 1
            if target.lethal_dodge_swap and nxt < len(order[enemy]) and hp[enemy][order[enemy][nxt]] > 0:
                # Dilophosaurus: it steps back and the next ally comes in (entering normally).
                order[enemy][active[enemy]], order[enemy][nxt] = order[enemy][nxt], d
                queue = [side for side in queue if side != enemy]
                intercepted = enemy
                _event(trace, "LETHAL_DODGE_SWAP", side=enemy, slot=active[enemy])
                if (_entry(battle, hp, max_hp, attack, active, order, entered, stolen, status, enemy, options.stat_rounding, trace, chance)
                        and _queues_entry_hit(base_battle, status, order, active, hp, enemy)):
                    queue.append(enemy)
        elif interceptor is not None:
            # Provisional: the displaced card swaps into the interceptor's slot.
            factor = target.piccolo_factor_override or battle.teams[enemy][interceptor].intercept_lethal_multiplier
            order[enemy][active[enemy]], order[enemy][behind] = interceptor, d
            for stats in (hp, max_hp, attack):
                stats[enemy][interceptor] = _round(stats[enemy][interceptor] * factor, options.stat_rounding)
            entered[enemy][interceptor] = True
            queue = [side for side in queue if side != enemy]  # The displaced card's entry hit is cancelled.
            intercepted = enemy
            _event(trace, "INTERCEPT", side=enemy, slot=active[enemy], displaced_to=behind, hit_index=hit_index,
                   hp=hp[enemy][interceptor], attack=attack[enemy][interceptor])
            # User-observed: Chronus steals from a swapping-in Piccolo (after its
            # 2x, following the Poseidon boost-then-steal order; provisional).
            _steal(battle, hp, max_hp, attack, active, order, stolen, enemy, options.stat_rounding, trace)
            _enemy_entry(battle, hp, order, active, status, enemy)
        else:
            old_hp = hp[enemy][d]
            if damage >= old_hp and survivals[enemy][d] >= target.lethal_survivals and (
                    not revived[enemy][d] and chance.roll(target.chance_survival_probability)):
                revived[enemy][d] = 1
                hp[enemy][d] = (max(1.0, _round(max_hp[enemy][d] * target.chance_survival_hp_fraction, options.stat_rounding))
                                if target.chance_survival_hp_fraction else 1.0)
                if target.chance_survival_freeze_turns:
                    frozen[actor][a] = max(frozen[actor][a], target.chance_survival_freeze_turns)
                _event(trace, "LETHAL_REPLACEMENT", side=enemy, effect="chance_revive")
            elif damage >= old_hp and survivals[enemy][d] >= target.lethal_survivals and (
                    not guarded[enemy][d] and chance.roll(0.0 if ext['negated'][enemy] else target.guardian_chance)):
                guarded[enemy][d] = 1  # Guardian Angel support: once per ally, 1 HP.
                hp[enemy][d] = 1.0
                _event(trace, "LETHAL_REPLACEMENT", side=enemy, effect="guardian_angel")
            elif damage >= old_hp and survivals[enemy][d] < target.lethal_survivals:
                if target.survival_damage_max_hp_fraction:
                    # Black box overload: damage the attacker by a share of own max HP.
                    hp[actor][a] -= _round(max_hp[enemy][d] * target.survival_damage_max_hp_fraction, options.rounding)
                survivals[enemy][d] += 1
                hp[enemy][d] = (max(1.0, _round(max_hp[enemy][d] * target.lethal_survival_hp_fraction, options.stat_rounding))
                                if target.lethal_survival_hp_fraction else 1.0)
                attack[enemy][d] = _round(attack[enemy][d] * target.survival_attack_multiplier, options.stat_rounding)
                _event(trace, "LETHAL_REPLACEMENT", side=enemy, effect="survive_at_one_hp", uses=survivals[enemy][d])
            elif damage >= old_hp and survivals[enemy][d] - target.lethal_survivals < target.death_reflects:
                survivals[enemy][d] += 1  # Parallax: the lethal hit is negated and the attacker dies.
                hp[actor][a] = min(hp[actor][a], 0.0)
                _event(trace, "LETHAL_REPLACEMENT", side=enemy, effect="death_reflect")
            elif damage >= old_hp and target.awaken_on_death and not triggered[enemy][d] & AWAKENED:
                triggered[enemy][d] |= AWAKENED  # Ultimate Brawler: awakens instead of its first death.
                hp[enemy][d] = max_hp[enemy][d]
                _scale(hp, max_hp, attack, enemy, d, target.transform_stat_multiplier, target.transform_stat_multiplier, options.stat_rounding)
                if target.transform_form:
                    ability[enemy][d] = POOL_CODE + target.transform_form - 1
                _event(trace, "LETHAL_REPLACEMENT", side=enemy, effect="awaken")
            elif damage >= old_hp and target.zero_hp_survival_turns and not triggered[enemy][d] & 2:
                # Zombie Dragon: survives at 0 HP (shown as 1), untouchable, and dies when its doom ends.
                triggered[enemy][d] |= 2
                hp[enemy][d] = 1.0
                undying[enemy][d] = target.zero_hp_survival_turns
                _event(trace, "LETHAL_REPLACEMENT", side=enemy, effect="zero_hp_survival")
            else:
                redirect = next((order[enemy][p] for p in range(active[enemy] + 1, len(order[enemy]))
                                 if hp[enemy][order[enemy][p]] > 0 and battle.teams[enemy][order[enemy][p]].ally_redirect_fraction), None)
                if redirect is not None and damage > 0:  # Bei Fang Xuan Wu takes 50% of the damage its allies take.
                    part = _round(damage * battle.teams[enemy][redirect].ally_redirect_fraction, options.rounding)
                    damage -= part
                    deck_hits.append((enemy, redirect, hp[enemy][redirect]))
                    _ability_hit(battle, hp, attack, order, active, enemy, redirect, part, options.rounding)
                hp[enemy][d] -= damage
                if attacker.overkill_carry and hp[enemy][d] < 0:  # Tricerotops: overkill carries to the next card.
                    waiting = _waiting(hp, order, active, enemy)
                    if waiting:
                        nxt = order[enemy][waiting[0]]
                        deck_hits.append((enemy, nxt, hp[enemy][nxt]))
                        _ability_hit(battle, hp, attack, order, active, enemy, nxt, -hp[enemy][d], options.rounding)
            dealt = max(0.0, old_hp - hp[enemy][d])
            _event(trace, "DAMAGE_APPLY", side=enemy, damage=damage,
                   hp=hp[enemy][d], hp_lost=dealt, hit_index=hit_index, critical=critical)
            # Working interpretation: heal from actual HP removed, excluding
            # overkill and HP preserved by lethal survival; reactions follow.
            actual_loss = min(old_hp, dealt)
            if actual_loss > 0:
                lifesteal = attacker.heal_damage_dealt_fraction + attacker.lost_hp_lifesteal * actor_lost
                if lifesteal:
                    _heal(hp, max_hp, actor, a, actual_loss * lifesteal, options, trace, "LIFESTEAL")
                if target.heal_damage_taken_fraction:
                    _heal(hp, max_hp, enemy, d, actual_loss * target.heal_damage_taken_fraction,
                          options, trace, "DAMAGE_RECOVERY")
                if target.damaged_toy_stat_gain:
                    gain = 1 + target.damaged_toy_stat_gain  # Awakened Nutcracker: every living awakened Toy, itself included.
                    for i, toy in enumerate(base_battle.teams[enemy]):
                        if toy.awakened_toy and hp[enemy][i] > 0:
                            _scale(hp, max_hp, attack, enemy, i, gain, gain, options.stat_rounding)
                _hit_effects(hp, max_hp, attack, status, actor, a, enemy, d, attacker, target, actual_loss, options, trace)
                partner = ext['link'][enemy][d] - 1
                if partner >= 0 and hp[enemy][partner] > 0:  # Fate Seamstress: the linked card takes the same damage.
                    deck_hits.append((enemy, partner, hp[enemy][partner]))
                    _ability_hit(battle, hp, attack, order, active, enemy, partner, actual_loss, options.rounding)
                if attacker.splash_damage_fraction:
                    # User: Academy Student's other enemies lose 30% of damage dealt, directly.
                    _bench_hits(battle, hp, max_hp, attack, order, active, dodges, survivals, enemy,
                                actual_loss * attacker.splash_damage_fraction, False, options, trace, status, chance, stored)
                if hp[enemy][d] > 0:
                    if attacker.hit_frostbite_turns:
                        frostbite[enemy][d] = max(frostbite[enemy][d], attacker.hit_frostbite_turns)
                    if attacker.hit_freeze_turns and chance.roll(attacker.hit_freeze_chance):
                        frozen[enemy][d] = max(frozen[enemy][d], attacker.hit_freeze_turns)
                    if attacker.hit_stun_chance and chance.roll(attacker.hit_stun_chance):
                        slowed[enemy][d] = max(slowed[enemy][d], 1)  # Staff of Perfect Enlightenment: a 1-turn stun.
                    if attacker.hit_burn_chance and chance.roll(attacker.hit_burn_chance):
                        burn[enemy][d] = max(burn[enemy][d], 2)  # Flame Wizard: burn for 2 turns.
                    if attacker.ratio_kill and chance.roll(min(1.0, actual_loss / old_hp)):
                        hp[enemy][d] = 0.0
                        _event(trace, "RATIO_KILL", side=enemy)
                if target.dodge_growth:
                    hits[enemy][d] += 1
                if hp[enemy][d] > 0:
                    if target.damaged_attack_loss_fraction:
                        attack[enemy][d] = max(0.0, _round(attack[enemy][d] - actual_loss * target.damaged_attack_loss_fraction,
                                                           options.stat_rounding))
                    if target.stored_damage_to_next_attack_fraction:
                        stored[enemy][d] = _round(stored[enemy][d] + actual_loss * target.stored_damage_to_next_attack_fraction,
                                                  options.stat_rounding)
                    if target.damage_taken_to_party_attack_fraction:
                        for position in range(active[enemy] + 1, len(order[enemy])):
                            ally = order[enemy][position]
                            if hp[enemy][ally] > 0:
                                attack[enemy][ally] = _round(attack[enemy][ally] + actual_loss
                                                             * target.damage_taken_to_party_attack_fraction, options.stat_rounding)
                    if (target.low_hp_transform_threshold and not triggered[enemy][d] & TRANSFORMED
                            and hp[enemy][d] < max_hp[enemy][d] * target.low_hp_transform_threshold):
                        triggered[enemy][d] |= TRANSFORMED  # Sun Wukong (x2 stats), Naga (heals, cursed form).
                        _scale(hp, max_hp, attack, enemy, d, target.transform_stat_multiplier, target.transform_stat_multiplier,
                               options.stat_rounding)
                        if target.transform_heal:
                            hp[enemy][d] = max_hp[enemy][d]
                        if target.transform_form:
                            ability[enemy][d] = POOL_CODE + target.transform_form - 1
                    if target.fade_threshold and not faded[enemy][d] and hp[enemy][d] < max_hp[enemy][d] * target.fade_threshold:
                        faded[enemy][d] = 1
                        if target.fade_stat_multiplier != 1:
                            back = 1 / target.fade_stat_multiplier
                            _scale(hp, max_hp, attack, enemy, d, back, back, options.stat_rounding)
                        _event(trace, "FADE", side=enemy)
    if not skipped and is_entry and attacker.entry_hit_all_enemies:
        # User (Seraphim): the entry attack hits the whole team for 25%; waiting cards can
        # dodge and die for real (the game's display drops one card from the front per death).
        _bench_hits(battle, hp, max_hp, attack, order, active, dodges, survivals, enemy,
                    attack[actor][a] * attacker.outgoing_multiplier * attacker.entry_hit_multiplier * kraken
                    * _field_multiplier(battle, order, active, hp), True, options, trace, status, chance, stored, hits)
    if not skipped and is_entry and attacker.entry_hit_next_card:
        # Ra (IMG_0325): the next enemy card takes the entry hit; its on-death ability does not trigger.
        _bench_hits(battle, hp, max_hp, attack, order, active, dodges, survivals, enemy,
                    attack[actor][a] * attacker.outgoing_multiplier * attacker.entry_hit_multiplier
                    * _field_multiplier(battle, order, active, hp), True, options, trace, status, chance, stored, hits,
                    limit=1, on_death_effects=False)
    if not skipped and is_entry and attacker.entry_hit_also_next:
        # War Scythe (IMG_0329/0342): the entry attack also hits the next enemy, ignoring on-death abilities.
        _bench_hits(battle, hp, max_hp, attack, order, active, dodges, survivals, enemy,
                    attack[actor][a] * attacker.outgoing_multiplier * attacker.entry_hit_multiplier
                    * _field_multiplier(battle, order, active, hp), True, options, trace, status, chance, stored, hits,
                    limit=1, on_death_effects=False)
    if not skipped and not is_counter and not is_entry and attacker.fossil_deck_hits and hit_index == total_hits:
        # Cyberdon: its attack also hits fossils + 1 waiting enemies (at most fossil_deck_hits).
        _bench_hits(battle, hp, max_hp, attack, order, active, dodges, survivals, enemy,
                    attack[actor][a] * attacker.outgoing_multiplier * _field_multiplier(battle, order, active, hp),
                    True, options, trace, status, chance, stored, hits, limit=min(fossils[actor] + 1, attacker.fossil_deck_hits))
    if not skipped and (chosen or deck_target is not None):
        # Chaos's extra targets; Brachiosaurus's waiting target (as attacks: they can dodge).
        _bench_hits(battle, hp, max_hp, attack, order, active, dodges, survivals, enemy,
                    attack[actor][a] * attacker.outgoing_multiplier * _field_multiplier(battle, order, active, hp),
                    True, options, trace, status, chance, stored, hits, positions=chosen + ([deck_target] if deck_target is not None else []))
    if not skipped and not is_counter and not is_entry and hit_index == 1:
        for position in range(active[actor] + 1, len(order[actor])):
            ally = order[actor][position]
            card = battle.teams[actor][ally]
            if card.rake_fraction and hp[actor][ally] > 0 and hp[enemy][d] > 0:  # The Rake attacks alongside for 25%.
                _ability_hit(battle, hp, attack, order, active, enemy, d, _round(attack[actor][ally] * card.rake_fraction, options.rounding),
                             options.rounding)
    for s2, index, before in deck_hits:
        if hp[s2][index] <= 0 < before and order[s2].index(index) > active[s2]:
            _event(trace, "DEATH", side=s2, reason="bench")
            _on_death(battle, hp, max_hp, attack, order, active, dodges, s2, index, options.stat_rounding, trace, status, chance, stored,
                      in_deck=True)
    if (not skipped and not is_counter and not is_entry and hit_index == 1
            and debuffed[enemy][d] < target.debuff_attacker_count):
        debuffed[enemy][d] += 1
        attack[actor][a] = _round(attack[actor][a] * target.debuff_attacker_multiplier, options.stat_rounding)
    # Forest Spirit heals before the defender's counter. For composed multi-hit
    # actions the current interpretation fires once, on the last hit or a KO.
    if (not skipped and not is_counter and not is_entry and attacker.after_attack_heal_max_hp_fraction
            and (remaining == 1 or (hp[enemy][d] <= 0 and hp[enemy][d] < enemy_hp_start))):
        _heal(hp, max_hp, actor, a, max_hp[actor][a] * attacker.after_attack_heal_max_hp_fraction,
              options, trace, "AFTER_ATTACK_HEAL")
    remaining = 0 if entry_context else state.hits_remaining if is_counter else remaining - 1
    if (not is_counter and not entry_context and attacker.followup_on_critical and not critical
            and hit_index == total_hits - 1):
        remaining = 0  # Durante: the follow-up only comes after a critical strike.
    # Provisional: a displaced actor's action ends, as it would on death.
    displaced_actor = intercepted == normal_actor
    pair_actions = 0 if intercepted is not None else state.pair_actions
    target_died = hp[enemy][d] <= 0 and hp[enemy][d] < enemy_hp_start
    if target_died:
        _blast(battle, hp, max_hp, enemy, d, actor, a, chance, options, trace)
    attacker_died = hp[actor][a] <= 0 and hp[actor][a] < actor_hp_start  # Retaliation, recoil, self-hits.
    if attacker_died:
        _blast(battle, hp, max_hp, actor, a, enemy, d, chance, options, trace)
        target_died = hp[enemy][d] <= 0 and hp[enemy][d] < enemy_hp_start
    if target_died or displaced_actor or attacker_died or skipped:
        remaining = 0  # User-observed: a KO ends the multi-hit action.
    elif (not is_counter and dealt > 0 and (target.counter_on_damage or target.counter_chance) and not frozen[enemy][d]
          and fossils[enemy] >= target.counter_min_fossils
          and (target.counter_on_damage or chance.roll(target.counter_chance))):  # Berserker: a chance to counter.
        _event(trace, "COUNTER_QUEUED", side=enemy, after_hit=hit_index)
        return _freeze(hp, max_hp, attack, used, survivals, completed, order, entered, dodges, stolen, turns, hits, own_dodges, burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked, inscale, stored, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, active, normal_actor,
                       pair_actions, remaining, enemy + 1, int(is_entry), queue, state.turn_actions_remaining), None
    if remaining:
        return _freeze(hp, max_hp, attack, used, survivals, completed, order, entered, dodges, stolen, turns, hits, own_dodges, burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked, inscale, stored, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, active, normal_actor,
                       pair_actions, remaining, entry_queue=queue,
                       turn_actions_remaining=state.turn_actions_remaining), None
    actions = pair_actions + (0 if entry_context else 1)
    changed = []
    stole = False
    if target_died:
        _event(trace, "DEATH", side=enemy, slot=active[enemy], reason="damage")
        if not attacker_died:
            # Extrapolated kill order: steal, stat growth, then heals.
            r = options.stat_rounding
            if attacker.kill_steal_ability and not triggered[actor][a] & 16:
                triggered[actor][a] |= 16  # Mother of Beasts: its first victim's ability.
                stole = True
                ability[actor][a] = _ability_code(ability, enemy, d)
            if attacker.kill_steal_fraction:
                gain_hp = max_hp[enemy][d] * attacker.kill_steal_fraction
                hp[actor][a] = _round(hp[actor][a] + gain_hp, r)
                max_hp[actor][a] = _round(max_hp[actor][a] + gain_hp, r)
                attack[actor][a] = _round(attack[actor][a] + attack[enemy][d] * attacker.kill_steal_fraction, r)
            if attacker.kill_hp_multiplier != 1 or attacker.kill_attack_multiplier != 1:
                _scale(hp, max_hp, attack, actor, a, attacker.kill_hp_multiplier, attacker.kill_attack_multiplier, r)
            if attacker.heal_on_kill:
                before = hp[actor][a]
                hp[actor][a] = max_hp[actor][a]
                _event(trace, "ON_KILL", side=actor, effect="heal_to_full", hp=hp[actor][a], healed=hp[actor][a] - before)
            if attacker.kill_heal_max_hp_fraction:
                _heal(hp, max_hp, actor, a, max_hp[actor][a] * attacker.kill_heal_max_hp_fraction, options, trace, "ON_KILL_HEAL")
            if attacker.kill_ally_stat_multiplier != 1:
                _scale_allies(battle, hp, max_hp, attack, order, active, actor, attacker.kill_ally_stat_multiplier, r)
            if attacker.kill_stat_add_base:  # Awakened cultivators: +50% of base stats per kill (IMG_0327: x1.6 -> x2.1).
                gain_hp = _round(attacker.hp * attacker.kill_stat_add_base, r)
                hp[actor][a] += gain_hp
                max_hp[actor][a] += gain_hp
                attack[actor][a] = _round(attack[actor][a] + attacker.attack * attacker.kill_stat_add_base, r)
            if attacker.kill_summon_form:  # Dr. Frankenstein: a Frankenstein with its stats joins the back.
                _summon(battle, hp, max_hp, attack, order, status, actor, max_hp[actor][a], attack[actor][a],
                        POOL_CODE + attacker.kill_summon_form - 1)
            if target.curse_killer:
                ext['curse'][actor][a] = CURSE_PERMANENT  # The Hanged Man (user): only its killer is cursed.
            if attacker.kill_extends_undying and undying[actor][a]:
                undying[actor][a] += 2  # Walking Dead: each kill buys another turn.
        active[enemy] += 1
        _on_death(battle, hp, max_hp, attack, order, active, dodges, enemy, d, options.stat_rounding, trace, status, chance, stored)
        queue = [side for side in queue if side != enemy]  # Cancel a dead card's queued entry hit.
        changed.append(enemy)
    if attacker_died:
        _event(trace, "DEATH", side=actor, slot=active[actor], reason="retaliation")
        active[actor] += 1
        _on_death(battle, hp, max_hp, attack, order, active, dodges, actor, a, options.stat_rounding, trace, status, chance, stored)
        queue = [side for side in queue if side != actor]
        changed.append(actor)
    # User-observed Inari: expiration follows the final attack's counter.
    normal_slot = state.order[normal_actor][state.active[normal_actor]]
    normal_fighter = battle.teams[normal_actor][normal_slot]
    if stole and normal_actor == actor:
        # User: Mother of Beasts uses a stolen ability at once (Priest's extra action right after the kill).
        normal_fighter = _view(base_battle, ability, suppressed, order, active, hp, ext['ability2'], failed).teams[normal_actor][normal_slot]
    lifetime = normal_fighter.lifetime_actions
    turn_end = not entry_context and not displaced_actor
    if turn_end and normal_fighter.action_end_heal_max_hp_fraction:
        _heal(hp, max_hp, normal_actor, normal_slot,
              max_hp[normal_actor][normal_slot] * normal_fighter.action_end_heal_max_hp_fraction,
              options, trace, "TURN_END_HEAL")
    if (turn_end and hp[normal_actor][normal_slot] > 0
            and normal_fighter.action_end_attack_multiplier != 1):
        before_attack = attack[normal_actor][normal_slot]
        scaled = before_attack * normal_fighter.action_end_attack_multiplier
        if not math.isfinite(scaled):
            raise ValueError("Turn-end attack calculation overflowed")
        attack[normal_actor][normal_slot] = _round(scaled, options.stat_rounding)
        _event(trace, "TURN_END_STAT", side=normal_actor, slot=state.active[normal_actor], stat="attack",
               before=before_attack, attack=attack[normal_actor][normal_slot],
               displayed_attack=math.ceil(attack[normal_actor][normal_slot]))
    first_action = not state.turn_actions_remaining
    if turn_end:
        triggered[normal_actor][normal_slot] &= ~SHIFTED
    if turn_end and _counts_turns(normal_fighter) and first_action:
        turns[normal_actor][normal_slot] += 1
    turn_index = turns[normal_actor][normal_slot] - 1
    if turn_end and hp[normal_actor][normal_slot] > 0:
        r = options.stat_rounding
        f = normal_fighter
        if f.action_end_stat_multiplier != 1 and (not f.action_end_stat_turns or turn_index < f.action_end_stat_turns):
            _scale(hp, max_hp, attack, normal_actor, normal_slot, f.action_end_stat_multiplier, f.action_end_stat_multiplier, r)
        if f.periodic_stat_period and first_action and (turn_index + 1) % f.periodic_stat_period == 0:
            _scale(hp, max_hp, attack, normal_actor, normal_slot, f.periodic_stat_multiplier, f.periodic_stat_multiplier, r)
        if f.action_end_attack_add_base:
            attack[normal_actor][normal_slot] = _round(attack[normal_actor][normal_slot] + f.attack * f.action_end_attack_add_base, r)
        if f.stat_gamble and not skipped:  # "After attacking": a skipped turn does not count.
            factor = STAT_GAMBLE_FACTORS[-1]
            for k, step in enumerate(STAT_GAMBLE_FACTORS[:-1]):
                if chance.roll(1 / (len(STAT_GAMBLE_FACTORS) - k)):
                    factor = step
                    break
            _scale(hp, max_hp, attack, normal_actor, normal_slot, 1, factor, r)
            _event(trace, "STAT_GAMBLE", side=normal_actor, factor=factor)
        if f.action_end_self_slow_turns and not skipped:
            slowed[normal_actor][normal_slot] = max(slowed[normal_actor][normal_slot], f.action_end_self_slow_turns)
        if f.block_max_hp_fraction and not skipped:
            block = max_hp[normal_actor][normal_slot] * f.block_max_hp_fraction
            block_pool[normal_actor][normal_slot] = min(block_pool[normal_actor][normal_slot] + block, block * f.block_max)
    if turn_end and hp[normal_actor][normal_slot] > 0:
        f = normal_fighter
        if f.action_end_ally_stat_multiplier != 1:
            _scale_allies(battle, hp, max_hp, attack, order, active, normal_actor, f.action_end_ally_stat_multiplier, options.stat_rounding)
        if f.action_end_incoming_multiplier != 1:
            inscale[normal_actor][normal_slot] *= f.action_end_incoming_multiplier
        if f.action_end_self_hp_loss_fraction:
            hp[normal_actor][normal_slot] -= _round(hp[normal_actor][normal_slot] * f.action_end_self_hp_loss_fraction, options.rounding)
    if turn_end and hp[normal_actor][normal_slot] > 0 and normal_fighter.action_end_hp_multiplier != 1:
        _scale(hp, max_hp, attack, normal_actor, normal_slot, normal_fighter.action_end_hp_multiplier, 1, options.stat_rounding)
    if turn_end and hp[normal_actor][normal_slot] > 0:
        f, r = normal_fighter, options.stat_rounding
        if f.growth_add_base and turn_index < f.growth_turns:  # Cultivators: +30% of base stats for 2 turns.
            growth = (1 + f.growth_add_base * (turn_index + 1)) / (1 + f.growth_add_base * turn_index)
            _scale(hp, max_hp, attack, normal_actor, normal_slot, growth, growth, r)
        if triggered[normal_actor][normal_slot] & SANTA:
            _scale(hp, max_hp, attack, normal_actor, normal_slot, SANTA_DECAY, SANTA_DECAY, r)
        if f.ally_class_stat_multiplier != 1:  # Longmu's Safeguarding: dragons waiting in the deck +20% stats.
            for position in range(active[normal_actor] + 1, len(order[normal_actor])):
                ally = order[normal_actor][position]
                if hp[normal_actor][ally] > 0 and battle.teams[normal_actor][ally].class_mask & f.ally_class_mask:
                    _scale(hp, max_hp, attack, normal_actor, ally, f.ally_class_stat_multiplier, f.ally_class_stat_multiplier, r)
        if ext['timer'][normal_actor][normal_slot] and f.entry_global_steal:
            ext['timer'][normal_actor][normal_slot] -= 1
            if not ext['timer'][normal_actor][normal_slot]:  # Eonus: the stolen stats decay.
                hp[normal_actor][normal_slot] = max(1.0, hp[normal_actor][normal_slot] - ext['bonus_hp'][normal_actor][normal_slot])
                max_hp[normal_actor][normal_slot] -= ext['bonus_hp'][normal_actor][normal_slot]
                attack[normal_actor][normal_slot] -= ext['bonus_attack'][normal_actor][normal_slot]
                ext['bonus_hp'][normal_actor][normal_slot] = ext['bonus_attack'][normal_actor][normal_slot] = 0.0
    if turn_end:
        other = 1 - normal_actor
        if active[other] < len(order[other]):
            pw = order[other][active[other]]
            drain = battle.teams[other][pw].enemy_turn_end_attack_drain
            if drain and hp[other][pw] > 0 and hp[normal_actor][normal_slot] > 0:
                take = _round(attack[normal_actor][normal_slot] * drain, options.stat_rounding)  # Poison Witch
                attack[normal_actor][normal_slot] -= take
                _heal(hp, max_hp, other, pw, take, options, trace, "POISON_WITCH")
        for c in order[normal_actor]:
            card = battle.teams[normal_actor][c]
            if card.doctor_period and hp[normal_actor][c] > 0:  # Divine Doctor: every 3 allied turns, all allies to full HP.
                ext['timer'][normal_actor][c] += 1
                if ext['timer'][normal_actor][c] % card.doctor_period == 0:
                    for ally in order[normal_actor]:
                        if hp[normal_actor][ally] > 0:
                            hp[normal_actor][ally] = max(hp[normal_actor][ally], max_hp[normal_actor][ally])
    if turn_end and lifetime and normal_actor not in changed:
        completed[normal_actor][normal_slot] += 1
        _event(trace, "ACTION_END", side=normal_actor, slot=state.active[normal_actor],
               completed_actions=completed[normal_actor][normal_slot])
        if completed[normal_actor][normal_slot] >= lifetime:
            hp[normal_actor][normal_slot] = 0.0
            _event(trace, "DEATH", side=normal_actor, slot=state.active[normal_actor], reason="lifetime")
            active[normal_actor] += 1
            _on_death(battle, hp, max_hp, attack, order, active, dodges, normal_actor, normal_slot, options.stat_rounding, trace, status, chance, stored)
            queue = [side for side in queue if side != normal_actor]
            changed.append(normal_actor)
    if (turn_end and triggered[normal_actor][normal_slot] & SAM_BOOST and not skipped and normal_actor not in changed
            and hp[normal_actor][normal_slot] > 0):
        hp[normal_actor][normal_slot] = 0.0  # Uncle Sam's boost: the card dies after attacking.
        _event(trace, "DEATH", side=normal_actor, slot=state.active[normal_actor], reason="uncle_sam")
        active[normal_actor] += 1
        _on_death(battle, hp, max_hp, attack, order, active, dodges, normal_actor, normal_slot, options.stat_rounding, trace, status, chance, stored)
        queue = [side for side in queue if side != normal_actor]
        changed.append(normal_actor)
    if turn_end and normal_actor not in changed and hp[normal_actor][normal_slot] > 0:
        if (not skipped and normal_fighter.dutchman_swaps and ext['timer'][normal_actor][normal_slot] < normal_fighter.dutchman_swaps
                and not triggered[normal_actor][normal_slot] & DUTCH_AWAY):
            waiting = _waiting(hp, order, active, normal_actor)
            if waiting:  # Flying Dutchman (user): swaps with a random ally, at most twice.
                p = waiting[chance.pick(len(waiting))]
                ally = order[normal_actor][p]
                order[normal_actor][active[normal_actor]], order[normal_actor][p] = ally, normal_slot
                ext['timer'][normal_actor][normal_slot] += 1
                triggered[normal_actor][normal_slot] |= DUTCH_AWAY
                triggered[normal_actor][ally] |= DUTCH_ALLY
                changed.append(normal_actor)
        elif triggered[normal_actor][normal_slot] & DUTCH_ALLY:
            triggered[normal_actor][normal_slot] &= ~DUTCH_ALLY  # The ally had its turn: the Dutchman swaps back.
            for p in range(active[normal_actor] + 1, len(order[normal_actor])):
                dutch = order[normal_actor][p]
                if triggered[normal_actor][dutch] & DUTCH_AWAY and hp[normal_actor][dutch] > 0:
                    triggered[normal_actor][dutch] &= ~DUTCH_AWAY
                    order[normal_actor][active[normal_actor]], order[normal_actor][p] = dutch, normal_slot
                    changed.append(normal_actor)
                    break
    if turn_end and not skipped and normal_fighter.swap_enemies and hp[normal_actor][normal_slot] > 0:
        other = 1 - normal_actor  # Marionette: swaps 2 random enemies (positions and stats), -15% each.
        living = [p for p in range(active[other], len(order[other])) if hp[other][order[other][p]] > 0]
        if len(living) >= 2:
            i = chance.pick(len(living))
            j = chance.pick(len(living) - 1)
            j += j >= i
            pi, pj = living[i], living[j]
            order[other][pi], order[other][pj] = order[other][pj], order[other][pi]
            for p in (pi, pj):
                m = normal_fighter.swap_stat_multiplier
                _scale(hp, max_hp, attack, other, order[other][p], m, m, options.stat_rounding)
            if active[other] in (pi, pj) and other not in changed:
                changed.append(other)
    changed.extend(side for side in relocated if side not in changed)
    kill_extra = (enemy in changed and actor == normal_actor and not entry_context and not is_counter
                  and not attacker_died and attacker.kill_extra_action and not displaced_actor)
    turn_ends = not entry_context and not kill_extra and (normal_actor in changed or displaced_actor or not (
        (state.turn_actions_remaining or ext['extra'][normal_actor][normal_slot] + _allowance(normal_fighter, turn_index, faded[normal_actor][normal_slot],
                   hp[normal_actor][normal_slot] < max_hp[normal_actor][normal_slot] * normal_fighter.low_hp_actions_threshold)) - 1))
    if turn_ends:
        _cleanse(battle, hp, order, status)
        for side in (0, 1):
            if suppressed[side]:
                suppressed[side] -= 1  # Fuxi's suppression wears off.
        for side in (normal_actor, 1 - normal_actor):
            # Waiting cards: Loch Ness growth and Zombie Dragon's countdown.
            for position, index in [(p, order[side][p]) for p in range(active[side] + 1, len(order[side]))]:
                if hp[side][index] <= 0:
                    continue
                if side == normal_actor and battle.teams[side][index].bench_growth_multiplier != 1:
                    # Loch Ness (IMG_0303: x1.1 per allied turn while waiting), capped relative to base stats.
                    card = battle.teams[side][index]
                    grow = card.bench_growth_multiplier
                    hp_factor = min(grow, card.bench_growth_cap * card.hp / max_hp[side][index]) if card.bench_growth_cap else grow
                    attack_factor = (min(grow, card.bench_growth_cap * card.attack / attack[side][index])
                                     if card.bench_growth_cap and attack[side][index] else grow)
                    if hp_factor > 1 or attack_factor > 1:
                        _scale(hp, max_hp, attack, side, index, max(1.0, hp_factor), max(1.0, attack_factor), options.stat_rounding)
                before_ticks = hp[side][index]
                if undying[side][index]:
                    undying[side][index] -= 1
                    if not undying[side][index]:
                        hp[side][index] = 0.0
                if death_timer[side][index]:
                    death_timer[side][index] -= 1
                    if not death_timer[side][index]:
                        hp[side][index] = 0.0
                if hp[side][index] <= 0 < before_ticks:
                    _event(trace, "DEATH", side=side, slot=position, reason="bench")
                    _on_death(battle, hp, max_hp, attack, order, active, dodges, side, index, options.stat_rounding, trace,
                              status, chance, stored, in_deck=True)
        for side in (normal_actor, 1 - normal_actor):
            if active[side] >= len(order[side]):
                continue
            slot = order[side][active[side]]
            hp_before_ticks = hp[side][slot]
            for kind, fraction in ((burn, BURN_MAX_HP_FRACTION), (bleed, BLEED_MAX_HP_FRACTION)):
                if kind[side][slot]:
                    kind[side][slot] -= 1
                    tick = _round(max_hp[side][slot] * fraction, options.rounding)
                    _ability_hit(battle, hp, attack, order, active, side, slot, tick, options.rounding)
                    _event(trace, "STATUS_TICK", side=side, status="burn" if kind is burn else "bleed", damage=tick, hp=hp[side][slot])
            if frozen[side][slot]:
                frozen[side][slot] -= 1
            if confused[side][slot] and confused[side][slot] < CONFUSION_UNTIL_SELF_HIT:
                confused[side][slot] -= 1
            if frostbite[side][slot]:
                frostbite[side][slot] -= 1
                if chance.roll(FROSTBITE_CHANCE):
                    tick = _round(max_hp[side][slot] * FROSTBITE_MAX_HP_FRACTION, options.rounding)
                    _ability_hit(battle, hp, attack, order, active, side, slot, tick, options.rounding)
                    slowed[side][slot] += 1  # Video 0290: the frostbitten card loses its next turn.
                    _event(trace, "STATUS_TICK", side=side, status="frostbite", damage=tick, hp=hp[side][slot])
            if undying[side][slot]:
                undying[side][slot] -= 1
                if not undying[side][slot]:
                    hp[side][slot] = 0.0  # Zombie Dragon: its turns at 0 HP are over.
                    _event(trace, "DOOM", side=side)
            if death_timer[side][slot]:
                death_timer[side][slot] -= 1
                if not death_timer[side][slot]:
                    hp[side][slot] = 0.0  # Velociraptor's timer ends.
                    _event(trace, "DOOM", side=side)
            if hp[side][slot] <= 0 and hp[side][slot] < hp_before_ticks:
                _event(trace, "DEATH", side=side, slot=active[side], reason="status")
                active[side] += 1
                _on_death(battle, hp, max_hp, attack, order, active, dodges, side, slot, options.stat_rounding, trace, status, chance, stored)
                queue = [q for q in queue if q != side]
                if side not in changed:
                    changed.append(side)
    if changed or intercepted is not None:
        actions = 0
    elif not entry_context and actions >= 2 * options.repeat_cycles:
        for side in (0, 1):
            dead = order[side][active[side]]
            hp[side][dead] = 0.0
            _event(trace, "DEATH", side=side, slot=active[side], reason="repeat_limit")
            active[side] += 1
            _on_death(battle, hp, max_hp, attack, order, active, dodges, side, dead, options.stat_rounding, trace, status, chance, stored)
            changed.append(side)
        actions = 0
    for side in (0, 1):
        # Waiting cards killed in the deck were already reported; the next living card plays.
        while active[side] < len(order[side]) and hp[side][order[side][active[side]]] <= 0 and active[side] > state.active[side]:
            active[side] += 1
    battle = _view(base_battle, ability, suppressed, order, active, hp, ext['ability2'], failed)
    outcome = _terminal(battle, active, order)
    if outcome is not None:
        _event(trace, "END_CHECK", outcome=("A", "B", "tie")[outcome])
        return None, outcome
    for side in sorted(changed):
        _event(trace, "TRANSITION", side=side, next_slot=active[side])
        first = _entry(battle, hp, max_hp, attack, active, order, entered, stolen, status, side, options.stat_rounding, trace, chance)
        if first and _queues_entry_hit(base_battle, status, order, active, hp, side):
            queue.append(side)
    left = normal_actor in changed or displaced_actor
    next_actor = normal_actor
    turn_remaining = state.turn_actions_remaining
    if not entry_context:
        turn_remaining = (turn_remaining or ext['extra'][normal_actor][normal_slot] + _allowance(normal_fighter, turn_index, faded[normal_actor][normal_slot],
                   hp[normal_actor][normal_slot] < max_hp[normal_actor][normal_slot] * normal_fighter.low_hp_actions_threshold)) - 1
        if not state.turn_actions_remaining:
            ext['extra'][normal_actor][normal_slot] = 0
        if kill_extra:
            turn_remaining += 1  # Immediately attacks again after a kill.
        elif left or not turn_remaining:
            next_actor, turn_remaining = 1-normal_actor, 0
    elif left and turn_remaining:
        # An already-earned allowance belongs to the deceased card, not its replacement.
        next_actor, turn_remaining = 1-normal_actor, 0
    return _freeze(hp, max_hp, attack, used, survivals, completed, order, entered, dodges, stolen, turns, hits, own_dodges, burn, bleed, frozen, slowed, frostbite, confused, charged, revived, guarded, faded, resting, debuffed, marked, inscale, stored, poison, poison_damage, weakness, triggered, undying, doom, strike_timer, strike_damage, blind, block_pool, death_timer, fossils, ability, suppressed, ext, active, next_actor, actions,
                   entry_queue=queue, turn_actions_remaining=turn_remaining), None


def _hit_chances(battle, state):
    """Dodge and critical probabilities for the next hit."""
    base = battle
    battle = _view(battle, state.ability, state.suppressed, state.order, state.active, state.hp, state.ext[0])
    acting_side = _hit_actor(state)
    target_side = 1 - acting_side
    target_index = state.order[target_side][state.active[target_side]]
    defender = battle.teams[target_side][target_index]
    p = defender.dodge_probability
    if defender.dodge_growth:
        p = min(defender.dodge_cap, p + defender.dodge_growth * state.hits_taken[target_side][target_index])
    p *= 1 - _fail_chance(base, state.ext, acting_side)  # End Times: a chance ability can fail too.
    hitter = battle.teams[acting_side][state.order[acting_side][state.active[acting_side]]]
    followup = bool(hitter.followup_multiplier and not state.counter_pending and not state.entry_queue
                    and state.hits_remaining == 1)
    li = None if hitter.domain else _luminant(battle, state.hp, state.order, target_side, target_index)
    if li is not None and state.ext[5][target_side][target_index] < LUMINANT_MAX:
        # Eclipseborn Luminant: allies get 40% evasion, -10% per evade (20% floor), at most 2 evades each.
        q = max(LUMINANT_FLOOR, battle.teams[target_side][li].luminant_evasion - LUMINANT_STEP * state.ext[5][target_side][target_index])
        p = 1 - (1 - p) * (1 - q)
    if (hitter.bypass_defenses or (hitter.entry_bypass_defenses and state.entry_queue and not state.counter_pending)
            or (followup and hitter.followup_bypass)):
        p = 0.0  # User: bypass also defeats chance dodges.
    return p, 0.0 if followup else hitter.critical_probability * (1 - _fail_chance(base, state.ext, target_side))


MIN_ROLLOUTS = 8
COLLAPSE_SEED = 0xC0FFEE1234567890  # stream for sampled (collapsed) chance rolls in branch mode


def _rollout(battle, pool, options, steps, playouts):
    """Playouts for one step's pruned states: (outcome masses [A, B, tie, unfinished], playouts used).

    Each playout starts from a state drawn in proportion to its mass (an unbiased estimate of the pool),
    so the cost depends on the pool's total mass M, not on how many states it holds. A fixed `rollouts`
    playouts, or with rollout_error e: at least MIN_ROLLOUTS and at most `rollouts`, stopping once
    M * max_k p_k(1-p_k)/n <= e^2 (p_k = (c_k+1)/(n+2)); summed over steps this bounds the result's
    variance by e^2."""
    total = sum(mass for _, mass in pool)
    counts = [0, 0, 0, 0]
    n = 0
    while n < options.rollouts:
        rng = _RNG(_playout_seed(options.seed, playouts + n))
        target, acc, start = rng.random() * total, 0.0, pool[-1][0]
        for state, mass in pool:
            acc += mass
            if target < acc:
                start = state
                break
        outcome = _playout(battle, start, options, rng, steps)
        counts[3 if outcome is None else outcome] += 1
        n += 1
        if options.rollout_error and n >= MIN_ROLLOUTS:
            spread = max((c + 1.0) / (n + 2.0) * (1.0 - (c + 1.0) / (n + 2.0)) for c in counts)
            if total * spread / n <= options.rollout_error * options.rollout_error:
                break
    return total, [total * c / n for c in counts], n


def _playout_seed(seed, counter):
    value = (seed ^ ((0x9E3779B97F4A7C15 * (counter + 1)) & ((1 << 64) - 1))) & ((1 << 64) - 1)
    return value or 0x9E3779B97F4A7C15


def _playout(battle, state, options, rng, steps):
    """One sampled continuation; returns the outcome index or None if unfinished."""
    for _ in range(steps):
        p, critical_chance = _hit_chances(battle, state)
        dodged = bool(p == 1 or (0 < p < 1 and rng.random() < p))
        critical = not dodged and bool(critical_chance == 1 or (0 < critical_chance < 1 and rng.random() < critical_chance))
        state, terminal = _advance(battle, state, dodged, options, None, critical, _Chance(rng=rng))
        if terminal is not None:
            return terminal
    return None


def simulate(battle: Battle, options: Options | None = None, *, trace: bool = False) -> Result:
    options = options or Options()
    options.validate()
    battle.validate()
    if trace and options.mode != "sample":
        raise ValueError("A linear trace requires sample mode")
    events = [] if trace else None
    rng = _RNG(options.seed)
    sample_rng = rng if options.mode == "sample" else None
    collapse = ([options.sample_below, _RNG(options.seed ^ COLLAPSE_SEED), 0.0]
                if options.mode == "branch" and options.sample_below else None)
    frontier, merged = {}, 0
    for state, weight in _expand(lambda chance: _initial(battle, options, events, chance), 1.0, sample_rng, collapse):
        merged += state in frontier
        frontier[state] = frontier.get(state, 0.0) + weight
    outcome_mass = [0.0, 0.0, 0.0]
    unresolved, expanded = 0.0, 0
    estimated, playouts = 0.0, 0
    for step_index in range(options.max_steps):
        following = {}
        for state, mass in frontier.items():
            expanded += 1
            acting_side = _hit_actor(state)
            target_side = 1 - acting_side
            p, critical_chance = _hit_chances(battle, state)
            if options.mode == "sample":
                dodged = bool(p == 1 or (0 < p < 1 and rng.random() < p))
                critical = not dodged and bool(critical_chance == 1 or (0 < critical_chance < 1 and rng.random() < critical_chance))
                outcomes = [(dodged, critical, 1.0)]
            else:
                outcomes = [(dodged, critical, probability) for dodged, critical, probability in (
                    (True, False, p), (False, True, (1-p)*critical_chance),
                    (False, False, (1-p)*(1-critical_chance))) if probability > 0]
            for dodge, critical, probability in outcomes:
                weight = mass * probability
                if weight == 0:
                    continue
                step = lambda chance, dodge=dodge, critical=critical: _advance(
                    battle, state, dodge, options, events, critical, chance)
                for (new_state, terminal), part in _expand(step, weight, sample_rng, collapse):
                    if terminal is not None:
                        outcome_mass[terminal] += part
                    else:
                        if new_state in following:
                            merged += 1
                        following[new_state] = following.get(new_state, 0.0) + part
        frontier = {}
        spent = options.mode == "branch" and bool(options.node_budget) and expanded >= options.node_budget
        pool = []
        # Stable ordering for equal mass makes capped searches reproducible.
        for state, mass in sorted(following.items(), key=lambda item: -item[1]):
            if options.mode == "branch" and (spent or mass < options.prune_probability or len(frontier) >= options.max_frontier):
                if not options.rollouts:
                    unresolved += mass
                else:
                    pool.append((state, mass))
            else:
                frontier[state] = mass
        if pool:
            # Monte Carlo estimate of the pruned branches, drawn by mass.
            total, shares, used = _rollout(battle, pool, options, options.max_steps - step_index - 1, playouts)
            estimated += total
            playouts += used
            for outcome in range(3):
                outcome_mass[outcome] += shares[outcome]
            unresolved += shares[3]
        if not frontier:
            break
    unresolved += sum(frontier.values())
    if collapse is not None:
        estimated += collapse[2]
    estimated = min(estimated, 1.0)  # a sampled path counts at every later step; only zero vs nonzero matters
    result = Result(*outcome_mass, unresolved, expanded, merged, trace=tuple(events or ()), estimated=estimated)
    if not math.isclose(result.accounted_probability, 1, rel_tol=1e-10, abs_tol=1e-10):
        raise ArithmeticError("Probability mass was lost")
    return result


def simulate_batch(battles, options=None):
    """Reference implementation; production batching uses the single native call."""
    options = options or Options()
    return [simulate(battle, replace(options, seed=(options.seed + index) % 2**64)) for index, battle in enumerate(battles)]
