"""Conservative full-card compilation for the initial primitive subset.

Source descriptions support these primitive interpretations, but shared battle
timing and rounding are still provisional. No partial ability is compiled.
"""

from dataclasses import fields, replace

from .reference import Battle, Fighter, POOL_FORM, POOL_GLAMOUR, POOL_LIMITED, MAX_FIGHTERS
from ..stats import base_stats


class UnsupportedCardError(ValueError):
    pass


# Each row encodes the complete combat effect in the full-information baseline.
# Revealing a hand is excluded from this subset rather than treated as a no-op.
SUPPORTED = {
    1: ("10 % chance to dodge attacks", {"dodge_probability": .1}),
    2: ("deduct 10 % max hp from damage", {"damage_reduction_max_hp": .1}),
    3: ("reduce enemy's damage by 15 % on entry", {"entry_enemy_attack_multiplier": .85}),
    5: ("damage taken is 2 x", {"incoming_multiplier": 2.0}),
    6: ("block first nonlethal hit", {"block_mode": 2}),
    7: ("heal 20 % hp after attack", {"after_attack_heal_max_hp_fraction": .2}),
    9: ("dodge attacks under 30 % of max hp", {"dodge_below_max_hp": .3}),
    10: ("survives lethal attack with 1 hp", {"lethal_survivals": 1}),
    14: ("half-damage attack on entry", {"entry_hit_multiplier": .5}),
    18: ("50 % chance to deal double damage", {"critical_probability": .5, "critical_multiplier": 2}),
    19: ("gain an extra turn", {"actions_per_turn": 2}),
    28: ("boosts damage and hp by 25 %", {"entry_self_multiplier": 1.25}),
    35: ("attack for 2.5 x damage on entry", {"entry_hit_multiplier": 2.5}),
    36: ("reduce enemy's damage by 30 % on entry", {"entry_enemy_attack_multiplier": .7}),
    38: ("reduce damage taken by 75 %", {"incoming_multiplier": .25}),
    44: ("boost damage and hp by 40 % on entry", {"entry_self_multiplier": 1.4}),
    47: ("boost hp by 200 % on entry", {"entry_hp_multiplier": 3}),
    59: ("damage cannot exceed 1 / 3 of max hp", {"damage_cap_max_hp": 1/3}),
    61: ("reduce damage by 15 % of max hp, then halve", {"damage_reduction_max_hp": .15, "incoming_multiplier": .5}),
    67: ("3 x attacks, each dealing 50 % damage", {"attacks_per_action": 3, "outgoing_multiplier": .5}),
    71: ("halve hp, 2 x atk", {"entry_hp_multiplier": .5, "entry_attack_multiplier": 2}),
    72: ("attack on entry", {"entry_hit_multiplier": 1}),
    73: ("block first attack", {"block_mode": 1}),
    74: ("heal 70 % of damage taken", {"heal_damage_taken_fraction": .7}),
    85: ("boost stats by 20 % on entry; heal 10 % hp for each turn",
         {"entry_self_multiplier": 1.2, "action_start_heal_max_hp_fraction": .1}),
    87: ("nullify damage below 75 % of self atk", {"nullify_below_attack": .75}),
    93: ("60 % chance to dodge attacks", {"dodge_probability": .6}),
    94: ("heal 35 % hp each turn", {"action_start_heal_max_hp_fraction": .35}),
    104: ("if card ahead would take lethal attack: self takes its place, blocks attack, and gains 2 x stats",
          {"intercept_lethal_multiplier": 2}),
    114: ("invincible for 3 turns then dies", {"invincible": 1, "lifetime_actions": 3}),
    121: ("gain 30 % atk each turn end", {"action_end_attack_multiplier": 1.3}),
    123: ("heal to full hp on kill", {"heal_on_kill": 1}),
    131: ("boost stats by 65 % on entry", {"entry_self_multiplier": 1.65}),
    162: ("deal 1 x damage and take 1 x damage", {}),
    164: ("on death, the next card will dodge an attack", {"next_card_dodges": 1}),
    165: ("take 1.5 x damage; deal 1.5 x damage", {"incoming_multiplier": 1.5, "outgoing_multiplier": 1.5}),
    166: ("5 x attacks, each dealing 1 / 4 damage", {"attacks_per_action": 5, "outgoing_multiplier": .25}),
    174: ("heal 50 % of the damage dealt", {"heal_damage_dealt_fraction": .5}),
    182: ("reduce enemy's damage by 40 % on entry", {"entry_enemy_attack_multiplier": .6}),
    205: ("steal 10 % of enemy's atk and hp on their entry", {"enemy_entry_steal_fraction": .1}),
    210: ("counterattack when damaged", {"counter_on_damage": 1}),
    261: ("on entry, attack doubles", {"entry_attack_multiplier": 2}),
    264: ("damage cannot exceed 1 / 2 max hp", {"damage_cap_max_hp": .5}),
    288: ("attack 8 times at half the damage", {"attacks_per_action": 8, "outgoing_multiplier": .5}),
    # Batch 1: extrapolated from card text for breadth; each mapping is provisional.
    11: ('increase damage by 25 % each time attacked',
          {"damaged_attack_multiplier": 1.25}),
    13: ('attacker loses hp equal to damage dealt',
          {"reflect_damage_fraction": 1}),
    17: ('steal 30 % hp and atk on kill',
          {"kill_steal_fraction": 0.3}),
    34: ('attacker takes damage equal to 75 % of self atk',
          {"thorns_attack_fraction": 0.75}),
    39: ("steal 10 % of enemy's hp and atk after damaging",
          {"hit_steal_fraction": 0.1}),
    46: ("half enemy's stats on entry",
          {"entry_enemy_stat_multiplier": 0.5}),
    58: ('enemy loses 25 % of their damage turn start',
          {"enemy_turn_start_attack_multiplier": 0.75}),
    60: ('deal 3 x damage, lose 30 % atk after attack',
          {"outgoing_multiplier": 3, "action_end_attack_multiplier": 0.7}),
    66: ('enemy atk drops 10 % after attack',
          {"hit_target_attack_multiplier": 0.9}),
    70: ('gain 50 % lifesteal and counterattack when attacked',
          {"heal_damage_dealt_fraction": 0.5, "counter_on_damage": 1}),
    78: ('on death, shield the next card from one attack',
          {"next_card_dodges": 1}),
    86: ("reduce target's atk by 40 % of damage dealt",
          {"hit_target_attack_reduction_fraction": 0.4}),
    91: ('steal atk and hp equal to damage dealt',
          {"hit_hp_gain_fraction": 1, "hit_target_max_hp_reduction_fraction": 1, "hit_attack_drain_fraction": 1}),
    112: ('enemy loses 20 % max hp each turn',
          {"enemy_turn_start_max_hp_loss_fraction": 0.2}),
    115: ('lose 1 / 8 stats when damaged',
          {"damaged_stat_multiplier": 0.875}),
    127: ('enemy loses 30 % of their current hp turn start',
          {"enemy_turn_start_hp_loss_fraction": 0.3}),
    160: ('self gains + 30 % damage each turn',
          {"action_end_attack_multiplier": 1.3}),
    163: ("reduce enemy's hp by 15 % on entry",
          {"entry_enemy_hp_multiplier": 0.85}),
    # User: the text is right; Maw seeming to lose HP was a display bug (25% of max HP, as observed).
    171: ('the enemy loses 25 % of its hp each turn',
          {"enemy_turn_start_max_hp_loss_fraction": 0.25}),
    176: ('heal 25 % hp after attacked',
          {"damaged_heal_max_hp_fraction": 0.25}),
    # User: nerfed to 10% (IMG_0330: Malik 655,360 -> 589,824 -> 530,842).
    180: ("attacks on entry; attacks reduce the target's damage by 10 %",
          {"entry_hit_multiplier": 1, "hit_target_attack_multiplier": 0.9}),
    185: ('defeating an enemy fully restores hp and permanently increases damage by 50 %, stacking with each kill',
          {"heal_on_kill": 1, "kill_attack_multiplier": 1.5}),
    207: ('heal and gain damage by 20 % on kill',
          {"kill_heal_max_hp_fraction": 0.2, "kill_attack_multiplier": 1.2}),
    209: ('heal 200 % of atk as hp each turn, lose 10 % atk',
          {"action_start_heal_attack_fraction": 2, "action_end_attack_multiplier": 0.9}),
    213: ('hp boosted by 100 x; when attacked counterattack at 10 % damage',
          {"counter_on_damage": 1, "counter_multiplier": 0.1}),
    219: ('2 x hp/atk on entry but lose 15 % each turn',
          {"entry_self_multiplier": 2, "action_end_attack_multiplier": 0.85, "action_end_hp_multiplier": 0.85}),
    222: ('add 1 / 2 stats to next card on death',
          {"next_card_stat_add_fraction": 0.5}),
    226: ("self max hp gets added to next ally's on death",
          {"next_card_max_hp_add_fraction": 1}),
    229: ("steal 50 % of the enemy's stats and heal 30 % hp when defeating them",
          {"kill_steal_fraction": 0.5, "kill_heal_max_hp_fraction": 0.3}),
    238: ('deal 3 x damage, take 1 / 3 damage but lose 10 % stats per attack',
          {"outgoing_multiplier": 3, "incoming_multiplier": 1/3, "action_end_attack_multiplier": 0.9, "action_end_hp_multiplier": 0.9}),
    241: ('enemy loses 25 % stats enemy turn start',
          {"enemy_turn_start_stat_multiplier": 0.75}),
    255: ('attacker takes damage equal to 1 / 2 your attack',
          {"thorns_attack_fraction": 0.5}),
    256: ('steal 20 % of enemy stats upon being attacked',
          {"attacked_steal_fraction": 0.2}),
    262: ('on entry, gain 3 attacks; dealing 1 / 3 damage',
          {"attacks_per_action": 3, "outgoing_multiplier": 1/3}),
    266: ('heal and boost atk by 25 % of damage dealt',
          {"heal_damage_dealt_fraction": 0.25, "hit_attack_gain_fraction": 0.25}),
    271: ("reduce enemy's max hp by damage dealt",
          {"hit_target_max_hp_reduction_fraction": 1}),
    282: ('gain an extra turn; heal 10 % each turn',
          {"actions_per_turn": 2, "action_start_heal_max_hp_fraction": 0.1}),
    # Batch 2: extrapolated from card text; provisional.
    23: ('deal 50 % more damage and bypass defenses',
          {"outgoing_multiplier": 1.5, "bypass_defenses": 1}),
    24: ('deal 2 x damage, 1 turn to recharge',
          {"charged_attack_multiplier": 2, "recharge_turns": 1}),
    25: ('deal extra damage equal to 70 % of lost hp',
          {"lost_hp_damage_fraction": 0.7}),
    42: ('deal 2.5 x damage, 1 turn to recharge; bypasses defense abilities',
          {"charged_attack_multiplier": 2.5, "recharge_turns": 1, "bypass_defenses": 1}),
    54: ('dodge every other attack, starting first',
          {"alternate_dodge": 1}),
    62: ('on entry, gain attack equal to combined total of all fallen allies attack; takes 1.5 x damage',
          {"entry_fallen_attack_fraction": 1, "incoming_multiplier": 1.5}),
    82: ('increase stats by 10 % for the first 5 turns',
          {"action_end_stat_multiplier": 1.1, "action_end_stat_turns": 5}),
    89: ('start with 2 turns, then gain 1 each cycle',
          {"actions_per_turn": 2, "actions_growth_per_turn": 1}),
    96: ('deal 3 x damage, skip next turn',  # User: it does skip its next turn.
          {"outgoing_multiplier": 3, "action_end_self_slow_turns": 1}),
    97: ('take 3 turns, then 1',
          {"first_turn_actions": 3}),
    102: ('dodge first attack, bypass defenses',
          {"block_mode": 1, "bypass_defenses": 1}),
    105: ('deal 2 x damage on entry if an ally is dead',
          {"entry_hit_multiplier": 2, "entry_hit_requires_fallen": 1}),
    108: ('gain 20 % stats each turn for 9 turns',
          {"action_end_stat_multiplier": 1.2, "action_end_stat_turns": 9}),
    113: ('dodge first lethal hit then 2 x damage',
          {"own_lethal_dodges": 1, "lethal_dodge_attack_multiplier": 2}),
    118: ('nullify every other attack and heal 80 % of its damage',
          {"alternate_dodge": 2, "alternate_dodge_heal_fraction": 0.8}),
    134: ('deal 150 % damage on entry, ignoring defenses',
          {"entry_hit_multiplier": 1.5, "entry_bypass_defenses": 1}),
    140: ('2 x hp and atk every 3 turns',
          {"periodic_stat_multiplier": 2, "periodic_stat_period": 3}),
    149: ('ignores defense abilities',
          {"bypass_defenses": 1}),
    156: ('deal 2 x damage if the enemy has 50 % or less hp; + 100 % atk on kill',
          {"weak_target_multiplier": 2, "weak_target_threshold": 0.5, "kill_attack_multiplier": 2}),
    158: ("add the enemy's attack to self damage",
          {"target_attack_damage_fraction": 1}),
    161: ('attacks convert 15 % max hp into damage',
          {"max_hp_damage_fraction": 0.15, "max_hp_damage_costs": 1}),
    # User Seraphim tests: three revives (one used up by the entry hit left two).
    169: ('self must be defeated 3 times before it can die',
          {"lethal_survivals": 3, "lethal_survival_hp_fraction": 1}),
    173: ('gain 30 % stats for each fallen ally; below 25 % hp, damage taken is halved',
          {"entry_fallen_stat_bonus": 0.3, "low_hp_incoming_multiplier": 0.5, "low_hp_incoming_threshold": 0.25}),
    179: ('attacks against enemies below 25 % hp will kill',
          {"execute_below_max_hp": 0.25}),
    188: ('for every 1 % of max hp lost, gain 1 % increased damage; additionally, gain 0.5 % lifesteal for every 1 % of max hp lost',
          {"lost_hp_damage_bonus": 1, "lost_hp_lifesteal": 0.5}),
    190: ('before attack, sacrifice 15 % of current hp to gain 30 % increased damage; defeating an enemy restores 20 % of max hp',
          {"self_turn_start_hp_loss_fraction": 0.15, "outgoing_multiplier": 1.3, "kill_heal_max_hp_fraction": 0.2}),
    193: ('deal 75 % increased damage while above the enemys hp; while below the enemys hp take 45 % reduced damage instead',
          {"advantage_damage_multiplier": 1.75, "disadvantage_incoming_multiplier": 0.55}),
    200: ('dodge first lethal attack and heal 50 % max hp',
          {"own_lethal_dodges": 1, "lethal_dodge_heal_max_hp_fraction": 0.5}),
    221: ('attack 3 times at 0.5 x, 1 x, and 2 x damage',
          {"attacks_per_action": 3, "outgoing_multiplier": 0.5, "hit_growth_multiplier": 2}),
    242: ('gain 100 % atk, then gain the initial boost each turn',
          {"entry_attack_multiplier": 2, "action_end_attack_add_base": 1}),
    243: ('gain 25 % atk for every card in ally party',
          {"entry_party_attack_bonus": 0.25}),
    244: ('deal 5 x damage once',
          {"first_attack_multiplier": 5}),
    246: ('gain 25 % stats for every fallen ally',
          {"entry_fallen_stat_bonus": 0.25}),
    248: ('when self falls below 30 % hp: both it and the enemy are defeated',
          {"mutual_destruction_below_max_hp": 0.3}),
    249: ('when the enemy falls below 30 % hp: it is instantly defeated',
          {"execute_after_below_max_hp": 0.3}),
    287: ('survives lethal attacks twice; attack resets to normal if it has been lowered',
          {"lethal_survivals": 2, "restore_lowered_attack": 1}),
    # Batch 3: statuses. Burn/bleed durations not in the text default to 3 turns.
    57: ('deal 4 x damage, 1 turn to recharge; on entry burn all enemies for 3 turns',
          {'charged_attack_multiplier': 4, 'recharge_turns': 1, 'entry_burn_turns': 3}),
    81: ('burn target for 3 turns when attacked',
          {'burn_on_attacked_turns': 3}),
    92: ('slow target on entry',
          {'entry_slow_turns': 1}),
    135: ('revive with 1 / 2 hp and 2 x atk; burn attackers for 2 turns',
          {'lethal_survivals': 1, 'lethal_survival_hp_fraction': 0.5, 'survival_attack_multiplier': 2, 'burn_on_attacked_turns': 2}),
    147: ('on entry freeze the enemy; attacks twice',
          {'entry_freeze_turns': 1, 'attacks_per_action': 2}),
    153: ('takes 25 % less damage; when attacked the enemy bleeds; counterattacks',
          {'incoming_multiplier': 0.75, 'bleed_on_attacked_turns': 3, 'counter_on_damage': 1}),
    167: ('burn enemy for 5 turns when attacked',
          {'burn_on_attacked_turns': 5}),
    198: ('counterattacks and sets enemy on fire',
          {'counter_on_damage': 1, 'hit_burn_turns': 3}),
    257: ('slow target on entry; attacks inflict bleed; attacks twice',
          {'entry_slow_turns': 1, 'hit_bleed_turns': 3, 'attacks_per_action': 2}),
    # Batch 4: chance mechanics. 50's 'infinite damage' is a 1e12x critical; 'cannot dodge
    # or gain shields' is not modelled. Frosty's frostbite lasts 1 turn (video 0290).
    15: ("30 % chance to skip enemy's turn",
          {"skip_enemy_turn_chance": 0.3}),
    21: ('50 % chance to freeze target for 1 turn',
          {"hit_freeze_turns": 1, "hit_freeze_chance": 0.5}),
    22: ('50 % chance to revive with 1 / 2 hp',
          {"chance_survival_probability": 0.5, "chance_survival_hp_fraction": 0.5}),
    29: ("enemy's hit chance is squared ratio of damage to self max hp",
          {"hit_chance_squared": 1}),
    31: ('50 % chance to revive with full hp',
          {"chance_survival_probability": 0.5, "chance_survival_hp_fraction": 1}),
    32: ('50 % chance to revive with full hp and freeze enemy',
          {"chance_survival_probability": 0.5, "chance_survival_hp_fraction": 1, "chance_survival_freeze_turns": 1}),
    50: ('50 % chance to deal infinite damage; cannot dodge or gain shields',
          {"critical_probability": 0.5, "critical_multiplier": 1000000000000.0}),
    63: ('attacks inflict 2 turns of frostbite; take 1 / 2 damage from enemies with frostbite',
          {"hit_frostbite_turns": 2, "frostbite_incoming_multiplier": 0.5}),
    64: ('on entry, 70 % chance to set enemy stats to its base borderless stats',
          {"entry_borderless_reset_chance": 0.7}),
    100: ("inflict confusion for 3 turns on entry; fails if enemy's atk is 400 % higher",
          {"entry_confusion_turns": 3, "confusion_fail_attack_ratio": 5}),
    148: ('40 % chance to dodge attacks; if it does, next attack deals 2 x damage',
          {"dodge_probability": 0.4, "dodge_bonus_multiplier": 2}),
    201: ('gain 20 % dodge chance and every hit taken increases the chance by 10 % up to 50 %',
          {"dodge_probability": 0.2, "dodge_growth": 0.1, "dodge_cap": 0.5}),
    216: ('deal 1 - 5 x damage at random',
          {"random_multiplier_max": 5}),
    220: ('40 % chance to dodge attacks, increase damage by 20 % on dodge',
          {"dodge_probability": 0.4, "dodge_attack_multiplier": 1.2}),
    232: ("deals 2 x damage; 50 % chance to skip enemy's turn",
          {"outgoing_multiplier": 2, "skip_enemy_turn_chance": 0.5}),
    245: ('inflicts death based on damage to hp ratio',
          {"ratio_kill": 1}),
    251: ('on death, inflict frostbite',
          {"death_frostbite_turns": 1}),
    252: ('on entry, randomly inflict 3-turn frostbite/slow/freeze',
          {"entry_random_debuff_turns": 3}),
    260: ('deal 4 x damage with an 80 % chance; heal the enemy 50 % otherwise',
          {"gamble_chance": 0.8, "gamble_multiplier": 4, "gamble_fail_heal_fraction": 0.5}),
    263: ('inflict confusion for 2 turns on entry',
          {"entry_confusion_turns": 2}),
    # Batch 5: fades, alternating rests, next-card/entry/turn-start effects, field and party
    # effects, rarity/pack/class matchups (classes are inferred from art, unverified).
    8: ("field: boost all cards' damage by 10 %",
          {"field_damage_multiplier": 1.1}),
    12: ('2 x damage to undead',
          {"bonus_vs_class_mask": 1, "class_damage_multiplier": 2}),
    20: ('deal 2 x damage, take half if border is rarer',
          {"border_advantage_multiplier": 2, "border_advantage_incoming_multiplier": 0.5}),
    40: ('50 % chance to transfer 1 / 2 stats to next card on death',
          {"next_card_stat_add_fraction": 0.5, "next_card_gift_chance": 0.5}),
    41: ("attack and steal 25 % of enemy's damage on entry",
          {"entry_hit_multiplier": 1, "entry_steal_attack_fraction": 0.25}),
    45: ('on entry attack with 3 x damage; + 10 % stats and damage reduction each turn',
          {"entry_hit_multiplier": 3, "turn_start_stat_growth": 0.1, "turn_start_incoming_reduction": 0.1, "incoming_reduction_cap": 0.5}),
    51: ("boost allies' stats by 25 % on kill; dodge normal attacks at the cost of 20 % hp",
          {"kill_ally_stat_multiplier": 1.25, "normal_attack_dodge_cost": 0.2}),
    55: ('gain 75 % atk on entry; when fighting a dragon: 2 x damage, 0.5 x enemy damage',
          {"entry_attack_multiplier": 1.75, "bonus_vs_class_mask": 4, "class_damage_multiplier": 2, "guard_vs_class_mask": 4, "class_incoming_multiplier": 0.5}),
    68: ('cards of reduce rarity deal 1 / 2 damage',
          {"lower_rarity_incoming_multiplier": 0.5}),
    84: ("boost next card's stats by 20 % on entry",
          {"entry_next_card_stat_multiplier": 1.2}),
    124: ('take 1 / 3 less damage from demons',
          {"guard_vs_class_mask": 2, "class_incoming_multiplier": 0.6666666666666666}),
    128: ("reduce damage by 35 %; boost allies' stats by 10 % each turn",
          {"incoming_multiplier": 0.65, "action_end_ally_stat_multiplier": 1.1}),
    132: ('gain 200 % atk on entry, lose atk equal to damage taken; heal 35 % and gain 35 % atk on kill',
          {"entry_attack_multiplier": 3, "damaged_attack_loss_fraction": 1, "kill_heal_max_hp_fraction": 0.35, "kill_attack_multiplier": 1.35}),
    136: ('heal 20 % hp each turn, with 50 % chance to gain 30 % atk',
          {"action_start_heal_max_hp_fraction": 0.2, "turn_start_attack_chance": 0.5, "turn_start_attack_multiplier": 1.3}),
    142: ('every other turn, gain a shield',
          {"block_mode": 1, "periodic_block_period": 2}),
    145: ('attack every other turn, dodges attack on turns it does not; 1.5 x damage',
          {"alternate_rest": 1, "outgoing_multiplier": 1.5}),
    150: ('40 % chance each turn to heal to full',
          {"turn_start_full_heal_chance": 0.4}),
    168: ("boost all cards' damage 20 %",
          {"field_damage_multiplier": 1.2}),
    170: ("deals damage equal to 50 % of target's current hp; loses 25 % hp after attack",
          {"outgoing_multiplier": 0, "target_current_hp_damage_fraction": 0.5, "action_end_self_hp_loss_fraction": 0.25}),
    181: ('+ 3 turns; fades below 50 % hp',
          {"fade_threshold": 0.5, "fade_extra_actions": 3}),
    187: ('the first attack against each enemy deals 100 % increased damage and ignores defense abilities',
          {"first_hit_per_enemy_multiplier": 2}),
    191: ('first time self would die: black box overloads; survive at 1 hp and deal 50 % self max hp damage to enemy',
          {"lethal_survivals": 1, "survival_damage_max_hp_fraction": 0.5}),
    197: ('on death, give 25 % stats to next ally and 1 turn shield',
          {"next_card_stat_add_fraction": 0.25, "next_card_dodges": 1}),
    202: ('whenever an enemy is defeated by self, permanently gain 30 % stats of defeated card and immediately extra attack',
          {"kill_steal_fraction": 0.3, "kill_extra_action": 1}),
    206: ('+ 50 % damage, - 30 % damage taken; fades below 1 / 2 hp; heal and gain damage by 20 % on kill',
          {"fade_threshold": 0.5, "fade_outgoing_multiplier": 1.5, "fade_incoming_multiplier": 0.7, "kill_heal_max_hp_fraction": 0.2, "kill_attack_multiplier": 1.2}),
    208: ("enemy's first 3 attacks drop its atk by 35 %",
          {"debuff_attacker_count": 3, "debuff_attacker_multiplier": 0.65}),
    217: ('+ 200 % damage, - 50 % damage taken; fades below 25 % hp',
          {"fade_threshold": 0.25, "fade_outgoing_multiplier": 3, "fade_incoming_multiplier": 0.5}),
    223: ('deals 100 % more damage to younger cards',
          {"younger_target_multiplier": 2}),
    227: ("does not attack; all damage taken at 1.5 x is added to next card's attack",
          {"never_attacks": 1, "stored_damage_to_next_attack_fraction": 1.5}),
    231: ('does not attack; all damage taken at 1.25 x is added to ally party attack',
          {"never_attacks": 1, "damage_taken_to_party_attack_fraction": 1.25}),
    235: ('when an enemy sees self, steal 20 % of its stats',
          {"enemy_entry_steal_fraction": 0.2, "entry_steal_fraction": 0.2}),
    258: ('every other turn, freeze self; self cannot take damage that turn',
          {"alternate_rest": 1}),
    259: ('on death, 2 x stats of next card and freeze it for 2 turns',
          {"next_card_stat_multiplier": 2, "next_card_freeze_turns": 2}),
    279: ('reduce enemy stats by 35 % of yours on entry',
          {"entry_enemy_stat_subtract_fraction": 0.35}),
    280: ("each turn steal 25 % of the enemy's hp before attack",
          {"turn_start_hp_steal_fraction": 0.25}),
    285: ('4 x stats on entry, fades below 1 / 2 hp',
          {"fade_threshold": 0.5, "fade_stat_multiplier": 4}),
    # Batch 6 (user): Academy Student's enemies lose 30% of damage dealt directly;
    # Seraphim's entry attack hits the whole team, waiting cards for 25%
    # of the damage dealt to the active card (fits the Inari test; provisional).
    88: ('each enemy loses hp equal to 30 % of damage dealt',  # User: nerfed; description edited.
         {"splash_damage_fraction": 0.3}),
    172: ('on entry, attack all enemies with 25 % of damage',
          {"entry_hit_multiplier": 0.25, "entry_hit_all_enemies": 1}),
    # Parallax (user): once, a lethal hit is negated and the attacking card dies, also when hit
    # in the deck by Seraphim's entry; #248's mutual destruction still kills it.
    65: ('once only, if self would die, the enemy dies instead',
         {"death_reflects": 1}),
    # Batch 7 (extrapolated from card text; provisional until user-tested).
    # Useless Seer: No battle effect (reveals the enemy hand).
    4: ("reveal enemy's hand on entry",
          {}),
    16: ('gain 2 turns if hp drops below 50 %',
          {"low_hp_extra_actions": 2, "low_hp_actions_threshold": 0.5}),
    26: ("inflict max hp as damage on death and boost allies' stats by 20 %",
          {"death_damage_max_hp_fraction": 1, "death_ally_stat_multiplier": 1.2}),
    27: ('50 % chance to deal max hp damage on death',
          {"death_damage_max_hp_fraction": 1, "death_damage_chance": 0.5}),
    80: ('extra attack for 50 % damage bypassing defenses',
          {"followup_multiplier": 0.5, "followup_bypass": 1}),
    # Tornado: 2x hit, then a 4x meteor follow-up (extrapolated).
    95: ('deal 2 x damage then 4 x with a meteor',
          {"outgoing_multiplier": 2, "followup_multiplier": 2}),
    # A0-ON1: Friendship bonus applied in compile_battle.
    109: ('gain 40 % stats for each unique friendship card',
          {}),
    # AK4-ON1: Friendship bonus applied in compile_battle.
    110: ('gain 40 % stats for each unique friendship card',
          {}),
    111: ('deal 2 x damage, gain extra turn, and + 20 % stats on kill',
          {"outgoing_multiplier": 2, "kill_extra_action": 1, "kill_attack_multiplier": 1.2, "kill_hp_multiplier": 1.2}),
    116: ('every 3 turns, enemies rests and you gain 100 % damage',
          {"periodic_attack_period": 3, "periodic_attack_multiplier": 2, "periodic_enemy_slow_turns": 1}),
    # Hollow: Void stacks read as every third turn (extrapolated).
    184: ('each turn gain 1 void stack; at 3 void, consume them to make the next attack deal 100 % increased damage and heal 25 % max hp',
          {"periodic_attack_period": 3, "periodic_attack_multiplier": 2, "periodic_attack_heal_max_hp_fraction": 0.25}),
    189: ('attacks restore hp equal to 25 % of damage dealt; every third attack performs waterfowl dance, striking 3 times for 50 % damage each',
          {"heal_damage_dealt_fraction": 0.25, "periodic_attack_period": 3, "periodic_attack_hits": 3, "periodic_attack_multiplier": 0.5}),
    192: ('while active, freeze each enemy on their first turn; deal 100 % increased damage to frozen enemies',
          {"encounter_freeze_turns": 1, "frozen_target_multiplier": 2}),
    194: ('30 % chance to critically strike, dealing 50 % increased damage; critical strikes immediately trigger a follow-up attack dealing 50 % damage',
          {"critical_probability": 0.3, "critical_multiplier": 1.5, "followup_multiplier": 0.5, "followup_on_critical": 1}),
    218: ('chance to kill based on damage to hp ratio; 50 % chance to self destruct on death, inflicting max hp as damage',
          {"ratio_kill": 1, "death_damage_max_hp_fraction": 1, "death_damage_chance": 0.5}),
    # Scarecrow: Bird cards read as the inferred avian class.
    269: ('inflict confusion to bird cards for 3 turns when you encounter one',
          {"encounter_confusion_turns": 3, "encounter_class_mask": 8}),
    # Poison (IMG_0297 and user Black Plague test).
    # Black Plague: User: wipes a team of Platinum Jamiys after 2 turns (ticks of 50% of its ATK).
    52: ('on entry, poison all enemies at 50 % effectiveness for 2 turns',
          {"entry_poison_turns": 2, "entry_poison_all": 1, "poison_fraction": 0.5}),
    # Dilophosaurus: IMG_0297: poison ticks for its full ATK; user (IMG_0303): it lasts the whole battle.
    146: ('on entry, attack and apply poison; dodge the first lethal attack and swap with the next ally',
          {"entry_hit_multiplier": 1, "entry_poison_turns": 2, "poison_permanent": 1, "own_lethal_dodges": 1, "lethal_dodge_swap": 1}),
    237: ('inflict poison for 2 turns when dealing/receiving damage',
          {"hit_poison_turns": 2}),  # IMG_0300: being hit by Ancient Egg did not poison the Egg.
    # Ability removal, theft and copying (user battles 5-9).
    # Hell's Army: IMG_0316: Inari showed 'No Ability' after its entry.
    33: ("cut target's hp in half and remove their ability on entry",
          {"entry_enemy_hp_multiplier": 0.5, "entry_disable_enemy": 1}),
    # Set: IMG_0316: Raze stayed disabled after Set's first damaging hit.
    75: ("remove enemy's ability after dealing damage",
          {"hit_disable": 1}),
    # Samurai: Every other card on both sides while in play (provisional).
    106: ('cancel all abilities while in play',
          {"cancels_all_abilities": 1}),
    137: ("enemy's cards cannot use abilities for 3 turns",
          {"entry_suppress_enemy_turns": 3}),
    # Hecate: IMG_0317: its text became Raze's 'Hatred'; the victim loses it (provisional).
    284: ('steals the ability of the first card it attacks',
          {"steal_first_attacked": 1}),
    # Mother of Beasts: IMG_0317: Inari 20,480 -> 17,409 -> 14,797 ATK.
    212: ('steal 15 % of enemy stats each turn; steal enemies ability on the first kill',
          {"turn_start_steal_fraction": 0.15, "kill_steal_ability": 1}),
    # Hades: IMG_0319: showed Raze's 'Hatred' after Raze fell; copies on entry (provisional).
    49: ("copy fallen ally's ability",
          {"copy_fallen_ally": 1}),
    247: ('gain the ability of a fallen ally',
          {"copy_fallen_ally": 1}),
    # Black Cat: IMG_0324: Judgement Day died to one hit (2x).
    69: ("enemy's rng abilities always fail; 2 x damage to rng-based cards",
          {"disable_enemy_class_mask": 16, "bonus_vs_class_mask": 16, "class_damage_multiplier": 2}),
    268: ('chance to steal 50 % atk on entry',  # User: 50% chance (IMG_0336: Pterodactylus 2,705 -> 1,353, Witch -> 4,058).
          {"entry_steal_attack_fraction": 0.5, "entry_steal_attack_chance": 0.5}),
    # Night Witch: Hex lasts the battle.
    276: ('hex target, making rng abilities fail',
          {"entry_disable_enemy_class_mask": 16}),
    # Sable The Envious: User: copies the enemy card in play.
    211: ('enemy abilities apply to you instead',
          {"copies_enemy_in_play": 1}),
    # Ra: IMG_0325: killed Arthur waiting behind Parallax.
    79: ('damages the next card, ignoring its on-death ability',
          {"entry_hit_multiplier": 1, "entry_hit_next_card": 1}),
    # Fossils (user, IMG_0314/0315): team counter; every dying dinosaur adds one (compile_fighter).
    # Ancient Egg: User: 3 fossils from its ability plus 1 for dying.
    141: ('on death, gain 3 fossils',
          {"death_fossils": 3}),
    # Meteosaurus: IMG_0314/0315: its 10x hit killed Raze and Heaven's Armor, then it died.
    151: ('attacks with 10 x damage and dies; on death, + 2 fossils',
          {"outgoing_multiplier": 10, "lifetime_actions": 1, "death_fossils": 2}),
    # Tyrannodon: IMG_0314: 8,678 x 1.5^7 = 148,272 with 7 fossils.
    159: ('on entry, for every fossil, 1.5 x damage',
          {"entry_fossil_attack_multiplier": 1.5}),
    157: ('takes 1 / 2 damage; deals 50 % more damage; if fossil count > 2, counterattacks',
          {"incoming_multiplier": 0.5, "outgoing_multiplier": 1.5, "counter_on_damage": 1, "counter_min_fossils": 3}),
    # Velociraptor: Death timer 5 - fossils (count capped at 2) turns, as the text reads.
    154: ('on damage inflict death in: 5 - fossil count (max 2) turns',
          {"fossil_death_timer": 5}),
    # Cyberdon: Charges a turn before each attack (provisional).
    155: ('charges for one turn before attack; deals 75 % damage; damages fossil count + 1 enemies in the deck (max of 3)',
          {"outgoing_multiplier": 0.75, "entry_self_slow_turns": 1, "action_end_self_slow_turns": 1, "fossil_deck_hits": 3}),
    # Batch 9 (user videos IMG_0299-0304 and user answers; provisional details noted).
    # Ghoul: IMG_0305: its poison kept ticking about 15% of Malik's max HP (far above Ghoul's 285 ATK)
    # for the rest of the fight; weakness does not stack.
    274: ('attacks inflict poison and weakness (enemy takes 30 % more damage)',
          {"hit_poison_turns": 2, "poison_permanent": 1, "poison_target_max_hp_fraction": 0.15, "hit_weakness_multiplier": 1.3}),
    # Zombie Nurse: Bleed 3 turns as other bleed cards; heals 50% max HP once.
    275: ('attacks inflict bleed; heals 50 % when below 1 / 2 hp only once',
          {"hit_bleed_turns": 3, "low_hp_heal_fraction": 0.5, "low_hp_trigger_threshold": 0.5}),
    # Zombie Dragon: Untouchable at 0 HP for 2 turns, then dies.
    178: ('survive for two turns once hp reaches zero; after attack, poison the target for 15 % of their max hp',
          {"zero_hp_survival_turns": 2, "hit_poison_turns": 2, "poison_target_max_hp_fraction": 0.15}),
    # Kira: User: kills in the same fashion Onmyoji damages (fires even after Kira dies).
    103: ('enemies you see die after 2 turns',
          {"encounter_doom_turns": 2}),
    # Onmyoji: User: counts the normal attacks of the ally in play, before each attack; hits the enemy in play.
    107: ('deal 300 % damage after 5 moves, even on death',
          {"delayed_strike_turns": 5, "delayed_strike_multiplier": 3}),
    # Rudolph: IMG_0299: Raze's attacks missed for 50 s; blind is permanent.
    253: ('blind target on entry',
          {"entry_blind": 1}),
    # Banshee: User: stuns once, for 1 turn; stun stops attacks, not abilities.
    267: ('stun attacker when hp drops below half',
          {"low_hp_stun_turns": 1, "low_hp_trigger_threshold": 0.5}),
    # Gambler: User: ATK -10%, +15% or +30% after attacking; IMG_0306: 22/13/15 of 50 draws (equal thirds kept).
    83: ('after attack, stats change by - 10 to + 30 %',
          {"stat_gamble": 1}),
    # Loch Ness: IMG_0303: 14,126 -> 20,682 over 4 allied turns (x1.1 each).
    196: ('while not the active card, gain 10 % damage and max hp after each allied turn; max of 300 % base',
          {"bench_growth_multiplier": 1.1, "bench_growth_cap": 3}),
    # Steven: User: Steve won the final battle (IMG_0304).
    183: ('after attack, place block in front of self; each block absorbs 40 % self max hp damage; max 3 blocks',
          {"block_max_hp_fraction": 0.4, "block_max": 3}),
    # Batch 12: the remaining cards (user answers 2026-09-30/10-01; IMG_0304/0310/0311/0327-0341). Provisional.
    # Mist Spirit: cards with any status cannot hit it, 5 dodges in all (user Part A).
    30: ('enemies with status effects cannot hit self; max of 5 dodges',
          {'status_dodge_max': 5}),
    # Pandora (user): two random abilities from every other card.
    37: ('on entry, gain 2 random abilities',
          {'random_abilities': 2}),
    # Loki: takes the enemy's ability and gives it a random one (user Part A).
    43: ("steal and randomize enemy's ability on entry",
          {'entry_ability_swap_random': 1}),
    # The Curse: a domain strips enemy defenses while it is in play (not Parallax); 2-5 slashes of 35%.
    48: ('on entry open a domain and disable all enemy defense abilities; slashes enemy 2 - 5 times dealing 35 % damage each',
          {'domain': 1, 'random_hits_min': 2, 'random_hits_max': 5, 'outgoing_multiplier': 0.35}),
    # Chaos (user): each of 4 enemies hit independently (50%) for 50% damage.
    53: ('randomly attacks 4 enemies dealing 50 % damage to each',
          {'chaos_targets': 4, 'chaos_hit_chance': 0.5, 'outgoing_multiplier': 0.5}),
    # Astraeus (user): seven arts, each a separate card; one with no art given draws an art on entry.
    56: ("gain an ability based on the card's art",
          {'form_first': 'card:56:scorpio', 'form_count': 7}),
    # Anubis (user): revives at half HP at the back when an ally dies.
    76: ('revive at 1 / 2 hp if an ally dies',
          {'revive_on_ally_death_hp': 0.5, 'revive_on_ally_death_chance': 1.0}),
    # Serket: while alive in the party, its side is immune to statuses.
    77: ('when alive and in party, all allies become immune to status effects',
          {'team_status_immunity': 1}),
    # Control Freak: defeated enemies join the back of its deck with their own stats and ability (provisional).
    90: ('defeated enemies fight on your side',
          {'recruit_defeated': 1}),
    # Ultimate Brawler (IMG_0327): first death awakens instead: 1.5x stats, dodges every other attack.
    98: ("enemy's hit chance is squared ratio of damage to self max hp; on first death awaken instead",
          {'hit_chance_squared': 1, 'awaken_on_death': 1, 'transform_stat_multiplier': 1.5, 'transform_form': 'form:mastered_ascension'}),
    # Cosmic Pop Star (IMG_0330, IMG_0336): 75% of the card behind's stats; that card is banished; counters.
    99: ('steal 75 % of the stats from the card behind self and banishes it; counterattacks',
          {'entry_absorb_behind': 0.75, 'counter_on_damage': 1}),
    # Cat Lady: once, evades a lethal hit and swaps with the next ally.
    101: ('dodge lethal and switch with an ally',
          {'own_lethal_dodges': 1, 'lethal_dodge_swap': 1}),
    # Tsukuyomi (user): the opponent's even turns are reflected onto it, with the effects attacks carry.
    117: ('every other turn, enemy damages themselves',
          {'reflect_even_turns': 1}),
    # Kuchisake-onna (user Part A): confusion until a self-hit, which costs 20% ATK.
    119: ('inflicts confusion on entry; if the enemy self-hits: - 20 % atk; else, confusion persists',
          {'persistent_confusion': 1}),
    # Demon Cultivator (IMG_0327): +30% base stats for 2 turns, then Dark Awakening.
    120: ('stats increase by 30 % for 2 turns; awaken after 2 turns',
          {'growth_add_base': 0.3, 'growth_turns': 2, 'awaken_turn': 3, 'transform_form': 'form:dark_awakening'}),
    # Heavenly Demon: 50% chance to pull a random waiting enemy forward and hit it for 3x.
    122: ("50 % chance to swap the enemy's current card and deal 3 x damage",
          {'swap_attack_chance': 0.5, 'swap_attack_multiplier': 3.0}),
    # Divine Doctor: every 3 allied turns, every living ally heals to full (provisional).
    125: ('heal by max hp after allies take 3 turns',
          {'doctor_period': 3}),
    # Immortal Cultivator (user): awakened, half damage taken and +50% stats per kill.
    126: ('gain 30 % stats for 2 turns and awaken on the third',
          {'growth_add_base': 0.3, 'growth_turns': 2, 'awaken_turn': 3, 'transform_form': 'form:immortal_ascension'}),
    # The Awakened One (IMG_0329, IMG_0342): one of six weapons; the Great Nirvana Sword - Zero gives 2 of the other 5.
    129: ('on entry, manifest 1 of 6 weapons, gaining its unique ability',
          {'form_first': 'form:twelve_devas_axe', 'form_count': 6, 'form_pair_last': 1}),
    # Nuwa (user, IMG_0336): on entry a random card (not Limited, not Glamour) with its stats goes to the back.
    130: ('create a random card with self stats and place it at the end of ally deck',
          {'summon_random': 1, 'pool_exclude_mask': 3}),
    # Bei Fang Xuan Wu (user): takes 50% of the damage its allies take, even from the deck.
    133: ('redirect 50 % of damage taken by allies to self',
          {'ally_redirect_fraction': 0.5}),
    # Sun Wukong (IMG_0327): x2 stats below 50% HP.
    138: ('transform when hp falls below 50 %',
          {'low_hp_transform_threshold': 0.5, 'transform_stat_multiplier': 2.0}),
    # Buddha (IMG_0303/0304/0311): revives or heals instead of attacking; to the back when all are healed.
    139: ('revive a dead ally or heal a living one by 50 % instead of attacking; move to end of deck if fully healed',
          {'buddha': 1}),
    # Brachiosaurus: its attack hits a random living waiting card.
    143: ('attack a random living card in the enemy deck',
          {'random_deck_target': 1}),
    # Tricerotops (user): entry attack; overkill carries to the next card.
    144: ('attack on entry; excess damage from kill carries over to the next card',
          {'entry_hit_multiplier': 1, 'overkill_carry': 1}),
    # Marrowclaw: 50% to negate the enemy's blue support; ability damage cannot kill it.
    152: ("on entry, 50 % chance to negate the enemy's blue aura card; cannot be killed by abilities",
          {'aura_negate_chance': 0.5, 'ability_death_immune': 1}),
    # Poison Witch: resets the enemy's stats on entry; takes 20% of the enemy's ATK as healing after each enemy turn.
    175: ("clear enemy stat changes on entry; enemy turn end; heal by and subtract 20 % target's atk",
          {'entry_reset_enemy_stats': 1, 'enemy_turn_end_attack_drain': 0.2}),
    # Juggernoid: reflects 25% of ability damage (at most 8x its ATK).
    177: ("reflects 25 % of ability damage taken (up to 800 % of self's damage)",
          {'ability_reflect_fraction': 0.25, 'ability_reflect_cap': 8.0}),
    # The Sack (user): three independent rolls on entry.
    186: ('on entry, roll the d6 to randomly gain 1 - 50 % damage, 1 - 50 % max hp, and 1 - 30 % damage reduction for battle',
          {'sack': 1}),
    # Fresno Nightcrawler (user): two copies with 35% stats, which do not summon.
    195: ("on entry, summon two copies of self to the back of ally deck; each copy has 35 % of self's stats",
          {'summon_copies': 2, 'summon_copy_fraction': 0.35}),
    # Jersey Devil (IMG_0341): shuffles the enemy lineup on entry.
    199: ('on entry, shuffle the lineup',
          {'entry_shuffle_enemy': 1}),
    # The Rake: from the deck, adds a 25% hit when its active ally attacks.
    203: ('field: attack alongside active card for 25 % damage',
          {'rake_fraction': 0.25}),
    # Kraken: entry hits on the whole enemy team, 1-8 times at 10%.
    204: ('on entry, attack all enemies 1 - 8 times, dealing 10 % damage per hit',
          {'entry_hit_multiplier': 0.1, 'entry_hit_all_enemies': 1, 'entry_hit_count_max': 8}),
    # Naga (user): heals and transforms below 65% HP; that form curses its killer side for 4 turns on death.
    214: ('transform and heal when hp drops below 65 %',
          {'low_hp_transform_threshold': 0.65, 'transform_heal': 1, 'transform_form': 'form:naga'}),
    # Uncle Sam (user Part A): the next card deals 5x and dies after attacking.
    215: ("buff next card's damage by 400 %, but it dies after attack",
          {'next_card_boost_multiplier': 5.0}),
    # Kid Gohan: Piccolo gains 1.5x instead of 2x when swapping in for it.
    224: ('when piccolo uses its ability to swap with self, piccolo gains 1.5 x stats',
          {'piccolo_factor_override': 1.5}),
    # Milk (user): gives its stats to Dad and dies; without Dad it fights normally.
    225: ('dies and transfer stats to dad cards',
          {'dad_card_id': 223}),
    # Eclipseborn Luminant: allies 40% evasion, -10% per evade (20% floor), 2 per card; gains 10% prevented damage.
    228: ('grants allies 40 % evasion, decreasing by 10 % per successful dodge to 20 %; max 2 per card; gain 10 % of prevented damage; 200 % cap',
          {'luminant_evasion': 0.4}),
    # The Broken One (user): Joy then Sorrow, no abilities, from its stats when it died.
    230: ('on death, split into joy and sorrow; each inherits 50 % of max hp and 60 % of damage, entering battle one after another',
          {'split_on_death': 1}),
    # Anubis & Hades (IMG_0311): revives an ally (itself first) at the back with 75% of its stats per fallen enemy.
    233: ('revive an ally with 75 % of self stats for each fallen enemy',
          {'enemy_death_revive_fraction': 0.75}),
    # Longmu: Mother of Dragons unless first in the deck (compile_battle switches it to Safeguarding).
    234: ('if first in deck, ability is safeguarding; otherwise, mother of dragons',
          {'team_class_shields': 2, 'ally_class_mask': 4}),
    # Eonus: 5% of every other card's stats; the stolen stats decay after 3 turns.
    236: ("on entry, steals 5 % of all other cards' stats, including allies; stolen stats decay after 3 turns",
          {'entry_global_steal': 0.05}),
    # Cthulu: every enemy it meets is confused for good; +50% stats per enemy self-hit.
    239: ('enemies you see are inflicted with confusion eternally; when a confused enemy hits themselves, + 50 % stats',
          {'encounter_confusion_eternal': 1, 'confused_self_hit_stat_multiplier': 1.5}),
    # Fate Seamstress: links the first 2 enemies; damage to one is dealt to the other.
    240: ('on entry, permanently link the first 2 enemies; damage dealt to one is shared with the other',
          {'link_first_two': 1}),
    # Time Lord Stryx (IMG_0310): on death the party returns in its initial order.
    250: ('on death, revive ally party',
          {'revive_party_on_death': 1}),
    # Gingerbread Man: dodges every other attack; one more attack next turn per dodge.
    254: ('dodge every other attack, and when you do, gain 1 more attack',
          {'alternate_dodge': 1, 'dodge_extra_action': 1}),
    # Santa Claws: living allies +50% stats, then -10% per turn in play.
    265: ("on entry, every living ally gains 50 % stats, but loses 10 % stats each turn they're in play",
          {'entry_ally_stat_multiplier': 1.5}),
    # Bloody Mary (IMG_0310): 50% chance to return at the back when an ally dies.
    270: ('50 % chance to return when an ally dies',
          {'revive_on_ally_death_hp': 1.0, 'revive_on_ally_death_chance': 0.5}),
    # Dr. Frankenstein: each kill adds a Frankenstein with its stats to the back.
    272: ('add a frankenstein with self stats to ally deck on kill',
          {'kill_summon_form': 'card:27'}),
    # Marionette (IMG_0336): after attacking, swaps 2 random enemies, -15% stats each.
    273: ("swap 2 random enemies' stats/positions, reducing stats by 15 % per swap",
          {'swap_enemies': 1, 'swap_stat_multiplier': 0.85}),
    # The Hanged Man (user): its killer loses 25% max HP at the start of each turn.
    277: ('enemy that kills self loses 25 % hp each turn',
          {'curse_killer': 1}),
    # Walking Dead (user): on borrowed time after dying; each kill buys another turn.
    278: ('survive 1 turn when dead; while dead: gain a turn of lifespan kill',
          {'zero_hp_survival_turns': 2, 'kill_extends_undying': 1}),
    # The Composer: from anywhere, 10% (+10% per turn) to confuse the enemy at each allied turn start.
    281: ("field: at the start of any ally's turn 10 % chance to confuse target; goes up 10 % each turn",
          {'composer_chance_step': 0.1}),
    # Flying Dutchman (user): swaps with a random ally after attacking (at most twice); the ally has one turn.
    283: ('after attack, switch places with a random ally',
          {'dutchman_swaps': 2}),
    # Sleep Paralysis (user, IMG_0341): stuns the enemy for 1 turn; after 3 turns both active cards die.
    286: ('on entry, the enemy is stunned; in 3 turns, both active cards die',
          {'perish_turns': 3, 'entry_slow_turns': 1}),
    # Glamour (user, IMG_0330): becomes a random card at the start of each turn.
    289: ('each turn, becomes a different card',
          {'turn_start_random_ability': 1}),
}

FRIENDSHIP_CARDS = frozenset(card for card in (109, 110))
FRIENDSHIP_STAT_BONUS = 0.4

# Inferred art-based classes (unverified) as Fighter.class_mask bits.
CLASS_BITS = {"undead": 1, "demon": 2, "dragon": 4, "avian": 8, "rng": 16, "toy": 32, "friendship": 64}


def _pack_indices():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).resolve().parents[2] / "data" / "clean" / "dataset.json").read_text())
    return {card["card_id"]: card["pack_id"] for card in data["cards"]}


_PACK_INDEX = _pack_indices()

# Forms and weapons (docs/card_forms.md): pool-only abilities, never drawn at random.
FORMS = {
    # The Awakened One's Six Realms weapons (IMG_0329, IMG_0342); the Great Nirvana Sword - Zero takes 2 of the others.
    "twelve_devas_axe": {"outgoing_multiplier": 2.5},  # Deals 150% increased damage.
    "shield_of_ahimsa": {"incoming_multiplier": 0.65, "shield_period": 2},  # A shield every other turn; 35% less damage.
    # 40% chance to parry an attack, taking no damage and reflecting 75% of its damage, capped at 75% of current HP.
    "vajra_short_sword": {"parry_chance": 0.4, "parry_reflect_fraction": 0.75},
    # On entry, attack the first 2 enemies, bypassing defensive and on-death abilities.
    "war_scythe": {"entry_hit_multiplier": 1, "entry_bypass_defenses": 1, "entry_hit_also_next": 1},
    # Deals 25% increased damage with a 25% chance to stun. Ignores defensive abilities.
    "staff_of_perfect_enlightenment": {"outgoing_multiplier": 1.25, "hit_stun_chance": 0.25, "bypass_defenses": 1},
    "great_nirvana_sword": {},
    # Awakenings and transformations (IMG_0327, user).
    "mastered_ascension": {"alternate_dodge": 1},
    "dark_awakening": {"outgoing_multiplier": 2.0, "heal_damage_dealt_fraction": 0.3, "kill_stat_add_base": 0.5},
    "immortal_ascension": {"incoming_multiplier": 0.5, "kill_stat_add_base": 0.5},
    "naga": {"death_curse_turns": 4},
}
# Astraeus (user): each art is a separate card under the name Astraeus (IMG_0333/0334; Taurus and Cancer, the crab,
# described by the user). Teams pick an art with compile_battle(arts=...); an unspecified one draws an art on entry.
ASTRAEUS_ARTS = {
    "scorpio": {"hit_poison_turns": 1, "poison_permanent": 1, "poison_fraction": 1.0},
    # User: heals 30% at or below 50% HP, otherwise +25% max HP.
    "aquarius": {"turn_start_max_hp_multiplier": 1.25, "turn_start_low_hp_heal_fraction": 0.3, "turn_start_low_hp_threshold": 0.5},
    "virgo": {"entry_shields": 2},
    "gemini": {"entry_same_card_stat_bonus": 0.5},
    "sagittarius": {"evade_punish_multiplier": 2.0},
    "taurus": {"lost_hp_damage_bonus": 1.5, "lost_hp_incoming_bonus": 1.5},
    # User: "nullify damage below 100% of Max HP, reducing the threshold by 15% each time" (each nullification).
    "cancer": {"nullify_max_hp_threshold": 1.0, "nullify_threshold_step": 0.15},
}
ASTRAEUS = 56
LONGMU_SAFEGUARDING = {"team_class_shields": 0, "block_mode": 1, "ally_class_stat_multiplier": 1.2, "ally_class_mask": 4}
SPARES_PER_KILLER = 3  # Dr. Frankenstein: spare slots for its summons


def _pool_keys():
    """Pool order: every card (Astraeus as one entry per art, user: separate cards), then the forms."""
    keys = []
    for card_id in sorted(SUPPORTED):
        keys += [f"card:{card_id}:{art}" for art in ASTRAEUS_ARTS] if card_id == ASTRAEUS else [f"card:{card_id}"]
    return keys + [f"form:{name}" for name in FORMS]


def _pool_index():
    return {key: i for i, key in enumerate(_pool_keys())}


_POOL_INDEX = _pool_index()


def _resolve(parameters):
    """Pool references ("form:name", "card:id") become pool index + 1."""
    return {key: _POOL_INDEX[value] + 1 if isinstance(value, str) else value for key, value in parameters.items()}


_POOLS = {}


def ability_pool(catalog):
    """Every supported card's ability, then the forms (codes POOL_CODE + index)."""
    cached = _POOLS.get(id(catalog))
    if cached is not None and cached[0] is catalog:
        return cached[1]
    entries = []
    for key in _pool_keys():
        kind, *rest = key.split(":")
        if kind == "form":
            entries.append(Fighter(1.0, 0.0, pool_flags=POOL_FORM, **_resolve(FORMS[rest[0]])))
            continue
        card_id = int(rest[0])
        card = catalog.card(card_id)
        flags = (POOL_LIMITED if "Limited" in card.packs else 0) | (POOL_GLAMOUR if card_id == 289 else 0)
        entries.append(replace(compile_fighter(catalog, card_id, 1, art=rest[1] if len(rest) > 1 else None), pool_flags=flags))
    pool = tuple(entries)
    _POOLS[id(catalog)] = (catalog, pool)
    return pool


# Support cards (IMG_0350-0354: the user's binder at base, Platinum, Crystal, Ruby and Galaxy). A support has exactly one
# border quality (user): tier 1 base (I) .. 5 Galaxy (IIIII). None = not read in the videos (refused, not guessed).
SUPPORT_TIERS = ("base", "Platinum", "Crystal", "Ruby", "Galaxy")
_R = None
# Red: (target kind, target, [(all %, matching %) per tier]); matching cards take the larger figure instead.
RED_SUPPORTS = {
    1: ("all", None, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]),
    2: ("hp", None, [(4, 4), (8, 8), (16, 16), (25, 25), (32, 32)]),
    3: ("pack", "Anime", [(6, 12), (12, 24), (25, 50), (42, 84), (51, 102)]),
    4: ("weather", "Storm", [(8, 16), (16, 32), (32, 64), (51, 102), (64, 128)]),
    5: ("weather", "Snow", [(8, 16), (16, 32), (32, 64), (51, 102), (64, 128)]),  # Galaxy as Stormcaller (equal at I-IIII)
    6: ("pack", "Rising Sun", [(8, 16), (17, 34), (34, 68), (55, 110), (68, 136)]),
    7: ("class", "demon", [(9, 16), (18, 32), (37, 66), (61, 109), (75, 135)]),
    8: ("weather", "Aurora", [(10, 20), (21, 42), (42, 84), (68, 136), (84, 168)]),
    9: ("pack", "Egypt", [(11, 22), (22, 44), (44, 88), (73, 146), (89, 178)]),
    10: ("weather", "Meteor Shower", [(12, 24), (24, 48), (48, 96), (78, 156), (97, 194)]),
    11: ("weather", "Time Storm", [(12, 24), (25, 50), (51, 102), (84, 168), (103, 206)]),
    12: ("weather", "Blood Rain", [(14, 28), (28, 56), (56, 112), (91, 182), (113, 225)]),
    13: ("class", "dragon", [(14, 28), (29, 58), (58, 116), (95, 190), (117, 234)]),
    14: ("class", "swordsmen", [(14, 28), (29, 58), (58, 116), (95, 190), (117, 234)]),  # developer's Swordsmen list
    15: ("class", "avian", [(14, 28), (29, 58), (58, 116), (95, 190), (117, 234)]),
    16: ("pack", "Immortal", [(15, 30), (30, 60), (61, 122), (99, 198), (125, 246)]),
    17: ("weather", "Shroud", [(16, 32), (32, 64), (64, 128), (103, 206), (128, 256)]),
    18: ("weather", "Eclipse", [(18, 36), (36, 72), (72, 144), (117, 234), (144, 288)]),
    19: ("pack", "Era 2", [(22, 36), (44, 72), (89, 146), (144, 237), (178, 293)]),
    20: ("pack", "Cryptid", [(24, 37), (48, 74), (97, 150), (157, 243), (194, 300)]),
    21: ("pack", "Video Game", [(24, 37), (48, 74), (97, 150), (157, 243), (194, 300)]),
    22: ("weather", "Rapture", [(25, 36), (51, 73), (103, 149), (168, 243), (207, 300)]),  # IIII-IIIII as Mangeka
    23: ("weather", "Manga", [(25, 36), (51, 73), (103, 149), (168, 243), (207, 300)]),
    24: ("weather", "Virus", [(26, 36), (53, 74), (107, 150), (174, 243), (214, 300)]),
    25: ("weather", "Armageddon", [(28, 36), (57, 74), (114, 149), (186, 244), (229, 300)]),
    26: ("pack", "Prehistoric", [(32, 51), (64, 103), (128, 207), (207, 334), (256, 414)]),  # IIII as Desmond
    27: ("class", "sin", [(32, 51), (64, 103), (128, 207), (207, 334), (256, 414)]),
    28: ("one_ring", None, [(39, 39), (78, 78), (157, 157), (157, 157), (315, 315)]),
}
# Blue: values per tier. Ruby (IIII) blue supports really use their base values (IMG_0353, user-confirmed).
BLUE_SUPPORTS = {
    1: [2, 5, 7, 2, 10],  # reduce damage taken by %
    2: [15, 25, 35, 15, 50],  # % chance to burn for 2 turns after attacking
    3: [5, 10, 15, 5, 20],  # % chance to counterattack
    4: [5, 10, 15, 5, 20],  # % chance to stun the target
    5: [(25, 5), (50, 10), (75, 15), (25, 5), (100, 20)],  # enhance life drain by %, grant % life drain to other allies
    6: [(8, 12), (10, 15), (12, 18), (8, 12), (15, 22)],  # avoid damage below % of HP (Time Storm cards)
    7: [15, 25, 35, 15, 50],  # % more damage to enemies below 30% HP
    8: [25, 50, 75, 25, 100],  # % chance to retry a failed ability
    9: [10, 15, 20, 10, 30],  # reflect % of damage taken
    10: [5, 7, 10, 5, 12],  # the next ally inherits % of a dying ally's stats
    11: [10, 15, 20, 10, 30],  # % chance to attack again at 50% damage after attacking
    12: [4, 3, 2, 4, 1],  # Magical Elf: awaken the Toys if the deck holds this many unique ones (one fewer per tier; Ruby 4)
    13: [10, 15, 20, 10, 25],  # prehistoric cards +% stats per prehistoric card in the deck
    14: [10, 15, 20, 10, 25],  # End Times: % chance for an opponent's ability to fail (a transient roll, user)
    15: [10, 15, 20, 10, 30],  # % chance to survive a lethal hit at 1 HP, once per ally
}
CHANCE_FIELDS = ("dodge_probability", "critical_probability", "hit_freeze_chance", "skip_enemy_turn_chance",
                 "chance_survival_probability", "gamble_chance", "entry_borderless_reset_chance", "turn_start_attack_chance",
                 "turn_start_full_heal_chance", "next_card_gift_chance", "death_damage_chance", "entry_steal_attack_chance",
                 "chaos_hit_chance", "swap_attack_chance", "revive_on_ally_death_chance", "aura_negate_chance", "hit_stun_chance",
                 "parry_chance")


def _support(entry):
    """A support given as an ID (base tier) or (ID, tier 1..5); 0 = none."""
    if isinstance(entry, int):
        entry = (entry, 1)
    if not isinstance(entry, tuple) or len(entry) != 2:
        raise ValueError("Supports are an ID or (ID, tier 1..5); zero means absent")
    support_id, tier = entry
    if type(support_id) is not int or support_id < 0 or type(tier) is not int or not 1 <= tier <= 5:
        raise ValueError("Supports are an ID or (ID, tier 1..5); zero means absent")
    return support_id, tier


def _validate_support(catalog, color, entry):
    support_id, tier = _support(entry)
    if support_id == 0:
        return
    catalog.support(color, support_id)
    table = RED_SUPPORTS if color == "red" else BLUE_SUPPORTS
    row = table.get(support_id)
    values = row[2] if color == "red" and row else row
    if row is None or values[tier - 1] is None:
        raise UnsupportedCardError(f"{color} support {support_id} at {SUPPORT_TIERS[tier - 1]}: value or effect not observed yet")


def _red_bonus(catalog, card, support_id, tier, weather_names):
    kind, target, values = RED_SUPPORTS[support_id]
    general, special = values[tier - 1]
    if kind in ("all", "hp"):
        return general
    if kind == "pack":
        match = target in card.packs
    elif kind == "weather":
        match = weather_names.get(card.weather_id) == target
    elif kind == "class":
        match = target in card.classes
    else:  # The One Ring: non-weather cards and Boss cards.
        return general if card.weather_id == 1 or "Bosses" in card.packs else 0
    return special if match else general


# Magical Elf's awakened Toys (IMG_0355), given the number of unique Toys in the deck.
TOYS = (261, 262, 263, 264)
AWAKENED_TOYS = {
    # Fluffy Aggression (user): 1,943 ATK -> 3,886 / 5,829 / 7,772 with 1-3 fallen Toys; 3 fallen = Platinum stats (15,544 HP).
    261: lambda unique: {"entry_fallen_toy_stat_bonus": 1.0},
    262: lambda unique: {"attacks_per_action": max(1, unique - 1)},  # Speedy Progression: an attack per other Toy
    263: lambda unique: {"enemy_entry_confusion_turns": unique},  # Pop-Up Impression
    264: lambda unique: {"damage_cap_max_hp": .25, "damaged_toy_stat_gain": .1},  # Shelter Obsession
}
_FIGHTER_DEFAULTS = {f.name: f.default for f in fields(Fighter)}


def _awaken(card, fighter, unique):
    cleared = {name: _FIGHTER_DEFAULTS[name] for name in SUPPORTED[card.id][1]}
    return replace(fighter, **{**cleared, **AWAKENED_TOYS[card.id](unique)}, awakened_toy=1)


def _apply_supports(catalog, team_ids, fighters, red, blue):
    """The side's supports, applied to every card of the team (provisional readings of the card texts)."""
    red_id, red_tier = _support(red)
    blue_id, blue_tier = _support(blue)
    weather_names = {w.id: w.name for w in catalog.weathers}
    out = []
    prehistoric = sum(1 for c in team_ids if "Prehistoric" in catalog.card(c).packs)
    unique_toys = len(set(team_ids) & set(TOYS))
    for card_id, fighter in zip(team_ids, fighters):
        card = catalog.card(card_id)
        if red_id:
            bonus = _red_bonus(catalog, card, red_id, red_tier, weather_names) / 100
            if RED_SUPPORTS[red_id][0] == "hp":
                fighter = replace(fighter, hp=fighter.hp * (1 + bonus))
            elif bonus:
                fighter = replace(fighter, hp=fighter.hp * (1 + bonus), attack=fighter.attack * (1 + bonus))
        if blue_id:
            v = BLUE_SUPPORTS[blue_id][blue_tier - 1]
            if blue_id == 8:  # Fate: a failed chance ability is retried with v% chance; the retry is another roll at p.
                fighter = replace(fighter, **{name: getattr(fighter, name) + (1 - getattr(fighter, name)) * v / 100 * getattr(fighter, name)
                                              for name in CHANCE_FIELDS if 0 < getattr(fighter, name) < 1})
            elif blue_id == 1:
                fighter = replace(fighter, aura_incoming_multiplier=1 - v / 100)  # an aura: Marrowclaw can negate it
            elif blue_id == 2:
                fighter = replace(fighter, hit_burn_chance=v / 100)
            elif blue_id == 3:
                fighter = replace(fighter, counter_chance=v / 100)
            elif blue_id == 4:
                fighter = replace(fighter, hit_stun_chance=1 - (1 - fighter.hit_stun_chance) * (1 - v / 100))
            elif blue_id == 5:
                enhance, grant = v
                # User: drains on direct attacks only; a card without its own drain (Black Plague) gets the granted %.
                if fighter.heal_damage_dealt_fraction or fighter.lost_hp_lifesteal or fighter.hit_hp_gain_fraction:
                    fighter = replace(fighter, heal_damage_dealt_fraction=fighter.heal_damage_dealt_fraction * (1 + enhance / 100),
                                      lost_hp_lifesteal=fighter.lost_hp_lifesteal * (1 + enhance / 100),
                                      hit_hp_gain_fraction=fighter.hit_hp_gain_fraction * (1 + enhance / 100))
                else:
                    fighter = replace(fighter, heal_damage_dealt_fraction=grant / 100)
            elif blue_id == 6:
                general, time_storm = v
                threshold = (time_storm if weather_names.get(card.weather_id) == "Time Storm" else general) / 100
                fighter = replace(fighter, dodge_below_max_hp=max(fighter.dodge_below_max_hp, threshold))
            elif blue_id == 7 and not fighter.weak_target_threshold:
                fighter = replace(fighter, weak_target_multiplier=1 + v / 100, weak_target_threshold=0.3)
            elif blue_id == 9:
                fighter = replace(fighter, reflect_damage_fraction=fighter.reflect_damage_fraction + v / 100)
            elif blue_id == 10:
                fighter = replace(fighter, next_card_stat_add_fraction=fighter.next_card_stat_add_fraction + v / 100)
            elif blue_id == 11 and not fighter.followup_multiplier:
                fighter = replace(fighter, followup_multiplier=0.5, followup_chance=v / 100)
            elif blue_id == 13 and "Prehistoric" in card.packs:
                factor = 1 + v / 100 * prehistoric
                fighter = replace(fighter, hp=fighter.hp * factor, attack=fighter.attack * factor)
            elif blue_id == 15:
                fighter = replace(fighter, guardian_chance=v / 100)
            elif blue_id == 12 and card_id in TOYS and unique_toys >= v:
                fighter = _awaken(card, fighter, unique_toys)
            elif blue_id == 14:
                fighter = replace(fighter, ability_fail_chance=v / 100)
        out.append(fighter)
    return out


def compile_fighter(catalog, card_id, border_id=1, *, mutation="None", art=None):
    card = catalog.card(card_id)
    if card_id not in SUPPORTED:
        raise UnsupportedCardError(f"{card_id} {card.name}: complete ability is not implemented; no battle label generated")
    expected, parameters = SUPPORTED[card_id]
    if card.description != expected:
        raise UnsupportedCardError(f"{card_id} {card.name}: description changed; review primitive mapping")
    if art is not None:
        if card_id != ASTRAEUS or art not in ASTRAEUS_ARTS:
            raise ValueError(f"Unknown art {art!r} for card {card_id}")
        parameters = ASTRAEUS_ARTS[art]
    pool_key = f"card:{card_id}:{art}" if art is not None else f"card:{card_id}" if card_id != ASTRAEUS else f"card:{card_id}:scorpio"
    stats = base_stats(catalog, card_id, border_id, mutation=mutation)
    borderless = base_stats(catalog, card_id, 1, mutation=mutation)
    classes = sum(bit for name, bit in CLASS_BITS.items() if name in card.classes)
    return Fighter(hp=stats.hp, attack=stats.attack, borderless_hp=borderless.hp, borderless_attack=borderless.attack,
                   card_rarity=float(card.rarity), border_rarity=float(catalog.border(border_id).rarity),
                   pack_index=_PACK_INDEX.get(card.id, 0), class_mask=classes, card_id=card.id,
                   pool_self=_POOL_INDEX[pool_key] + 1,
                   dinosaur=int("Prehistoric" in card.packs), **_resolve(parameters))  # User: every dying dinosaur adds a fossil.


def compile_battle(catalog, teams, *, borders=None, mutations=None, red_supports=(1, 1), blue_supports=(0, 0), first_side=0,
                   arts=None):
    """arts: optional {(side, position): Astraeus art name}; an Astraeus without one draws an art on entry."""
    arts = arts or {}
    for color, ids in (("red", red_supports), ("blue", blue_supports)):
        if len(ids) != 2:
            raise ValueError("Each side requires an explicit support ID (zero for absent)")
        for identifier in ids:
            _validate_support(catalog, color, identifier)
    if len(teams) != 2:
        raise ValueError("Expected two teams")
    if borders is None:
        borders = [[1] * len(team) for team in teams]
    if len(borders) != 2 or any(len(team) != len(ids) for team, ids in zip(teams, borders, strict=True)):
        raise ValueError("Each card requires exactly one border ID")
    if mutations is None:
        mutations = [["None"] * len(team) for team in teams]
    if (not isinstance(mutations, (list, tuple)) or len(mutations) != 2
            or any(not isinstance(names, (list, tuple)) or len(team) != len(names)
                   or any(not isinstance(name, str) for name in names)
                   for team, names in zip(teams, mutations, strict=True))):
        raise ValueError("Each card requires exactly one mutation name")
    compiled = []
    for side, (team, ids, names) in enumerate(zip(teams, borders, mutations, strict=True)):
        fighters = []
        for position, (c, b, m) in enumerate(zip(team, ids, names, strict=True)):
            fighter = compile_fighter(catalog, c, b, mutation=m, art=arts.get((side, position)))
            if c in FRIENDSHIP_CARDS:
                # A0-ON1 / AK4-ON1: +40% stats per other unique friendship card on the team (user: a lone one gets nothing).
                bonus = 1 + FRIENDSHIP_STAT_BONUS * len({card for card in team if card in FRIENDSHIP_CARDS and card != c})
                fighter = replace(fighter, entry_hp_multiplier=fighter.entry_hp_multiplier * bonus,
                                  entry_attack_multiplier=fighter.entry_attack_multiplier * bonus)
            if c == 234 and not fighters:
                fighter = replace(fighter, **LONGMU_SAFEGUARDING)  # Longmu first in the deck: Safeguarding.
            fighters.append(fighter)
        compiled.append(tuple(_apply_supports(catalog, team, fighters, red_supports[side], blue_supports[side])))
    lineup = tuple(len(team) for team in compiled)
    spares = [_spares(compiled[side], compiled[1 - side]) for side in (0, 1)]
    teams = tuple(team + tuple(extra[:MAX_FIGHTERS - len(team)]) for team, extra in zip(compiled, spares))
    battle = Battle(teams, first_side, lineup if any(spares) else (), ability_pool(catalog))
    battle.validate()
    return battle


def _blank(fighter):
    from .reference import IDENTITY_FIELDS
    return Fighter(**{name: getattr(fighter, name) for name in IDENTITY_FIELDS if name != 'spare'}, spare=1)


RANDOM_ABILITY_FIELDS = ("summon_random", "random_abilities", "turn_start_random_ability", "entry_ability_swap_random")


def _spares(team, enemies):
    """Spare slots (blank cards) for the summons a team can make. A side that can draw random abilities (its own Nuwa,
    Pandora or Glamour, or an enemy Loki) may draw a summoner too (IMG_0330: Glamour's The Broken One split into Joy
    and Sorrow), so it gets every free slot."""
    if any(getattr(f, name) for f in team for name in RANDOM_ABILITY_FIELDS[:3]) or any(f.entry_ability_swap_random for f in enemies):
        return [_blank(team[0])] * (MAX_FIGHTERS - len(team))
    extra = []
    for fighter in team:
        count = (fighter.summon_copies + fighter.summon_random + 2 * fighter.split_on_death
                 + (SPARES_PER_KILLER if fighter.kill_summon_form else 0))
        extra += [_blank(fighter)] * count
        if fighter.recruit_defeated:
            extra += [_blank(enemy) for enemy in enemies]  # Control Freak: one per enemy card.
    return extra


def coverage(catalog):
    rows = []
    for card in catalog.cards:
        try:
            compile_fighter(catalog, card.id)
            status, reason = "experimental_complete_ability_subset", "Shared event/rounding rules still require game validation"
        except UnsupportedCardError as exc:
            status, reason = "unsupported", str(exc)
        rows.append({"card_id": card.id, "name": card.name, "status": status,
                     "training_labels_allowed": False, "reason": reason})
    return rows


def support_coverage(catalog):
    rows = []
    for support in catalog.red_supports + catalog.blue_supports:
        try:
            _validate_support(catalog, support.color, support.id)
            status, reason = "experimental_complete_passive_subset", "Passive composition and shared battle rules remain provisional"
        except UnsupportedCardError as exc:
            status, reason = "unsupported", str(exc)
        rows.append({"color": support.color, "support_id": support.id, "name": support.name,
                     "status": status, "reason": reason, "training_labels_allowed": False})
    return rows
