import math
import unittest
from unittest import mock

from card_engine.simulator.catalog_rules import SUPPORTED

from card_engine.catalog import load_catalog
from card_engine.simulator import Battle, Fighter, Options, simulate
from card_engine.simulator.catalog_rules import BLUE_SUPPORTS, compile_battle, compile_fighter, coverage, UnsupportedCardError


def duel(a, b, first_side=0):
    return Battle(((a,), (b,)), first_side)


class ReferenceTests(unittest.TestCase):
    def test_basic_damage_and_initiative(self):
        fighter = Fighter(10, 5)
        self.assertEqual(simulate(duel(fighter, fighter)).p_a, 1)
        self.assertEqual(simulate(duel(fighter, fighter, 1)).p_b, 1)

    def test_entry_order_matches_good_boy_observation(self):
        fighter = Fighter(19, 10, entry_enemy_attack_multiplier=.85)
        result = simulate(duel(fighter, fighter), Options(mode="sample", rounding="ceil"), trace=True)
        self.assertEqual([e["phase"] for e in result.trace[:3]], ["ENTRY", "ENTRY", "ATTACK_DECLARE"])
        self.assertEqual([e["side"] for e in result.trace[:3]], [0, 1, 0])
        self.assertEqual(result.p_a, 1)
        self.assertFalse(result.training_labels_allowed)

    def test_exact_bernoulli_win_probability(self):
        result = simulate(duel(Fighter(1, 1), Fighter(1, 1, dodge_probability=.5)))
        self.assertEqual((result.p_a, result.p_b, result.tie, result.unresolved), (.5, .5, 0, 0))

    def test_poseidon_boost_precedes_chronus_enemy_entry_theft(self):
        catalog = load_catalog()
        battle = compile_battle(catalog, ((205,), (44,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode="sample"), trace=True)
        theft = next(i for i, e in enumerate(result.trace) if e["phase"] == "ENEMY_ENTRY")
        poseidon_entry = next(i for i, e in enumerate(result.trace) if e["phase"] == "ENTRY" and e["side"] == 1)
        self.assertGreater(theft, poseidon_entry)
        self.assertEqual(result.p_b, 1)

    def test_chronus_own_entry_does_not_steal(self):
        catalog = load_catalog()
        battle = compile_battle(catalog, ((3, 205), (44,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode="sample"), trace=True)
        self.assertTrue(any(e["phase"] == "ENTRY" and e["side"] == 0 and e["slot"] == 1 for e in result.trace))
        self.assertFalse(any(e["phase"] == "ENEMY_ENTRY" for e in result.trace))
        # User observed final ATK531 after Good Boy debuff precedes Poseidon boost.
        poseidon_hits = [e["damage"] for e in result.trace if e["phase"] == "DAMAGE_APPLY" and e["side"] == 0]
        self.assertEqual(poseidon_hits, [531, 531])

    def test_first_block_and_nonlethal_boundary(self):
        attacker = Fighter(10, 10)
        self.assertEqual(simulate(duel(attacker, Fighter(10, 10, block_mode=1))).p_b, 1)
        self.assertEqual(simulate(duel(attacker, Fighter(10, 10, block_mode=2))).p_a, 1)

    def test_dancer_followup_hits_bypass_horus_consumed_block(self):
        battle = compile_battle(load_catalog(), ((67,), (73,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode="sample"), trace=True)
        blocks = [e for e in result.trace if e["phase"] == "PRE_HIT" and e["outcome"] == "block"]
        dancer_damage = [e for e in result.trace if e["phase"] == "DAMAGE_APPLY" and e["side"] == 1]
        self.assertEqual([e["hit_index"] for e in blocks], [1])
        self.assertEqual([(e["hit_index"], e["damage"]) for e in dancer_damage], [(2, 33), (3, 33)])
        self.assertEqual([e["side"] for e in result.trace if e["phase"] == "ATTACK_DECLARE"], [0, 1])

    def test_each_multi_hit_dodges_independently(self):
        battle = duel(Fighter(1, 1, attacks_per_action=3), Fighter(3, 1, dodge_probability=.5))
        result = simulate(battle)
        # Only three landed hits win before the opponent's lethal action.
        self.assertEqual((result.p_a, result.p_b, result.unresolved), (.125, .875, 0))

    def test_dodged_hit_does_not_consume_block_during_multi_hit_action(self):
        battle = duel(Fighter(10, 1, attacks_per_action=3), Fighter(100, 100, dodge_probability=.5, block_mode=1))
        result = simulate(battle, Options(mode="sample", seed=1), trace=True)
        prevention = [e for e in result.trace if e["phase"] == "PRE_HIT"]
        self.assertEqual([(e["hit_index"], e["outcome"]) for e in prevention], [(1, "dodge"), (2, "block")])
        self.assertEqual([e["hit_index"] for e in result.trace if e["phase"] == "DAMAGE_APPLY" and e["side"] == 1], [3])

    def test_multi_hit_repeat_counter_counts_completed_actions(self):
        a = Fighter(10, 1, dodge_probability=.5, incoming_multiplier=0, attacks_per_action=3)
        b = Fighter(10, 1, dodge_probability=.5, incoming_multiplier=0, attacks_per_action=2)
        result = simulate(duel(a, b), Options(repeat_cycles=2))
        self.assertEqual((result.p_b, result.tie), (1, 0))
        self.assertEqual(result.expanded_states, 10)
        self.assertGreater(result.merged_states, 0)
        self.assertEqual(simulate(duel(a, b), Options(repeat_cycles=2, max_steps=9)).unresolved, 1)

    def test_knockout_cancels_followup_hits(self):
        for enemy in ((Fighter(1, 1),), (Fighter(1, 1), Fighter(10, 10))):
            battle = Battle(((Fighter(10, 10, attacks_per_action=3),), enemy))
            for mode in ("sample", "branch"):
                with self.subTest(enemy_count=len(enemy), mode=mode):
                    result = simulate(battle, Options(mode=mode))
                    self.assertEqual(result.p_a, 1 if len(enemy) == 1 else 0)
                    self.assertEqual(result.p_b, 0 if len(enemy) == 1 else 1)

    def test_dancer_does_not_retarget_poseidon_after_good_boy_dies(self):
        battle = compile_battle(load_catalog(), ((3, 44), (67,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode="sample"), trace=True)
        dancer_hits = [e for e in result.trace if e["phase"] == "DAMAGE_APPLY" and e["side"] == 0]
        self.assertEqual(len(dancer_hits), 1)
        self.assertEqual(dancer_hits[0]["hit_index"], 1)

    def test_simultaneous_depletion_loses_for_initiator(self):
        fighter = Fighter(10, 0)
        for initiator in (0, 1):
            result = simulate(duel(fighter, fighter, initiator), Options(repeat_cycles=1))
            self.assertEqual((result.p_a, result.p_b, result.tie), (initiator, 1-initiator, 0))

    def test_ko_on_final_hit_advances_normally(self):
        battle = Battle(((Fighter(5, 1, attacks_per_action=3),), (Fighter(3, 1), Fighter(10, 10))))
        result = simulate(battle, Options(mode="sample"), trace=True)
        self.assertEqual(result.p_b, 1)
        self.assertEqual(len([e for e in result.trace if e["phase"] == "TRANSITION"]), 1)
        self.assertEqual([e["side"] for e in result.trace if e["phase"] == "ATTACK_DECLARE"], [0, 1])

    def test_multi_hit_count_is_bounded_integer(self):
        for count in (0, -1, 17, 2.5, True):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "attacks_per_action"):
                    simulate(duel(Fighter(1, 1, attacks_per_action=count), Fighter(1, 1)))

    def test_compiler_preserves_per_card_mutation_assignment(self):
        catalog = load_catalog()
        battle = compile_battle(catalog, ((67, 67), (73,)), mutations=(("Storm", "Snow"), ("None",)))
        self.assertEqual([(f.hp, f.attack) for f in battle.teams[0]], [(143, 72), (156, 78)])
        self.assertEqual(battle.teams[1][0], compile_fighter(catalog, 73))
        self.assertEqual(battle.teams[0][0], compile_fighter(catalog, 67, mutation="Storm"))

    def test_compiler_rejects_missing_or_invalid_mutation_assignments(self):
        catalog = load_catalog()
        for mutations in ((("Storm",),), ("Storm", "None"), (("Storm", "Snow"), ("None",)),
                          ((None,), ("None",))):
            with self.subTest(mutations=mutations):
                with self.assertRaisesRegex(ValueError, "Each card requires exactly one mutation name"):
                    compile_battle(catalog, ((67,), (73,)), mutations=mutations)
        with self.assertRaisesRegex(ValueError, "Unknown mutation"):
            compile_fighter(catalog, 67, mutation="unverified")

    def test_repeat_deaths_enter_successors(self):
        immortal = Fighter(10, 1, incoming_multiplier=0)
        battle = Battle(((immortal, Fighter(10, 10)), (immortal, Fighter(10, 1))))
        result = simulate(battle, Options(mode="sample", repeat_cycles=1), trace=True)
        deaths = [e for e in result.trace if e["phase"] == "DEATH" and e["reason"] == "repeat_limit"]
        self.assertEqual(len(deaths), 2)
        self.assertEqual(len([e for e in result.trace if e["phase"] == "TRANSITION"]), 2)
        self.assertEqual(result.p_a, 1)

    def test_merge_identical_states_includes_repeat_counter(self):
        fighter = Fighter(10, 1, dodge_probability=.5, incoming_multiplier=0)
        result = simulate(duel(fighter, fighter), Options(repeat_cycles=2))
        self.assertEqual((result.p_b, result.tie), (1, 0))
        self.assertGreater(result.merged_states, 0)
        self.assertEqual(result.expanded_states, 4)

    def test_pruning_and_budget_account_for_all_probability(self):
        result = simulate(duel(Fighter(1, 1), Fighter(1, 1, dodge_probability=.1)), Options(prune_probability=.2))
        self.assertAlmostEqual(result.p_a, .9)
        self.assertAlmostEqual(result.unresolved, .1)
        result = simulate(duel(Fighter(10, 1), Fighter(10, 1)), Options(max_steps=1))
        self.assertEqual(result.unresolved, 1)
        self.assertEqual(result.accounted_probability, 1)

    def test_probability_frontier_cap_never_fabricates_labels(self):
        result = simulate(duel(Fighter(3, 1, dodge_probability=.5), Fighter(3, 1, dodge_probability=.5)), Options(max_frontier=1))
        self.assertGreater(result.unresolved, 0)
        self.assertAlmostEqual(result.accounted_probability, 1)

    def test_rounding_policies_are_explicit(self):
        battle = duel(Fighter(10, 1.4), Fighter(2.5, 20))
        for policy in ("ceil", "floor", "unrounded", "nearest_half_up"):
            result = simulate(battle, Options(mode="sample", rounding=policy), trace=True)
            hit = next(e for e in result.trace if e["phase"] == "DAMAGE_APPLY")
            self.assertEqual(hit["damage"], {"ceil": 2, "floor": 1, "unrounded": 1.4, "nearest_half_up": 1}[policy])

    def test_nonfinite_inputs_and_empty_teams_rejected(self):
        for fighter in (Fighter(float("nan"), 1), Fighter(1, -1), Fighter(1, 1, dodge_probability=1.1)):
            with self.assertRaises(ValueError):
                simulate(duel(fighter, Fighter(1, 1)))
        with self.assertRaises(ValueError):
            simulate(Battle(((), (Fighter(1, 1),))))

    def test_entry_self_scaling_cannot_create_zero_hp_attacker(self):
        battle = duel(Fighter(.2, 20, entry_self_multiplier=.5), Fighter(10, 1))
        for rounding in ("floor", "nearest_half_up"):
            with self.subTest(rounding=rounding):
                with self.assertRaisesRegex(ValueError, "Lethal entry self scaling"):
                    simulate(battle, Options(stat_rounding=rounding))

    def test_boolean_pruning_probability_rejected(self):
        for value in (False, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Invalid pruning probability"):
                    simulate(duel(Fighter(1, 1), Fighter(1, 1)), Options(prune_probability=value))

    def test_unsupported_card_or_support_does_not_silently_become_vanilla(self):
        catalog = load_catalog()
        with mock.patch.dict(SUPPORTED, {37: ("changed text", {})}), self.assertRaises(UnsupportedCardError):
            compile_battle(catalog, ((37,), (38,)))  # Every card is mapped; a changed description is unsupported.
        with mock.patch.dict(BLUE_SUPPORTS, {12: [None] * 5}), self.assertRaises(UnsupportedCardError):
            compile_battle(catalog, ((3,), (3,)), blue_supports=(12, 0))  # an unread support value
        result = simulate(compile_battle(catalog, ((3,), (3,))), Options(rounding="ceil"))
        self.assertEqual(result.p_a, 1)
        rows = coverage(catalog)
        self.assertEqual(len(rows), 289)
        self.assertTrue(all(not r["training_labels_allowed"] for r in rows))


if __name__ == "__main__":
    unittest.main()
