"""Synthetic healing contracts; parity is not evidence of true-game timing."""

from dataclasses import replace
import random
import tempfile
from pathlib import Path
import unittest

from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch
from card_engine.simulator import native
from card_engine.catalog import load_catalog
from card_engine.stats import base_stats
from card_engine.simulator.catalog_rules import compile_battle


class HealingReactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'healing.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for py, c in zip(simulate_batch(battles, options),
                         native.simulate_batch(battles, options, library_path=self.library), strict=True):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=10, msg=key)
            self.assertFalse(py.training_labels_allowed)

    def trace(self, a, b, steps=10, first_side=0):
        battle = Battle(((a,), (b,)), first_side)
        options = Options(mode='sample', max_steps=steps)
        self.compare([battle], options)
        return simulate(battle, options, trace=True).trace

    def test_lifesteal_changes_exact_outcome_probability_without_renormalizing(self):
        a = Fighter(6, 4, heal_damage_dealt_fraction=1)
        b = Fighter(8, 4, dodge_probability=.5)
        battle = Battle(((a,), (b,)), 1)
        result = simulate(battle, Options(max_steps=5))
        self.assertEqual((result.p_a, result.p_b, result.unresolved), (.25, .75, 0))
        short = simulate(battle, Options(max_steps=4))
        self.assertEqual((short.p_a, short.p_b, short.unresolved), (.25, .5, .25))
        without = Battle(((replace(a, heal_damage_dealt_fraction=0),), (b,)), 1)
        self.assertEqual(simulate(without).p_b, 1)
        self.compare([battle, without], Options(max_steps=5))

    def test_lifesteal_excludes_overkill_and_lethal_protected_hp(self):
        for survival, expected in ((0, 3), (1, 2)):
            events = self.trace(Fighter(10, 100, heal_damage_dealt_fraction=1),
                                Fighter(3, 7, lethal_survivals=survival), steps=2, first_side=1)
            heal = next(e for e in events if e['phase'] == 'LIFESTEAL')
            self.assertEqual((heal['amount'], heal['healed']), (expected, expected))

    def test_lifesteal_is_per_hit_before_counter_and_counter_can_lifesteal(self):
        events = self.trace(Fighter(30, 2, attacks_per_action=3, heal_damage_dealt_fraction=.5),
                            Fighter(30, 3, counter_on_damage=1, heal_damage_dealt_fraction=.5), steps=6)
        heals = [e for e in events if e['phase'] == 'LIFESTEAL']
        self.assertEqual([e['side'] for e in heals], [0, 1, 0, 1, 0, 1])
        self.assertEqual([e['amount'] for e in heals], [1, 2, 1, 2, 1, 2])
        phases = [e['phase'] for e in events]
        self.assertLess(phases.index('LIFESTEAL'), phases.index('COUNTER_DECLARE'))

    def test_prevented_or_zero_damage_never_heals(self):
        for defense in ({'block_mode': 1}, {'dodge_probability': 1}, {'invincible': 1},
                        {'incoming_multiplier': 0}, {'nullify_below_attack': 10}):
            events = self.trace(Fighter(10, 3, heal_damage_dealt_fraction=1),
                                Fighter(10, 2, heal_damage_taken_fraction=1, **defense), steps=1)
            self.assertFalse(any(e['phase'] in ('LIFESTEAL', 'DAMAGE_RECOVERY') for e in events))

    def test_surviving_damage_recovery_uses_post_defense_loss_before_counter(self):
        events = self.trace(Fighter(100, 10),
                            Fighter(20, 1, incoming_multiplier=.5, heal_damage_taken_fraction=.7,
                                    counter_on_damage=1), steps=2)
        recovery = next(e for e in events if e['phase'] == 'DAMAGE_RECOVERY')
        self.assertEqual((recovery['before'], recovery['amount'], recovery['hp']), (15, 4, 19))
        phases = [e['phase'] for e in events]
        self.assertLess(phases.index('DAMAGE_RECOVERY'), phases.index('COUNTER_DECLARE'))

    def test_recovery_cannot_rescue_lethal_hit_but_can_follow_survival(self):
        for survival in (0, 1):
            events = self.trace(Fighter(100, 20), Fighter(10, 1, heal_damage_taken_fraction=2,
                                                        lethal_survivals=survival), steps=1)
            heals = [e for e in events if e['phase'] == 'DAMAGE_RECOVERY']
            self.assertEqual(len(heals), survival)
            if survival:
                self.assertEqual((heals[0]['amount'], heals[0]['hp']), (18, 10))

    def test_action_end_heal_once_after_multihit_reactions_not_entry(self):
        a = Fighter(100, 1, attacks_per_action=3, entry_hit_multiplier=1,
                    action_end_heal_max_hp_fraction=.2)
        events = self.trace(a, Fighter(100, 10, counter_on_damage=1), steps=8)
        heals = [e for e in events if e['phase'] == 'TURN_END_HEAL']
        self.assertEqual(len(heals), 1)
        self.assertEqual((heals[0]['before'], heals[0]['hp']), (60, 80))
        self.assertEqual(sum(e['phase'] == 'COUNTER_DECLARE' for e in events), 4)

    def test_end_heal_after_block_and_each_extra_action_but_not_lethal_counter(self):
        a = Fighter(10, 1, actions_per_turn=2, action_end_heal_max_hp_fraction=.5)
        events = self.trace(a, Fighter(20, 1, block_mode=1), steps=2)
        self.assertEqual(sum(e['phase'] == 'TURN_END_HEAL' for e in events), 2)
        events = self.trace(a, Fighter(20, 20, counter_on_damage=1), steps=2)
        self.assertFalse(any(e['phase'] == 'TURN_END_HEAL' for e in events))

    def test_invalid_healing_and_arithmetic_overflow_are_rejected(self):
        for field in ('heal_damage_dealt_fraction', 'heal_damage_taken_fraction', 'action_end_heal_max_hp_fraction',
                      'after_attack_heal_max_hp_fraction'):
            for value in (-1, True, float('nan'), float('inf')):
                battle = Battle(((Fighter(10, 1, **{field: value}),), (Fighter(10, 1),)))
                with self.assertRaises(ValueError): simulate(battle)
                with self.assertRaises(ValueError): native.simulate(battle, library_path=self.library)
        cases = [Battle(((Fighter(1e308, 1, action_end_heal_max_hp_fraction=2),), (Fighter(10, 1),))),
                 Battle(((Fighter(1e308, 1, after_attack_heal_max_hp_fraction=2),), (Fighter(10, 1),))),
                 Battle(((Fighter(10, 2, heal_damage_dealt_fraction=1e308),), (Fighter(10, 1),))),
                 Battle(((Fighter(10, 2),), (Fighter(10, 1, heal_damage_taken_fraction=1e308),)))]
        for battle in cases:
            with self.assertRaises(ValueError): simulate(battle)
            with self.assertRaises(ValueError): native.simulate(battle, library_path=self.library)

    def test_mixed_healing_state_branch_and_sample_parity(self):
        rng = random.Random(974)
        battles = []
        for _ in range(35):
            teams = tuple(tuple(Fighter(rng.randint(2, 25), rng.randint(1, 8),
                heal_damage_dealt_fraction=rng.choice((0, .5, 1)),
                heal_damage_taken_fraction=rng.choice((0, .7)),
                action_end_heal_max_hp_fraction=rng.choice((0, .2)),
                after_attack_heal_max_hp_fraction=rng.choice((0, .2)),
                action_start_heal_max_hp_fraction=rng.choice((0, .35)),
                attacks_per_action=rng.choice((1, 3)), actions_per_turn=rng.choice((1, 2)),
                counter_on_damage=rng.randrange(2), entry_hit_multiplier=rng.choice((0, .5)),
                lethal_survivals=rng.randrange(2), lifetime_actions=rng.choice((0, 3)),
                dodge_probability=rng.choice((0, .25)), critical_probability=rng.choice((0, .5)))
                for _ in range(rng.randint(1, 3))) for _ in range(2))
            battles.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            self.compare(battles, Options(mode=mode, max_steps=30, max_frontier=80,
                         repeat_cycles=4, prune_probability=.0001, seed=29))

    def test_full_card_compositions_keep_labels_provisional(self):
        catalog = load_catalog()
        battles = [compile_battle(catalog, ((card,), (opponent,)), red_supports=(0, 0))
                   for card in (7, 74, 85, 174) for opponent in (73, 94, 210)]
        self.compare(battles)
        darling = compile_battle(catalog, ((85,), (73,)), red_supports=(0, 0))
        events = simulate(darling, Options(mode='sample', max_steps=1), trace=True).trace
        entry = next(e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0)
        heal = next(e for e in events if e['phase'] == 'TURN_START_HEAL')
        self.assertEqual(heal['max_hp'], entry['max_hp'])

    def test_forest_spirit_heals_before_lethal_raze_counter(self):
        battle = compile_battle(load_catalog(), ((7,), (210,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample'), trace=True)
        phases = [e['phase'] for e in result.trace]
        self.assertLess(phases.index('DAMAGE_APPLY'), phases.index('AFTER_ATTACK_HEAL'))
        self.assertLess(phases.index('AFTER_ATTACK_HEAL'), phases.index('COUNTER_DECLARE'))
        self.assertLess(phases.index('COUNTER_DECLARE'), phases.index('DEATH'))
        self.assertEqual(phases.count('AFTER_ATTACK_HEAL'), 1)
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_shu_uses_user_read_stats(self):
        stats = base_stats(load_catalog(), 74)
        self.assertEqual((stats.hp, stats.attack), (1230, 362))
        self.assertIn('1230 HP / 362 ATK', stats.verification)

    def test_shu_dies_to_lethal_raze_hit_without_recovery(self):
        battle = compile_battle(load_catalog(), ((74,), (210,)), red_supports=(0, 0), first_side=1)
        result = simulate(battle, Options(mode='sample'), trace=True)
        phases = [e['phase'] for e in result.trace]
        self.assertNotIn('DAMAGE_RECOVERY', phases)
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_pre_counter_healing_can_prevent_counter_lethality(self):
        a = Fighter(10, 2, after_attack_heal_max_hp_fraction=.5)
        b = Fighter(20, 6, counter_on_damage=1)
        battle = Battle(((a,), (b,)), 1)
        options = Options(max_steps=3)
        without = Battle(((replace(a, after_attack_heal_max_hp_fraction=0),), (b,)), 1)
        self.assertEqual(simulate(battle, options).unresolved, 1)
        self.assertEqual(simulate(without, options).p_b, 1)
        self.compare([battle, without], options)

    def test_count_muscula_lifesteals_before_lethal_raze_counter(self):
        battle = compile_battle(load_catalog(), ((174,), (210,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample'), trace=True)
        phases = [e['phase'] for e in result.trace]
        self.assertLess(phases.index('DAMAGE_APPLY'), phases.index('LIFESTEAL'))
        self.assertLess(phases.index('LIFESTEAL'), phases.index('COUNTER_DECLARE'))
        self.assertLess(phases.index('COUNTER_DECLARE'), phases.index('DEATH'))
        self.assertEqual(phases.count('LIFESTEAL'), 1)
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_after_attack_heal_is_not_repeated_by_entry_or_counter_resume(self):
        a = Fighter(100, 1, attacks_per_action=3, entry_hit_multiplier=1,
                    after_attack_heal_max_hp_fraction=.2)
        events = self.trace(a, Fighter(100, 10, counter_on_damage=1), steps=8)
        heals = [e for e in events if e['phase'] == 'AFTER_ATTACK_HEAL']
        self.assertEqual(len(heals), 1)
        self.assertEqual((heals[0]['before'], heals[0]['hp']), (70, 90))
        phases = [e['phase'] for e in events]
        last_counter = max(i for i, phase in enumerate(phases) if phase == 'COUNTER_DECLARE')
        self.assertLess(phases.index('AFTER_ATTACK_HEAL'), last_counter)
        # A knockout also completes the action before unused hits; no transfer.
        battle = Battle(((a,), (Fighter(1, 0), Fighter(100, 0))))
        result = simulate(battle, Options(mode='sample', max_steps=2), trace=True)
        # Entry kills the first enemy; next normal hit starts a 3-hit action.
        self.assertFalse(any(e['phase'] == 'AFTER_ATTACK_HEAL' for e in result.trace))
        self.compare([battle], Options(max_steps=2))
        battle = Battle(((replace(a, entry_hit_multiplier=0),), (Fighter(1, 0), Fighter(100, 0))))
        result = simulate(battle, Options(mode='sample', max_steps=1), trace=True)
        self.assertEqual(sum(e['phase'] == 'AFTER_ATTACK_HEAL' for e in result.trace), 1)
        self.compare([battle], Options(max_steps=1))


if __name__ == '__main__':
    unittest.main()
