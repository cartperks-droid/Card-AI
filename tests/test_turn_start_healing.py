"""Turn-start healing is one event per normal action, including at full HP."""
from pathlib import Path
import random
import tempfile
import unittest

from card_engine.catalog import load_catalog
from card_engine.stats import base_stats
from card_engine.simulator import Battle, Fighter, Options, simulate, simulate_batch
from card_engine.simulator.catalog_rules import compile_fighter
from card_engine.simulator import native


class TurnStartHealingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / native.DEFAULT_LIBRARY.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        py = simulate_batch(battles, options)
        c = native.simulate_batch(battles, options, library_path=self.library)
        for i, (expected, actual) in enumerate(zip(py, c, strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states'):
                self.assertAlmostEqual(getattr(expected, key), actual[key], places=10, msg=f'{i}/{key}')

    def test_witch_healing_activates_before_attack_even_at_full_health(self):
        catalog = load_catalog()
        stats = base_stats(catalog, 94)
        witch = compile_fighter(catalog, 94)
        battle = Battle(((witch,), (compile_fighter(catalog, 73),)))
        result = simulate(battle, Options(mode='sample'), trace=True)
        heals = [(i, e) for i, e in enumerate(result.trace) if e['phase'] == 'TURN_START_HEAL']
        attacks = [i for i, e in enumerate(result.trace) if e['phase'] == 'ATTACK_DECLARE' and e['side'] == 0]
        self.assertEqual(len(heals), 4)
        self.assertTrue(all(i < attack for (i, _), attack in zip(heals, attacks, strict=True)))
        self.assertEqual(heals[0][1]['before'], stats.hp)
        self.assertEqual(heals[0][1]['healed'], 0)
        self.assertEqual(heals[1][1]['amount'], 304)  # Max-HP/ceiling policy, not a reported number.
        self.compare([battle])

    def test_video_witch_mirror_reaches_100_turn_cutoff_with_sourced_stats(self):
        witch = compile_fighter(load_catalog(), 94)
        self.assertEqual((witch.hp, witch.attack), (868, 290))
        battle = Battle(((witch,), (witch,)))
        result = simulate(battle, Options(mode='sample'), trace=True)
        attacks = [e['side'] for e in result.trace if e['phase'] == 'ATTACK_DECLARE']
        self.assertEqual(attacks, [0, 1]*50)
        self.assertEqual([e['reason'] for e in result.trace if e['phase'] == 'DEATH'], ['repeat_limit']*2)
        heals = [e for e in result.trace if e['phase'] == 'TURN_START_HEAL']
        self.assertEqual(heals[0]['healed'], 0)
        self.assertTrue(all(e['hp'] == 868 for e in heals))
        self.assertTrue(all(e['healed'] == 290 for e in heals[1:]))
        self.assertFalse(result.training_labels_allowed)
        self.compare([battle])

    def test_witch_stat_ratio_is_applied_before_rounding_without_changing_card_modifier(self):
        import math
        catalog = load_catalog()
        stats = base_stats(catalog, 94)
        self.assertEqual(stats.card_modifier, 1)
        self.assertEqual((stats.hp_ratio_multiplier, stats.attack_ratio_multiplier), (1.2, .8))
        self.assertEqual(stats.hp, 868)
        self.assertEqual(math.ceil(math.ceil(10 * 2**math.log10(1500000))*1.2), 869)
        self.assertIn('868 HP user-confirmed', stats.verification)
        self.assertIn('Immortal_Witch', stats.stat_ratio_source)
        self.assertIn('extrapolated', base_stats(catalog, 94, mutation='Storm').verification)
        for identifier in (33, 38):
            rapture = base_stats(catalog, identifier)
            self.assertEqual((rapture.hp, rapture.attack), (2560, 1280))
            self.assertEqual((rapture.hp_ratio_multiplier, rapture.attack_ratio_multiplier), (1, 1))

    def test_healing_can_change_exact_branch_win_probability(self):
        a = Fighter(2, 1, action_start_heal_max_hp_fraction=.35)
        b = Fighter(2, 1, critical_probability=.5)
        battle = Battle(((a,), (b,)), first_side=1)
        result = simulate(battle, Options(max_steps=4))
        self.assertEqual((result.p_a, result.p_b, result.unresolved), (.25, .75, 0))
        self.compare([battle], Options(max_steps=4))

    def test_multihits_and_resuming_after_counters_do_not_reheal(self):
        a = Fighter(100, 1, attacks_per_action=3, action_start_heal_max_hp_fraction=.1)
        battle = Battle(((a,), (Fighter(100, 1, counter_on_damage=1),)))
        result = simulate(battle, Options(mode='sample', max_steps=13), trace=True)
        heals = [e for e in result.trace if e['phase'] == 'TURN_START_HEAL']
        self.assertEqual([e['healed'] for e in heals], [0, 4])
        damage = [e for e in result.trace if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 0]
        self.assertEqual(damage[-1]['hp'], 97)
        self.compare([battle], Options(max_steps=13))

    def test_extra_actions_each_get_a_start_heal(self):
        a = Fighter(100, 1, actions_per_turn=2, action_start_heal_max_hp_fraction=.1)
        battle = Battle(((a,), (Fighter(100, 3, counter_on_damage=1),)))
        result = simulate(battle, Options(mode='sample', max_steps=4), trace=True)
        self.assertEqual([e['healed'] for e in result.trace if e['phase'] == 'TURN_START_HEAL'], [0, 3])
        self.compare([battle], Options(max_steps=4))

    def test_entry_hits_and_counters_do_not_trigger_start_healing(self):
        a = Fighter(100, 1, entry_hit_multiplier=1, counter_on_damage=1, action_start_heal_max_hp_fraction=.1)
        b = Fighter(100, 1, entry_hit_multiplier=1, counter_on_damage=1)
        battle = Battle(((a,), (b,)))
        result = simulate(battle, Options(mode='sample', max_steps=4), trace=True)
        self.assertFalse(any(e['phase'] == 'TURN_START_HEAL' for e in result.trace))
        result = simulate(battle, Options(mode='sample', max_steps=5), trace=True)
        self.assertEqual([e['healed'] for e in result.trace if e['phase'] == 'TURN_START_HEAL'], [2])
        self.compare([battle], Options(max_steps=5))

    def test_healing_uses_modified_max_hp_and_caps_effective_gain(self):
        a = Fighter(10, 1, entry_hp_multiplier=2, action_start_heal_max_hp_fraction=.35)
        battle = Battle(((a,), (Fighter(100, 3),)), first_side=1)
        result = simulate(battle, Options(mode='sample', max_steps=2), trace=True)
        heal = next(e for e in result.trace if e['phase'] == 'TURN_START_HEAL')
        self.assertEqual((heal['before'], heal['amount'], heal['healed'], heal['hp'], heal['max_hp']), (17, 7, 3, 20, 20))
        self.compare([battle], Options(max_steps=2))

    def test_healing_cap_never_increases_full_or_max_hp(self):
        a = Fighter(10, 1, action_start_heal_max_hp_fraction=2)
        battle = Battle(((a,), (Fighter(100, 1),)))
        result = simulate(battle, Options(mode='sample', max_steps=1), trace=True)
        heal = next(e for e in result.trace if e['phase'] == 'TURN_START_HEAL')
        self.assertEqual((heal['amount'], heal['healed'], heal['hp'], heal['max_hp']), (20, 0, 10, 10))
        self.compare([battle], Options(max_steps=1))

    def test_dodged_attack_still_gets_start_heal(self):
        a = Fighter(10, 1, action_start_heal_max_hp_fraction=.2)
        battle = Battle(((a,), (Fighter(10, 3, dodge_probability=1),)), first_side=1)
        result = simulate(battle, Options(mode='sample', max_steps=2), trace=True)
        self.assertEqual([e['hp'] for e in result.trace if e['phase'] == 'TURN_START_HEAL'], [9])
        self.assertEqual([e['outcome'] for e in result.trace if e['phase'] == 'PRE_HIT'], ['dodge'])
        self.compare([battle], Options(max_steps=2))

    def test_dead_healer_never_starts_action_or_revives_itself(self):
        a = Fighter(1, 1, action_start_heal_max_hp_fraction=10)
        battle = Battle(((a, Fighter(100, 1)), (Fighter(10, 2),)), first_side=1)
        result = simulate(battle, Options(mode='sample', max_steps=2), trace=True)
        self.assertFalse(any(e['phase'] == 'TURN_START_HEAL' for e in result.trace))
        self.compare([battle])

    def test_invalid_and_overflowing_healing_fails_before_cap_masks_it(self):
        for kwargs in ({'action_start_heal_max_hp_fraction': -1}, {'action_start_heal_max_hp_fraction': float('inf')},
                       {'action_start_heal_max_hp_fraction': True}, {'action_start_heal_max_hp_fraction': float('nan')}):
            battle = Battle(((Fighter(10, 1, **kwargs),), (Fighter(10, 1),)))
            with self.assertRaises(ValueError):
                simulate(battle)
            with self.assertRaises(ValueError):
                native.simulate(battle, library_path=self.library)
        for hp, fraction in ((1e308, 2), (1e308, 1)):
            battle = Battle(((Fighter(hp, 1, action_start_heal_max_hp_fraction=fraction),), (Fighter(10, 1),)))
            with self.assertRaisesRegex(ValueError, 'overflow'):
                simulate(battle)
            with self.assertRaises(ValueError):
                native.simulate(battle, library_path=self.library)

    def test_healing_branch_mass_and_state_native_parity(self):
        rng = random.Random(241)
        battles = []
        for _ in range(32):
            teams = tuple(tuple(Fighter(hp=rng.randint(2, 30), attack=rng.randint(1, 10),
                            action_start_heal_max_hp_fraction=rng.choice((0, .1, .35)),
                            entry_hp_multiplier=rng.choice((1, 1.2)), actions_per_turn=rng.choice((1, 2)),
                            attacks_per_action=rng.choice((1, 3)), action_end_attack_multiplier=rng.choice((1, 1.3)),
                            lifetime_actions=rng.choice((0, 3)), entry_hit_multiplier=rng.choice((0, .5)),
                            counter_on_damage=rng.randrange(2), dodge_probability=rng.choice((0, .25)),
                            critical_probability=rng.choice((0, .5))) for _ in range(rng.randint(1, 3))) for _ in range(2))
            battles.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            self.compare(battles, Options(mode=mode, max_steps=35, max_frontier=100, repeat_cycles=3,
                                         prune_probability=.0001, seed=61))


if __name__ == '__main__':
    unittest.main()
