"""Observed Martial Artist growth plus action/reaction boundary regressions."""
from pathlib import Path
import random
import tempfile
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import Battle, Fighter, Options, simulate, simulate_batch
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator import native


class TurnEndStatTests(unittest.TestCase):
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

    def test_martial_artist_boosts_after_blocked_hit_before_horus_attacks(self):
        battle = compile_battle(load_catalog(), ((121,), (73,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample'), trace=True)
        boosts = [(i, e) for i, e in enumerate(result.trace) if e['phase'] == 'TURN_END_STAT']
        replies = [i for i, e in enumerate(result.trace) if e['phase'] == 'ATTACK_DECLARE' and e['side'] == 1]
        self.assertEqual([e['displayed_attack'] for _, e in boosts], [338, 440])
        self.assertEqual(boosts[0][1]['before'], 260)
        self.assertAlmostEqual(boosts[1][1]['attack'], 439.4)
        self.assertTrue(all(boost < reply for (boost, _), reply in zip(boosts, replies, strict=True)))
        self.assertEqual([e['outcome'] for e in result.trace if e['phase'] == 'PRE_HIT'], ['block'])
        self.compare([battle])

    def test_raze_lethal_counter_prevents_martial_artist_boost(self):
        battle = compile_battle(load_catalog(), ((121,), (210,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample'), trace=True)
        self.assertEqual([e['kind'] for e in result.trace if e['phase'] == 'HIT_DECLARE'], ['normal', 'counter'])
        self.assertFalse(any(e['phase'] == 'TURN_END_STAT' for e in result.trace))
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_buff_runs_once_after_all_hits_and_their_counters(self):
        a = Fighter(1000, 10, attacks_per_action=3, action_end_attack_multiplier=2)
        b = Fighter(1000, 1, counter_on_damage=1)
        battle = Battle(((a,), (b,)))
        result = simulate(battle, Options(mode='sample', max_steps=7), trace=True)
        damage = [e['damage'] for e in result.trace if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1]
        boosts = [i for i, e in enumerate(result.trace) if e['phase'] == 'TURN_END_STAT']
        counter_hits = [i for i, e in enumerate(result.trace) if e['phase'] == 'HIT_DECLARE' and e['kind'] == 'counter']
        self.assertEqual(damage, [10, 10, 10])
        self.assertEqual(len(boosts), 1)
        self.assertGreater(boosts[0], counter_hits[-1])
        self.compare([battle], Options(max_steps=7))

    def test_entry_hits_and_counters_do_not_apply_own_turn_end_boost(self):
        a = Fighter(1000, 10, entry_hit_multiplier=1, counter_on_damage=1, action_end_attack_multiplier=2)
        b = Fighter(1000, 1, entry_hit_multiplier=1, counter_on_damage=1)
        battle = Battle(((a,), (b,)))
        result = simulate(battle, Options(mode='sample', max_steps=4), trace=True)
        self.assertEqual([e['kind'] for e in result.trace if e['phase'] == 'HIT_DECLARE'], ['entry', 'counter', 'entry', 'counter'])
        self.assertFalse(any(e['phase'] == 'TURN_END_STAT' for e in result.trace))
        self.compare([battle], Options(max_steps=4))

    def test_each_extra_action_boosts_and_growth_uses_current_entry_modified_attack(self):
        a = Fighter(1000, 10, entry_attack_multiplier=3, actions_per_turn=2, action_end_attack_multiplier=1.3)
        battle = Battle(((a,), (Fighter(1000, 1),)))
        result = simulate(battle, Options(mode='sample', max_steps=2), trace=True)
        self.assertEqual([e['damage'] for e in result.trace if e['phase'] == 'DAMAGE_APPLY'], [30, 39])
        self.assertEqual([e['displayed_attack'] for e in result.trace if e['phase'] == 'TURN_END_STAT'], [39, 51])
        self.compare([battle], Options(max_steps=2))

    def test_growth_persists_across_enemy_replacements_and_does_not_transfer_to_new_ally(self):
        a = Fighter(1, 2, action_end_attack_multiplier=2)
        battle = Battle(((a, Fighter(100, 3)), (Fighter(1, 1), Fighter(100, 1))))
        result = simulate(battle, Options(mode='sample', max_steps=3), trace=True)
        self.assertEqual([e['displayed_attack'] for e in result.trace if e['phase'] == 'TURN_END_STAT'], [4])
        self.assertEqual([e['damage'] for e in result.trace if e['phase'] == 'DAMAGE_APPLY'], [2, 1, 3])
        self.compare([battle])

    def test_fractional_and_per_effect_rounding_are_explicit_alternatives(self):
        a = Fighter(1000, 260, action_end_attack_multiplier=1.3)
        battle = Battle(((a,), (Fighter(10000, 1),)))
        expected = {'unrounded': [338, 440, 572], 'ceil': [338, 440, 572]}
        # These three displays do not distinguish the alternatives: inspect the
        # retained fractional state instead of claiming the game revealed it.
        for policy in ('unrounded', 'ceil'):
            result = simulate(battle, Options(mode='sample', max_steps=5, stat_rounding=policy), trace=True)
            boosts = [e for e in result.trace if e['phase'] == 'TURN_END_STAT']
            self.assertEqual([e['displayed_attack'] for e in boosts], expected[policy])
            self.assertAlmostEqual(boosts[1]['attack'], 439.4 if policy == 'unrounded' else 440)
            self.compare([battle], Options(max_steps=5, stat_rounding=policy))

    def test_invalid_or_overflowing_growth_is_rejected_even_at_terminal_action(self):
        for factor in (-1, float('inf'), float('nan'), True):
            battle = Battle(((Fighter(10, 1, action_end_attack_multiplier=factor),), (Fighter(10, 1),)))
            with self.assertRaises(ValueError):
                simulate(battle)
            with self.assertRaises(ValueError):
                native.simulate(battle, library_path=self.library)
        battle = Battle(((Fighter(10, 1e308, action_end_attack_multiplier=2),), (Fighter(1, 1),)))
        with self.assertRaisesRegex(ValueError, 'overflow'):
            simulate(battle)
        with self.assertRaises(ValueError):
            native.simulate(battle, library_path=self.library)

    def test_branching_growth_and_reactions_native_parity(self):
        rng = random.Random(229)
        battles = []
        for _ in range(32):
            teams = tuple(tuple(Fighter(hp=rng.randint(2, 50), attack=rng.randint(1, 10),
                            action_end_attack_multiplier=rng.choice((0, .7, 1, 1.3)),
                            actions_per_turn=rng.choice((1, 2)), attacks_per_action=rng.choice((1, 3)),
                            lifetime_actions=rng.choice((0, 3)), invincible=rng.randrange(2),
                            entry_hit_multiplier=rng.choice((0, .5)), counter_on_damage=rng.randrange(2),
                            dodge_probability=rng.choice((0, .25)), critical_probability=rng.choice((0, .5)))
                          for _ in range(rng.randint(1, 3))) for _ in range(2))
            battles.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            self.compare(battles, Options(mode=mode, max_steps=35, max_frontier=100, repeat_cycles=3,
                                         prune_probability=.0001, seed=59))


if __name__ == '__main__':
    unittest.main()
