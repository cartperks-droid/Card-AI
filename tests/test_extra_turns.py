"""Separate action allowances versus per-action hits and reaction interruptions."""
from pathlib import Path
import random
import tempfile
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import Battle, Fighter, Options, simulate, simulate_batch
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator import native


class ExtraTurnTests(unittest.TestCase):
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

    def test_wind_spirit_gets_two_actions_every_cycle_with_one_horus_block(self):
        battle = compile_battle(load_catalog(), ((19,), (73,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample'), trace=True)
        self.assertEqual([e['side'] for e in result.trace if e['phase'] == 'ATTACK_DECLARE'], [0, 0, 1, 0, 0, 1])
        self.assertEqual([e['outcome'] for e in result.trace if e['phase'] == 'PRE_HIT'], ['block'])
        self.assertEqual([e['hit_index'] for e in result.trace if e['phase'] == 'HIT_DECLARE'], [1]*6)
        self.compare([battle])

    def test_two_sides_receive_their_own_action_allowance(self):
        a, b = Fighter(1000, 1, actions_per_turn=2), Fighter(1000, 1, actions_per_turn=3)
        for first in (0, 1):
            battle = Battle(((a,), (b,)), first)
            result = simulate(battle, Options(mode='sample', max_steps=10), trace=True)
            sides = [e['side'] for e in result.trace if e['phase'] == 'ATTACK_DECLARE']
            cycle = [0, 0, 1, 1, 1] if first == 0 else [1, 1, 1, 0, 0]
            self.assertEqual(sides, cycle*2)
            self.compare([battle], Options(max_steps=10))

    def test_wind_spirit_unused_extra_turn_attacks_replacement_poseidon(self):
        battle = compile_battle(load_catalog(), ((3, 44), (19,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample', max_steps=4), trace=True)
        attacks = [e for e in result.trace if e['phase'] == 'ATTACK_DECLARE']
        self.assertEqual([(e['side'], e['slot'], e['target_slot']) for e in attacks],
                         [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 1, 0)])
        self.compare([battle])

    def test_multi_hit_actions_and_counters_do_not_consume_extra_action(self):
        a = Fighter(1000, 1, actions_per_turn=2, attacks_per_action=3, entry_hit_multiplier=1)
        b = Fighter(1000, 1, counter_on_damage=1)
        battle = Battle(((a,), (b,)))
        result = simulate(battle, Options(mode='sample', max_steps=15), trace=True)
        self.assertEqual([e['side'] for e in result.trace if e['phase'] == 'ATTACK_DECLARE'], [0, 0, 1])
        self.assertEqual(len([e for e in result.trace if e['phase'] == 'COUNTER_DECLARE']), 7)
        self.assertEqual([e['hit_index'] for e in result.trace if e['phase'] == 'HIT_DECLARE' and e['kind'] == 'normal'], [1, 2, 3, 1, 2, 3, 1])
        self.compare([battle], Options(max_steps=15))

    def test_lifetime_expires_after_own_actions_not_whole_allowance_cycles(self):
        a = Fighter(1, 1, invincible=1, lifetime_actions=3, actions_per_turn=2)
        battle = Battle(((a,), (Fighter(100, 1),)))
        result = simulate(battle, Options(mode='sample'), trace=True)
        self.assertEqual([e['side'] for e in result.trace if e['phase'] == 'ATTACK_DECLARE'], [0, 0, 1, 0])
        self.assertEqual([e['completed_actions'] for e in result.trace if e['phase'] == 'ACTION_END'], [1, 2, 3])
        self.compare([battle])

    def test_counter_kill_discards_dead_fighters_unused_actions(self):
        battle = Battle(((Fighter(1, 1, actions_per_turn=3), Fighter(100, 1)),
                         (Fighter(100, 1, counter_on_damage=1),)))
        result = simulate(battle, Options(mode='sample', max_steps=3), trace=True)
        self.assertEqual([(e['side'], e['slot']) for e in result.trace if e['phase'] == 'ATTACK_DECLARE'], [(0, 0), (1, 0)])
        self.compare([battle])

    def test_invalid_action_allowances_fail_explicitly(self):
        for n in (0, 17, -1, 1.5, True):
            battle = Battle(((Fighter(10, 1, actions_per_turn=n),), (Fighter(10, 1),)))
            with self.assertRaises(ValueError):
                simulate(battle)
            with self.assertRaises(ValueError):
                native.simulate(battle, library_path=self.library)

    def test_replacement_entry_kill_does_not_transfer_dead_cards_extra_allowance(self):
        battle = Battle(((Fighter(1, 2, actions_per_turn=3), Fighter(100, 1)),
                         (Fighter(1, 1), Fighter(100, 1, entry_hit_multiplier=1))))
        result = simulate(battle, Options(mode='sample', max_steps=3), trace=True)
        hits = [e for e in result.trace if e['phase'] == 'HIT_DECLARE']
        self.assertEqual([(e['side'], e['slot'], e['kind']) for e in hits],
                         [(0, 0, 'normal'), (1, 1, 'entry'), (1, 1, 'normal')])
        self.compare([battle])

    def test_branch_state_preserves_extra_actions_across_reactions(self):
        rng = random.Random(211)
        battles = []
        for _ in range(32):
            teams = tuple(tuple(Fighter(hp=rng.randint(2, 40), attack=rng.randint(1, 10),
                            actions_per_turn=rng.choice((1, 2, 3)), attacks_per_action=rng.choice((1, 3)),
                            lifetime_actions=rng.choice((0, 2, 3)), invincible=rng.randrange(2),
                            entry_hit_multiplier=rng.choice((0, .5, 1)), counter_on_damage=rng.randrange(2),
                            dodge_probability=rng.choice((0, .25)), critical_probability=rng.choice((0, .5)))
                          for _ in range(rng.randint(1, 3))) for _ in range(2))
            battles.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            self.compare(battles, Options(mode=mode, max_steps=35, max_frontier=100, repeat_cycles=3,
                                         prune_probability=.0001, seed=53))


if __name__ == '__main__':
    unittest.main()
