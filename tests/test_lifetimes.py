"""Own-action lifetime expiration after reactions, with invincibility."""
from pathlib import Path
import random
import tempfile
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import Battle, Fighter, Options, simulate, simulate_batch
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator import native


class LifetimeTests(unittest.TestCase):
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

    def test_inari_ignores_damage_and_expires_after_third_own_attack(self):
        battle = compile_battle(load_catalog(), ((114,), (38,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample'), trace=True)
        attacks = [e['side'] for e in result.trace if e['phase'] == 'ATTACK_DECLARE']
        self.assertEqual(attacks, [0, 1, 0, 1, 0])
        self.assertFalse(any(e['phase'] == 'DAMAGE_APPLY' and e['side'] == 0 for e in result.trace))
        self.assertEqual([e['reason'] for e in result.trace if e['phase'] == 'DEATH'], ['lifetime'])
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_raze_third_counter_precedes_inari_death(self):
        battle = compile_battle(load_catalog(), ((114,), (210,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample'), trace=True)
        counters = [i for i, e in enumerate(result.trace) if e['phase'] == 'COUNTER_DECLARE']
        death = next(i for i, e in enumerate(result.trace) if e['phase'] == 'DEATH')
        self.assertEqual(len(counters), 3)
        self.assertLess(counters[-1], death)
        self.assertEqual(result.trace[counters[-1]+2]['outcome'], 'invincible')
        self.assertEqual([e['completed_actions'] for e in result.trace if e['phase'] == 'ACTION_END'], [1, 2, 3])
        self.compare([battle])

    def test_entries_counters_and_individual_hits_do_not_age_lifetime(self):
        a = Fighter(1000, 1, lifetime_actions=2, entry_hit_multiplier=1, counter_on_damage=1, attacks_per_action=3)
        b = Fighter(1000, 1, entry_hit_multiplier=1, counter_on_damage=1)
        battle = Battle(((a,), (b,)))
        result = simulate(battle, Options(mode='sample'), trace=True)
        hits = [e for e in result.trace if e['phase'] == 'HIT_DECLARE' and e['side'] == 0]
        self.assertEqual([e['kind'] for e in hits].count('normal'), 6)
        self.assertEqual([e['kind'] for e in hits].count('entry'), 1)
        self.assertEqual([e['kind'] for e in hits].count('counter'), 2)
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_lifetime_survives_enemy_replacements_and_is_per_card(self):
        a = Fighter(1, 1, invincible=1, lifetime_actions=3)
        battle = Battle(((a, a), (Fighter(1, 1), Fighter(1, 1), Fighter(1, 1), Fighter(100, 1))))
        result = simulate(battle, Options(mode='sample'), trace=True)
        endings = [e for e in result.trace if e['phase'] == 'ACTION_END']
        self.assertEqual([(e['slot'], e['completed_actions']) for e in endings],
                         [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)])
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_expiration_does_not_award_enemy_kill_heal_or_use_lethal_charge(self):
        a = Fighter(10, 1, invincible=1, lifetime_actions=1, lethal_survivals=1)
        b = Fighter(10, 1, heal_on_kill=1)
        battle = Battle(((a,), (b,)))
        result = simulate(battle, Options(mode='sample'), trace=True)
        self.assertFalse(any(e['phase'] in ('ON_KILL', 'LETHAL_REPLACEMENT') for e in result.trace))
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_simultaneous_damage_and_lifetime_depletion_loses_for_initiator(self):
        a = Fighter(10, 10, invincible=1, lifetime_actions=1)
        b = Fighter(1, 1)
        for first, teams in ((0, ((a,), (b,))), (1, ((b,), (a,)))):
            battle = Battle(teams, first)
            result = simulate(battle, Options(mode='sample'), trace=True)
            self.assertEqual((result.p_a, result.p_b), (first, 1-first))
            self.assertEqual([e['reason'] for e in result.trace if e['phase'] == 'DEATH'], ['damage', 'lifetime'])
            self.compare([battle])

    def test_invincibility_protects_from_entry_and_critical_hits(self):
        a = Fighter(1, 1, invincible=1, lifetime_actions=1, block_mode=1)
        b = Fighter(100, 100, entry_hit_multiplier=10, critical_probability=1)
        battle = Battle(((a,), (b,)), first_side=1)
        result = simulate(battle, Options(mode='sample'), trace=True)
        self.assertEqual([e['outcome'] for e in result.trace if e['phase'] == 'PRE_HIT'], ['invincible', 'invincible'])
        self.compare([battle])

    def test_invalid_lifetime_and_invincibility_fail_before_simulation(self):
        for kwargs in ({'lifetime_actions': -1}, {'lifetime_actions': True}, {'lifetime_actions': 2**31},
                       {'invincible': 2}, {'invincible': .5}, {'invincible': True}):
            battle = Battle(((Fighter(10, 1, **kwargs),), (Fighter(10, 1),)))
            with self.assertRaises(ValueError):
                simulate(battle)
            with self.assertRaises(ValueError):
                native.simulate(battle, library_path=self.library)

    def test_lifetime_and_pending_entry_state_branch_parity(self):
        rng = random.Random(193)
        battles = []
        for _ in range(32):
            teams = tuple(tuple(Fighter(hp=rng.randint(2, 30), attack=rng.randint(1, 10),
                             invincible=rng.randrange(2), lifetime_actions=rng.choice((0, 1, 2, 3)),
                             entry_hit_multiplier=rng.choice((0, .5, 1)), counter_on_damage=rng.randrange(2),
                             attacks_per_action=rng.choice((1, 3)), dodge_probability=rng.choice((0, .25)),
                             critical_probability=rng.choice((0, .5)), heal_on_kill=rng.randrange(2))
                          for _ in range(rng.randint(1, 3))) for _ in range(2))
            battles.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            self.compare(battles, Options(mode=mode, max_steps=35, max_frontier=100, repeat_cycles=3,
                                         prune_probability=.0001, seed=47))


if __name__ == '__main__':
    unittest.main()
