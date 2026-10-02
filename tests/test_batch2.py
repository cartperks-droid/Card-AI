"""Batch-2 extrapolated primitives: card-text readings pinned for semantics and
Python/C parity, not game observations."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import (Battle, Fighter, Options, simulate, simulate_batch,
                                             _BATCH2_FLOATS, _BATCH2_INTS)

BATCH2_CARDS = (23, 24, 25, 42, 54, 62, 82, 89, 96, 97, 102, 105, 108, 113, 118, 134, 140, 149, 156, 158, 161,
                169, 171, 173, 179, 188, 190, 193, 200, 221, 242, 243, 244, 246, 248, 249, 287)


class BatchTwoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'batch2.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def events(self, a, b, first_side=0, steps=20):
        battle = Battle((tuple(a), tuple(b)), first_side)
        self.compare([battle], Options(mode='sample', max_steps=steps))
        return simulate(battle, Options(mode='sample', max_steps=steps), trace=True).trace

    def test_every_batch_card_runs_in_both_engines(self):
        catalog = load_catalog()
        battles = [compile_battle(catalog, ((7, card), (opponent, 3)), red_supports=(0, 0), first_side=side)
                   for card in BATCH2_CARDS for opponent in (73, 210, 94) for side in (0, 1)]
        self.compare(battles, Options(max_frontier=200, prune_probability=1e-9))

    def test_recharge_turns_attack_normally(self):
        # User: recharge turns attack without the charged bonus; nothing is skipped.
        events = self.events([Fighter(100, 5, recharge_turns=1, charged_attack_multiplier=2)], [Fighter(1000, 1)], steps=8)
        damage = [e['damage'] for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1]
        self.assertEqual(damage, [10, 5, 10, 5])

    def test_user_observed_alternation_bypass_and_conversion_cost(self):
        catalog = load_catalog()
        # Amaterasu (#118) nullifies even hits; Deus Ex (#54) dodges odd hits.
        events = self.events([Fighter(100, 5)], [Fighter(100, 1, alternate_dodge=2)], steps=7)
        outcomes = [e.get('outcome', 'hit') for e in events if e.get('side') == 1 and e['phase'] in ('PRE_HIT', 'DAMAGE_APPLY')]
        self.assertEqual(outcomes, ['hit', 'alternate_dodge', 'hit', 'alternate_dodge'])
        # Bypass defeats both Horus's block and a 60% chance dodge.
        for defender in (73, 93):
            battle = compile_battle(catalog, ((149,), (defender,)), red_supports=(0, 0))
            first = next(e for e in simulate(battle, Options(mode='sample'), trace=True).trace
                         if e.get('side') == 1 and e['phase'] in ('PRE_HIT', 'DAMAGE_APPLY'))
            self.assertEqual(first['phase'], 'DAMAGE_APPLY')
        # #161 loses the converted 15% of max HP with every attack.
        events = self.events([Fighter(100, 1, max_hp_damage_fraction=.15, max_hp_damage_costs=1)], [Fighter(10**6, 0)], steps=5)
        damage = [e['damage'] for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1]
        self.assertEqual(damage, [16, 14, 12])  # 1 + 15, 1 + 12.75, 1 + 10.84 (rounded up).

    def test_first_turn_and_growing_allowances(self):
        events = self.events([Fighter(100, 1, first_turn_actions=3)], [Fighter(1000, 1)], steps=6)
        sides = [e['side'] for e in events if e['phase'] == 'ATTACK_DECLARE']
        self.assertEqual(sides[:6], [0, 0, 0, 1, 0, 1])
        events = self.events([Fighter(100, 1, actions_per_turn=2, actions_growth_per_turn=1)], [Fighter(1000, 1)], steps=9)
        sides = [e['side'] for e in events if e['phase'] == 'ATTACK_DECLARE']
        self.assertEqual(sides[:9], [0, 0, 1, 0, 0, 0, 1, 0, 0])

    def test_alternate_dodge_starts_with_the_first_attack(self):
        events = self.events([Fighter(100, 5)], [Fighter(100, 1, alternate_dodge=1)], steps=7)
        outcomes = [e.get('outcome', 'hit') for e in events
                    if e.get('side') == 1 and e['phase'] in ('PRE_HIT', 'DAMAGE_APPLY')]
        self.assertEqual(outcomes, ['alternate_dodge', 'hit', 'alternate_dodge', 'hit'])

    def test_revive_at_full_and_first_attack_bonus(self):
        events = self.events([Fighter(10, 1000, first_attack_multiplier=5)],
                             [Fighter(50, 60, lethal_survivals=2, lethal_survival_hp_fraction=1)], steps=2)
        hits = [e for e in events if e['phase'] == 'DAMAGE_APPLY']
        self.assertEqual((hits[0]['damage'], hits[0]['hp']), (5000, 50))  # 5x first attack; revived at full HP.

    def test_execute_kills_below_threshold(self):
        events = self.events([Fighter(100, 30, execute_after_below_max_hp=.3)], [Fighter(100, 1)], steps=5)
        self.assertIn('EXECUTE', [e['phase'] for e in events])

    def test_randomized_batch_two_parity(self):
        rng = random.Random(4242)
        float_choices = {name: ((1, 1, .7, 1.5) if 'multiplier' in name else (0, 0, .25, .5)) for name in _BATCH2_FLOATS}
        cases = []
        for _ in range(64):
            teams = []
            for side in range(2):
                teams.append(tuple(Fighter(
                    hp=rng.randint(10, 40), attack=rng.randint(1, 12), dodge_probability=rng.choice((0, 0, .25)),
                    counter_on_damage=rng.choice((0, 1)), attacks_per_action=rng.choice((1, 1, 3)),
                    lethal_survivals=rng.choice((0, 0, 1)), block_mode=rng.randrange(3),
                    next_card_dodges=rng.choice((0, 1)), entry_hit_multiplier=rng.choice((0, 0, .5)),
                    **{name: rng.choice(values) for name, values in float_choices.items() if rng.random() < .25},
                    **{name: rng.choice((0, 1, 2)) for name in _BATCH2_INTS if rng.random() < .2},
                ) for _ in range(rng.randint(1, 4))))
            cases.append(Battle(tuple(teams), rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=120, max_frontier=64,
                                            repeat_cycles=6, prune_probability=1e-6, seed=9))


if __name__ == '__main__':
    unittest.main()
