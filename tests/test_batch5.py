"""Batch-5 extrapolated primitives: card-text readings pinned for semantics and
Python/C parity, not game observations. Classes are inferred from card art."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import (Battle, Fighter, Options, simulate, simulate_batch,
                                             _BATCH5_FLOATS, _BATCH5_INTS, _BATCH5_CHANCES)

BATCH5_CARDS = (8, 12, 20, 40, 41, 45, 51, 55, 68, 84, 124, 128, 132, 136, 142, 145, 150, 168, 170, 181, 187, 191,
                197, 202, 206, 208, 217, 223, 227, 231, 235, 258, 259, 279, 280, 285)


class BatchFiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'batch5.dylib')

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
        battles = [compile_battle(catalog, ((card, 7, 3), (opponent, 3)), red_supports=(0, 0), first_side=side)
                   for card in BATCH5_CARDS for opponent in (73, 210, 94, 213) for side in (0, 1)]
        self.compare(battles, Options(max_frontier=300, prune_probability=1e-9))

    def test_fade_removes_bonuses_below_threshold(self):
        events = self.events([Fighter(100, 10, fade_threshold=.5, fade_outgoing_multiplier=3)], [Fighter(1000, 30)], steps=6)
        damage = [e['damage'] for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1]
        self.assertEqual(damage[:3], [30, 30, 10])  # Fades after dropping to 40 HP.

    def test_alternate_rest_is_untouchable(self):
        events = self.events([Fighter(100, 5, alternate_rest=1)], [Fighter(1000, 7)], steps=8)
        own = [e['phase'] for e in events if e.get('side') == 0 and e['phase'] in ('ATTACK_DECLARE', 'SKIP')]
        self.assertEqual(own[:4], ['ATTACK_DECLARE', 'SKIP', 'ATTACK_DECLARE', 'SKIP'])
        self.assertIn('invincible', [e.get('outcome') for e in events])

    def test_kill_extra_action_attacks_the_replacement_immediately(self):
        events = self.events([Fighter(100, 50, kill_extra_action=1)], [Fighter(10, 1), Fighter(200, 1)], steps=3)
        self.assertEqual([e['side'] for e in events if e['phase'] == 'ATTACK_DECLARE'][:2], [0, 0])

    def test_class_matchups_use_inferred_classes(self):
        catalog = load_catalog()
        siegfried = compile_battle(catalog, ((55,), (210,)), red_supports=(0, 0))  # Raze is inferred a dragon.
        events = simulate(siegfried, Options(mode='sample', max_steps=1), trace=True).trace
        entry = next(e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0)
        hit = next(e for e in events if e['phase'] == 'DAMAGE_APPLY')
        self.assertEqual(hit['damage'], -(-entry['attack'] * 2 // 1))
        self.compare([siegfried])

    def test_randomized_batch_five_parity(self):
        rng = random.Random(55)
        cases = []
        for _ in range(48):
            teams = []
            for side in range(2):
                floats = {}
                for name in _BATCH5_FLOATS:
                    if rng.random() < .2:
                        floats[name] = (rng.choice((.3, .5, 1.0)) if name in _BATCH5_CHANCES else
                                        rng.choice((.5, 1.5)) if 'multiplier' in name or name == 'border_rarity' else
                                        rng.choice((.25, .5)) if name != 'card_rarity' else rng.choice((10., 1000.)))
                ints = {name: rng.choice((1, 2, 3)) for name in _BATCH5_INTS if rng.random() < .15}
                teams.append(tuple(Fighter(
                    hp=rng.randint(10, 60), attack=rng.randint(1, 12), dodge_probability=rng.choice((0, 0, .25)),
                    counter_on_damage=rng.choice((0, 1)), attacks_per_action=rng.choice((1, 1, 2)),
                    block_mode=rng.choice((0, 1)), lethal_survivals=rng.choice((0, 0, 1)),
                    **floats, **ints) for _ in range(rng.randint(1, 3))))
            cases.append(Battle(tuple(teams), rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=80, max_frontier=64,
                                            repeat_cycles=6, prune_probability=1e-6, seed=23))


if __name__ == '__main__':
    unittest.main()
