"""Batch-3 statuses. User-observed: statuses tick every turn (both sides), refresh
on re-application, stack across kinds, and freeze blocks counters (Ice Queen vs
Raze). Tick sizes are fits to the user's 100 Men tests; parity is not evidence."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch, _BATCH3_INTS

BATCH3_CARDS = (57, 81, 92, 135, 147, 153, 167, 198, 257)


class StatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'statuses.dylib')

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
        battles = [compile_battle(catalog, ((card, 7), (opponent, 3)), red_supports=(0, 0), first_side=side)
                   for card in BATCH3_CARDS for opponent in (73, 210, 94, 213) for side in (0, 1)]
        self.compare(battles, Options(max_frontier=200, prune_probability=1e-9))

    def test_bleed_ticks_every_turn_and_refreshes(self):
        # Ankylosaurus-style: attacking it bleeds the attacker, ticking on both sides' turns.
        events = self.events([Fighter(1000, 1)], [Fighter(10**6, 0, bleed_on_attacked_turns=3)], steps=6)
        ticks = [e for e in events if e['phase'] == 'STATUS_TICK']
        self.assertEqual([e['damage'] for e in ticks], [160] * len(ticks))
        self.assertEqual(len(ticks), 6)  # One per turn end, refreshed by every attack.

    def test_freeze_blocks_counters_but_expires_before_the_next_turn(self):
        # User: Ice Queen froze Raze on hit; Raze's counter was frozen, its next turn was not.
        freezer = Fighter(100, 1, entry_freeze_turns=1)
        events = self.events([freezer], [Fighter(100, 5, counter_on_damage=1)], first_side=0, steps=3)
        phases = [e['phase'] for e in events]
        self.assertNotIn('COUNTER_DECLARE', phases)
        self.assertEqual([e['side'] for e in events if e['phase'] == 'ATTACK_DECLARE'][:2], [0, 1])

    def test_slow_skips_the_next_own_turn(self):
        events = self.events([Fighter(100, 1, entry_slow_turns=1)], [Fighter(100, 5)], first_side=1, steps=4)
        self.assertEqual([e['phase'] for e in events if e.get('side') == 1 and e['phase'] in ('SKIP', 'ATTACK_DECLARE')][:2],
                         ['SKIP', 'ATTACK_DECLARE'])

    def test_status_ticks_can_kill(self):
        events = self.events([Fighter(10, 1, hit_burn_turns=5)], [Fighter(20, 0)], steps=12)
        self.assertIn('status', [e.get('reason') for e in events if e['phase'] == 'DEATH'])

    def test_randomized_status_parity(self):
        rng = random.Random(3)
        cases = []
        for _ in range(64):
            teams = []
            for side in range(2):
                teams.append(tuple(Fighter(
                    hp=rng.randint(10, 60), attack=rng.randint(1, 12), dodge_probability=rng.choice((0, 0, .25)),
                    counter_on_damage=rng.choice((0, 1)), attacks_per_action=rng.choice((1, 1, 2)),
                    lethal_survivals=rng.choice((0, 0, 1)), survival_attack_multiplier=rng.choice((1, 2)),
                    actions_per_turn=rng.choice((1, 1, 2)), entry_hit_multiplier=rng.choice((0, 0, .5)),
                    **{name: rng.choice((0, 1, 3)) for name in _BATCH3_INTS if rng.random() < .35},
                ) for _ in range(rng.randint(1, 4))))
            cases.append(Battle(tuple(teams), rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=120, max_frontier=64,
                                            repeat_cycles=6, prune_probability=1e-6, seed=13))


if __name__ == '__main__':
    unittest.main()
