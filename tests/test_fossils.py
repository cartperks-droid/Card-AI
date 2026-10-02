"""Fossils (user, IMG_0314/0315): a team counter; every dying dinosaur adds one, Ancient Egg
and Meteosaurus add more; Tyrannodon, T-Rex, Velociraptor and Cyberdon scale with it."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch


class FossilTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'fossils.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states', 'estimated'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def test_img_0314_fossil_counts_and_tyrannodon(self):
        battle = compile_battle(load_catalog(), ((210, 65, 38), (141, 151, 159, 157)), borders=((2, 1, 1), (1, 1, 1, 1)),
                                red_supports=(0, 0))
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        self.assertEqual([e['count'] for e in events if e['phase'] == 'FOSSILS'], [4, 7, 8, 9])  # User's tally.
        tyrannodon = [e for e in events if e['phase'] == 'ENTRY' and e['side'] == 1][2]
        self.assertEqual(-(-tyrannodon['attack'] // 1), 148272)  # Displayed in IMG_0314.
        self.compare([battle])

    def test_velociraptor_cyberdon_and_t_rex(self):
        raptor = Fighter(10**6, 1, fossil_death_timer=5)
        events = simulate(Battle(((raptor,), (Fighter(10**6, 1),)), 0), Options(mode='sample', max_steps=30), trace=True).trace
        self.assertIn('DOOM', [e['phase'] for e in events])  # 5 turns with no fossils.
        cyberdon = Fighter(10**6, 100, fossil_deck_hits=3, entry_self_slow_turns=1, action_end_self_slow_turns=1)
        team = tuple(Fighter(10**6, 1) for _ in range(4))
        events = simulate(Battle(((cyberdon,), team), 0), Options(mode='sample', max_steps=6), trace=True).trace
        self.assertEqual(events[[e['phase'] for e in events].index('SKIP')]['side'], 0)  # Charges first.
        bench = [e['slot'] for e in events if e['phase'] == 'BENCH_HIT']
        self.assertEqual(bench[:1], [1])  # No fossils: only one waiting card is hit.
        t_rex = Fighter(1000, 10, counter_on_damage=1, counter_min_fossils=3)
        events = simulate(Battle(((Fighter(1000, 10),), (t_rex,)), 0), Options(mode='sample', max_steps=4), trace=True).trace
        self.assertNotIn('COUNTER_QUEUED', [e['phase'] for e in events])  # Needs 3 fossils.
        self.compare([Battle(((raptor, Fighter(50, 5)), (Fighter(300, 40),)), 0), Battle(((cyberdon,), team), 0)],
                     Options(max_steps=60))

    def test_randomized_parity(self):
        rng = random.Random(41)
        catalog = load_catalog()
        pool = (141, 151, 154, 155, 157, 159, 148, 146, 10, 38, 65, 210)
        battles = [compile_battle(catalog, (tuple(rng.sample(pool, 3)), tuple(rng.sample(pool, 3))), red_supports=(0, 0),
                                  first_side=rng.randrange(2)) for _ in range(12)]
        cases = []
        for _ in range(40):
            teams = tuple(tuple(Fighter(
                hp=rng.randint(10, 80), attack=rng.randint(1, 15), dodge_probability=rng.choice((0, 0, .5)),
                dinosaur=rng.choice((0, 1)), death_fossils=rng.choice((0, 0, 2, 3)), entry_fossil_attack_multiplier=rng.choice((1, 1.5)),
                counter_on_damage=rng.choice((0, 1)), counter_min_fossils=rng.choice((0, 3)), fossil_death_timer=rng.choice((0, 0, 5)),
                fossil_deck_hits=rng.choice((0, 0, 3)), entry_self_slow_turns=rng.choice((0, 1)), lifetime_actions=rng.choice((0, 0, 1)),
            ) for _ in range(rng.randint(1, 4))) for _ in range(2))
            cases.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(battles + cases, Options(mode=mode, max_steps=150, max_frontier=64, repeat_cycles=6,
                                                      prune_probability=1e-6, seed=17))


if __name__ == '__main__':
    unittest.main()
