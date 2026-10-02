"""Poison (user video IMG_0297 and the Black Plague test): each tick takes the poisoner's
ATK x effectiveness directly, ignoring damage reduction, at every turn end, waiting
cards included. Dilophosaurus swaps with the next ally after its lethal dodge."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch


class PoisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'poison.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states', 'estimated'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def test_img_0297_dilophosaurus_vs_heavens_armor(self):
        battle = compile_battle(load_catalog(), ((146,), (38,)), red_supports=(0, 0))
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        ticks = [e for e in events if e['phase'] == 'STATUS_TICK']
        # Ticks come at the start of the poisoned card's turn (IMG_0297/0300/0305): one tick, then
        # Heaven's Armor attacks, then Dilophosaurus's hit finishes it (the flash at 3.7 s).
        self.assertEqual([e['damage'] for e in ticks], [1559])  # Full ATK, ignoring the 75% reduction.
        self.assertEqual(ticks[0]['hp'], 221)  # 8.6% of 2,560, as the video bar shows.
        hit = next(e for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 0)
        self.assertEqual(hit['hp'], 3117 - 1280)  # 58.9%; the video bar shows 58.4%.
        self.assertEqual([e['reason'] for e in events if e['phase'] == 'DEATH'], ['damage'])
        self.compare([battle])

    def test_black_plague_wipes_platinum_jamiys_after_two_turns(self):
        battle = compile_battle(load_catalog(), ((52,), (39, 39, 39, 39)), borders=((1,), (2, 2, 2, 2)), red_supports=(0, 0))
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        ticks = [e for e in events if e['phase'] == 'STATUS_TICK']
        self.assertTrue(ticks and all(e['damage'] == 1920 for e in ticks))  # 50% of 3,840 ATK.
        self.assertEqual(simulate(battle).p_a, 1)
        self.assertEqual(sum(e['phase'] == 'DEATH' and e['side'] == 1 for e in events), 4)
        # Every Jamiy that is not killed first dies to the second tick, waiting ones included.
        self.assertTrue(any(e['phase'] == 'DEATH' and e['reason'] == 'bench' for e in events))
        self.compare([battle])

    def test_dilophosaurus_swaps_after_its_lethal_dodge(self):
        dilo = Fighter(100, 10, own_lethal_dodges=1, lethal_dodge_swap=1)
        battle = Battle(((dilo, Fighter(500, 10)), (Fighter(1000, 200),)), 1)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        self.assertIn('LETHAL_DODGE_SWAP', [e['phase'] for e in events])
        entries = [e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0]
        self.assertEqual(entries[1]['hp'], 500)  # The ally comes in next.
        self.compare([battle])

    def test_catalog_and_randomized_parity(self):
        catalog = load_catalog()
        rng = random.Random(3)
        pool = (52, 146, 237, 10, 22, 38, 93, 104, 114, 164, 172, 210)
        battles = [compile_battle(catalog, (tuple(rng.sample(pool, 3)), tuple(rng.sample(pool, 3))), red_supports=(0, 0),
                                  first_side=rng.randrange(2)) for _ in range(12)]
        cases = []
        for _ in range(40):
            teams = tuple(tuple(Fighter(
                hp=rng.randint(10, 80), attack=rng.randint(1, 15), dodge_probability=rng.choice((0, 0, .5)),
                poison_fraction=rng.choice((1, .5)), entry_poison_turns=rng.choice((0, 0, 2)), entry_poison_all=rng.choice((0, 1)),
                hit_poison_turns=rng.choice((0, 0, 2)), attacked_poison_turns=rng.choice((0, 0, 2)),
                own_lethal_dodges=rng.choice((0, 0, 1)), lethal_dodge_swap=rng.choice((0, 1)),
                entry_hit_multiplier=rng.choice((0, .25, 1)), entry_hit_all_enemies=rng.choice((0, 0, 1)),
                lethal_survivals=rng.choice((0, 0, 1)), next_card_dodges=rng.choice((0, 0, 1)),
            ) for _ in range(rng.randint(1, 4))) for _ in range(2))
            cases.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(battles + cases, Options(mode=mode, max_steps=150, max_frontier=64, repeat_cycles=6,
                                                      prune_probability=1e-6, seed=9))


if __name__ == '__main__':
    unittest.main()
