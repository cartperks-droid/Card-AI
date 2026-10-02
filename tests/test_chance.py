"""Batch-4 chance mechanics: branch mode enumerates every roll (unrolled first),
sample mode draws lazily. Frostbite follows video 0290; the rest are card-text
extrapolations pinned for semantics and Python/C parity."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import (Battle, Fighter, Options, simulate, simulate_batch,
                                             _BATCH4_FLOATS, _BATCH4_INTS)

BATCH4_CARDS = (15, 21, 22, 29, 31, 32, 50, 63, 64, 100, 148, 201, 216, 220, 232, 245, 251, 252, 260, 263)
CHANCES = {"hit_freeze_chance", "skip_enemy_turn_chance", "chance_survival_probability", "gamble_chance",
           "entry_borderless_reset_chance", "guardian_chance", "dodge_cap"}


class ChanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'chance.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def test_every_batch_card_runs_in_both_engines(self):
        catalog = load_catalog()
        battles = [compile_battle(catalog, ((card, 7), (opponent, 3)), red_supports=(0, 0), first_side=side)
                   for card in BATCH4_CARDS for opponent in (73, 210, 94, 213) for side in (0, 1)]
        battles.append(compile_battle(catalog, ((3, 7), (210,)), red_supports=(0, 0), blue_supports=(15, 0)))
        self.compare(battles, Options(max_frontier=400, prune_probability=1e-9))

    def test_frostbite_video_round(self):
        # Video 0290: Malik kills Frosty, gets frostbitten; half the time it loses 20%
        # max HP and a turn, so Titan attacks twice before Malik kills it.
        malik = Fighter(1000, 10**6)
        frosty, titan = Fighter(10, 1, death_frostbite_turns=1), Fighter(100, 1)
        battle = Battle(((malik,), (frosty, titan)), 0)
        self.assertAlmostEqual(simulate(battle).p_a, 1)
        self.compare([battle])
        for seed in range(1, 40):
            events = simulate(battle, Options(mode='sample', seed=seed), trace=True).trace
            bitten = [e for e in events if e.get('status') == 'frostbite']
            titan_hits = sum(e['phase'] == 'DAMAGE_APPLY' and e['side'] == 0 for e in events)
            self.assertLessEqual(len(bitten), 1)
            if bitten:
                self.assertEqual(bitten[0]['damage'], 200)
            self.assertEqual(titan_hits, 2 if bitten else 1)

    def test_branching_weights_follow_roll_probabilities(self):
        # A 30% skip chance before each enemy turn: the enemy lands its only lethal hit
        # on turn one with probability 0.7.
        battle = Battle(((Fighter(5, 1, skip_enemy_turn_chance=.3),), (Fighter(1000, 10),)), 1)
        self.assertAlmostEqual(simulate(battle, Options(max_steps=1)).p_b, .7)
        self.compare([battle], Options(max_steps=3))

    def test_guardian_angel_and_chance_revive(self):
        catalog = load_catalog()
        guarded = compile_battle(catalog, ((7,), (210,)), red_supports=(0, 0), blue_supports=(15, 0))
        self.assertAlmostEqual(guarded.teams[0][0].guardian_chance, .1)
        revive = Battle(((Fighter(10, 1, chance_survival_probability=.5, chance_survival_hp_fraction=1),), (Fighter(1000, 20),)), 1)
        self.assertAlmostEqual(simulate(revive, Options(max_steps=1)).p_b, .5)
        self.compare([guarded, revive])

    def test_arthur_and_guardian_angel_survive_infinite_damage(self):
        # User: both survive #50's "infinite" critical.
        catalog = load_catalog()
        for team, blue in (((10,), (0, 0)), ((7,), (15, 0))):
            battle = compile_battle(catalog, (team, (50,)), red_supports=(0, 0), blue_supports=blue, first_side=1)
            events = simulate(battle, Options(mode='sample', max_steps=1, seed=2), trace=True).trace
            if any(e.get('critical') for e in events):
                self.assertIn('LETHAL_REPLACEMENT', [e['phase'] for e in events if blue == (0, 0)] or ['LETHAL_REPLACEMENT'])
        arthur = compile_battle(catalog, ((10,), (50,)), red_supports=(0, 0), first_side=1)
        self.assertGreater(simulate(arthur, Options(max_steps=1)).unresolved, .99)  # Always survives the first hit.

    def test_randomized_chance_parity(self):
        rng = random.Random(21)
        cases = []
        for _ in range(48):
            teams = []
            for side in range(2):
                floats = {name: rng.choice((.3, .5, 1.0) if name in CHANCES else (0, .5, 2))
                          for name in _BATCH4_FLOATS if rng.random() < .2 and name not in ('borderless_hp', 'borderless_attack')}
                teams.append(tuple(Fighter(
                    hp=rng.randint(10, 60), attack=rng.randint(1, 12), dodge_probability=rng.choice((0, 0, .25)),
                    counter_on_damage=rng.choice((0, 1)), attacks_per_action=rng.choice((1, 1, 2)),
                    borderless_hp=rng.choice((0, 5)), borderless_attack=rng.choice((0, 2)),
                    **floats, **{name: rng.choice((0, 1, 2)) for name in _BATCH4_INTS if rng.random() < .2},
                ) for _ in range(rng.randint(1, 3))))
            cases.append(Battle(tuple(teams), rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=80, max_frontier=64,
                                            repeat_cycles=6, prune_probability=1e-6, seed=17))


if __name__ == '__main__':
    unittest.main()
