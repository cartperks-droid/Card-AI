"""Ability removal, theft and copying (user battles 5-9: IMG_0316, 0317, 0319, 0324, 0325).
A card always keeps its own stats and class; only its ability is removed, stolen,
copied or cancelled."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch

RAZE = Fighter(1000, 50, counter_on_damage=1)


class AbilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'abilities.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states', 'estimated'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def counters(self, battle, steps=40):
        events = simulate(battle, Options(mode='sample', max_steps=steps), trace=True).trace
        return [e['side'] for e in events if e['phase'] == 'COUNTER_QUEUED']

    def test_removal_and_cancellation_stop_raze_countering(self):
        self.assertEqual(self.counters(Battle(((RAZE,), (Fighter(500, 10, entry_disable_enemy=1),)), 1)), [])  # Hell's Army.
        self.assertEqual(self.counters(Battle(((RAZE,), (Fighter(500, 10, hit_disable=1),)), 1)), [0])  # Set: after its hit.
        self.assertEqual(self.counters(Battle(((RAZE,), (Fighter(500, 10, cancels_all_abilities=1),)), 1)), [])  # Samurai.
        fuxi = Battle(((RAZE,), (Fighter(10**6, 10, entry_suppress_enemy_turns=3),)), 1)
        events = simulate(fuxi, Options(mode='sample', max_steps=12), trace=True).trace
        first = next(k for k, e in enumerate(events) if e['phase'] == 'COUNTER_QUEUED')
        self.assertGreaterEqual(sum(e['phase'] == 'ATTACK_DECLARE' for e in events[:first]), 3)  # Suppressed for 3 turns.
        self.compare([Battle(((RAZE,), (Fighter(500, 10, hit_disable=1),)), 1), fuxi], Options(max_steps=40))

    def test_theft_and_copies(self):
        hecate = Battle(((RAZE,), (Fighter(500, 10, steal_first_attacked=1),)), 1)
        self.assertEqual(set(self.counters(hecate)[1:]), {1})  # Only Hecate counters after the steal.
        sable = Battle(((RAZE,), (Fighter(500, 10, copies_enemy_in_play=1),)), 0)
        self.assertEqual(set(self.counters(sable)), {1})  # User: Sable takes Raze's counter and disables it on Raze.
        hades = Battle(((Fighter(10**6, 300),), (RAZE, Fighter(500, 10, copy_fallen_ally=1))), 0)
        events = simulate(hades, Options(mode='sample'), trace=True).trace
        death = next(k for k, e in enumerate(events) if e['phase'] == 'DEATH')
        self.assertIn('COUNTER_QUEUED', [e['phase'] for e in events[death:]])  # Hades counters like the fallen Raze.
        self.compare([hecate, sable, hades], Options(max_steps=60))

    def test_mother_of_beasts_img_0317(self):
        battle = compile_battle(load_catalog(), ((114,), (212,)), borders=((4,), (2,)), red_supports=(0, 0), first_side=1)
        events = simulate(battle, Options(mode='sample', max_steps=6), trace=True).trace
        from card_engine.simulator.reference import _advance, _initial
        state = _initial(battle, Options(), None)
        attacks = [state.attack[0][0]]
        for _ in range(4):
            state, _ = _advance(battle, state, False, Options(), None)
            attacks.append(state.attack[0][0])
        self.assertEqual(sorted(set(round(a) for a in attacks), reverse=True)[:3], [20480, 17408, 14797])
        self.assertTrue(events)
        self.compare([battle], Options(max_steps=40))

    def test_black_cat_and_ra(self):
        catalog = load_catalog()
        cat = compile_battle(catalog, ((50,), (69,)), borders=((1,), (3,)), red_supports=(0, 0), first_side=1)
        hits = [e['damage'] for e in simulate(cat, Options(mode='sample'), trace=True).trace if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 0]
        self.assertEqual(hits[0], 2 * 2266)  # IMG_0324: Judgement Day died to one 2x hit.
        ra = compile_battle(catalog, ((65, 10), (79,)), red_supports=(0, 0), first_side=0)
        deaths = [(e['side'], e['slot'], e['reason']) for e in simulate(ra, Options(mode='sample'), trace=True).trace if e['phase'] == 'DEATH']
        self.assertEqual(deaths[0], (0, 1, 'bench'))  # IMG_0325: Arthur died waiting behind Parallax.
        self.compare([cat, ra])

    def test_randomized_parity(self):
        rng = random.Random(57)
        catalog = load_catalog()
        pool = (33, 75, 106, 137, 284, 212, 49, 247, 69, 276, 211, 79, 10, 93, 114, 210, 50, 65)
        battles = [compile_battle(catalog, (tuple(rng.sample(pool, 3)), tuple(rng.sample(pool, 3))), red_supports=(0, 0),
                                  first_side=rng.randrange(2)) for _ in range(14)]
        cases = []
        for _ in range(40):
            teams = tuple(tuple(Fighter(
                hp=rng.randint(10, 80), attack=rng.randint(1, 15), dodge_probability=rng.choice((0, 0, .5)),
                counter_on_damage=rng.choice((0, 1)), class_mask=rng.choice((0, 16)), lethal_survivals=rng.choice((0, 1)),
                entry_disable_enemy=rng.choice((0, 0, 1)), hit_disable=rng.choice((0, 0, 1)), cancels_all_abilities=rng.choice((0, 0, 0, 1)),
                entry_suppress_enemy_turns=rng.choice((0, 0, 3)), steal_first_attacked=rng.choice((0, 0, 1)),
                turn_start_steal_fraction=rng.choice((0, 0, .15)), kill_steal_ability=rng.choice((0, 1)),
                copy_fallen_ally=rng.choice((0, 0, 1)), disable_enemy_class_mask=rng.choice((0, 0, 16)),
                entry_disable_enemy_class_mask=rng.choice((0, 16)), copies_enemy_in_play=rng.choice((0, 0, 1)),
                entry_hit_multiplier=rng.choice((0, 1)), entry_hit_next_card=rng.choice((0, 1)),
            ) for _ in range(rng.randint(1, 4))) for _ in range(2))
            cases.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(battles + cases, Options(mode=mode, max_steps=150, max_frontier=64, repeat_cycles=6,
                                                      prune_probability=1e-6, seed=23))


if __name__ == '__main__':
    unittest.main()
