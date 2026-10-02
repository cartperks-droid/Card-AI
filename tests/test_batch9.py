"""Batch 9: Ghoul, Zombie Nurse, Zombie Dragon, Kira, Onmyoji, Rudolph (blind),
Banshee (stun), Gambler, Loch Ness and Steve (user videos IMG_0299-0304 and answers)."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch


class BatchNineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'batch9.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states', 'estimated'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def trace(self, battle, seed=1, steps=400):
        return simulate(battle, Options(mode='sample', seed=seed, max_steps=steps), trace=True).trace

    def test_rudolph_blinds_raze_forever(self):  # IMG_0299
        battle = compile_battle(load_catalog(), ((210,), (253,)), borders=((2,), (1,)), red_supports=(0, 0))
        events = self.trace(battle, steps=60)
        self.assertFalse(any(e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1 for e in events))
        self.assertIn('blind_miss', [e.get('outcome') for e in events])
        self.compare([battle], Options(max_steps=60))

    def test_loch_ness_grows_ten_percent_per_allied_turn_while_waiting(self):  # IMG_0303
        loch = Fighter(1000, 14126, bench_growth_multiplier=1.1, bench_growth_cap=3)
        battle = Battle(((Fighter(10**7, 1), loch), (Fighter(10**7, 1),)), 0)
        from card_engine.simulator.reference import _initial, _advance
        state_attacks = []
        state = _initial(battle, Options(), None)
        for _ in range(8):
            state, _ = _advance(battle, state, False, Options(), None)
            state_attacks.append(state.attack[0][1])
        self.assertAlmostEqual(max(state_attacks) / 14126, 1.1 ** 4, places=6)
        self.compare([battle], Options(max_steps=30))

    def test_gambler_draws(self):
        battle = Battle(((Fighter(10**6, 100, stat_gamble=1),), (Fighter(10**6, 1),)), 0)
        seen = set()
        for seed in range(1, 25):
            seen.update(e['factor'] for e in self.trace(battle, seed=seed, steps=2) if e['phase'] == 'STAT_GAMBLE')
        self.assertEqual(seen, {0.9, 1.15, 1.3})
        self.compare([Battle(((Fighter(500, 100, stat_gamble=1),), (Fighter(900, 60),)), 0)])

    def test_kira_and_onmyoji_count_allied_attacks_even_after_death(self):
        # User: both count the normal attacks of the ally in play, just before each attack, and hit the enemy in play.
        kira = Fighter(10, 1, encounter_doom_turns=2)
        battle = Battle(((kira, Fighter(10**6, 1)), (Fighter(10**6, 100),)), 1)
        events = self.trace(battle)
        deaths = [(e['side'], e['reason']) for e in events if e['phase'] == 'DEATH']
        self.assertEqual(deaths[:2], [(0, 'damage'), (1, 'damage')])
        declares = [e for e in events if e['phase'] == 'ATTACK_DECLARE' and e['side'] == 0]
        self.assertEqual(len(declares), 1)  # The second allied attack is pre-empted by the doom.
        self.assertIn('DOOM', [e['phase'] for e in events])
        onmyoji = Fighter(10, 50, delayed_strike_turns=5, delayed_strike_multiplier=3)
        battle = Battle(((onmyoji, Fighter(10**6, 1)), (Fighter(10**6, 100),)), 1)
        events = self.trace(battle)
        strike = next(k for k, e in enumerate(events) if e['phase'] == 'DELAYED_STRIKE')
        self.assertEqual(events[strike]['damage'], 150)
        self.assertEqual(sum(e['phase'] == 'ATTACK_DECLARE' and e['side'] == 0 for e in events[:strike]), 4)  # Before the 5th.
        self.compare([Battle(((kira, Fighter(500, 5)), (Fighter(300, 100), Fighter(300, 100))), 1), battle])

    def test_zombie_dragon_nurse_ghoul_banshee_steve(self):
        dragon = Fighter(100, 10, zero_hp_survival_turns=2, hit_poison_turns=2, poison_target_max_hp_fraction=.15)
        events = self.trace(Battle(((dragon, Fighter(10**6, 1)), (Fighter(1000, 500),)), 1))
        self.assertIn('zero_hp_survival', [e.get('effect') for e in events])
        self.assertEqual([e['damage'] for e in events if e['phase'] == 'STATUS_TICK'][0], 150)
        nurse = Fighter(100, 1, low_hp_heal_fraction=.5, low_hp_trigger_threshold=.5)
        heals = [e for e in self.trace(Battle(((nurse,), (Fighter(1000, 30),)), 1)) if e['phase'] == 'LOW_HP_HEAL']
        self.assertEqual(len(heals), 1)
        ghoul = Battle(((Fighter(1000, 10, hit_weakness_multiplier=1.3),), (Fighter(1000, 1),)), 0)
        hits = [e['damage'] for e in self.trace(ghoul, steps=6) if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1]
        self.assertEqual(hits[:2], [10, 13])
        banshee = Battle(((Fighter(100, 1, low_hp_stun_turns=1, low_hp_trigger_threshold=.5, low_hp_heal_fraction=.5),),
                          (Fighter(1000, 30),)), 1)
        self.assertEqual([e['phase'] for e in self.trace(banshee)].count('SKIP'), 1)  # User: stuns only once.
        steve = Battle(((Fighter(1000, 10, block_max_hp_fraction=.4, block_max=3),), (Fighter(10**6, 300),)), 0)
        absorbed = [e['absorbed'] for e in self.trace(steve, steps=4) if e['phase'] == 'BLOCK_ABSORB']
        self.assertEqual(absorbed[0], 300)
        self.compare([ghoul, banshee, steve, Battle(((dragon,), (Fighter(1000, 500),)), 1)])

    def test_catalog_and_randomized_parity(self):
        catalog = load_catalog()
        rng = random.Random(21)
        pool = (274, 275, 178, 103, 107, 253, 267, 83, 196, 183, 10, 38, 114, 146, 210)
        battles = [compile_battle(catalog, (tuple(rng.sample(pool, 3)), tuple(rng.sample(pool, 3))), red_supports=(0, 0),
                                  first_side=rng.randrange(2)) for _ in range(14)]
        cases = []
        for _ in range(40):
            teams = tuple(tuple(Fighter(
                hp=rng.randint(10, 80), attack=rng.randint(1, 15), dodge_probability=rng.choice((0, 0, .5)),
                hit_weakness_multiplier=rng.choice((1, 1, 1.3)), low_hp_heal_fraction=rng.choice((0, .5)),
                low_hp_trigger_threshold=rng.choice((0, .5)), low_hp_stun_turns=rng.choice((0, 1)),
                zero_hp_survival_turns=rng.choice((0, 0, 2)), hit_poison_turns=rng.choice((0, 2)),
                poison_target_max_hp_fraction=rng.choice((0, .15)), encounter_doom_turns=rng.choice((0, 0, 2)),
                delayed_strike_turns=rng.choice((0, 0, 3)), delayed_strike_multiplier=3, entry_blind=rng.choice((0, 0, 1)),
                stat_gamble=rng.choice((0, 1)), bench_growth_multiplier=rng.choice((1, 1.1)), bench_growth_cap=3,
                block_max_hp_fraction=rng.choice((0, .4)), block_max=3, entry_hit_all_enemies=rng.choice((0, 1)),
                entry_hit_multiplier=rng.choice((0, .25)),
            ) for _ in range(rng.randint(1, 4))) for _ in range(2))
            cases.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(battles + cases, Options(mode=mode, max_steps=150, max_frontier=64, repeat_cycles=6,
                                                      prune_probability=1e-6, seed=13))


if __name__ == '__main__':
    unittest.main()
