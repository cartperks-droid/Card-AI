"""Support cards at every tier (IMG_0350-0354: the user's binder at base, Platinum, Crystal, Ruby and Galaxy)."""

import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import (BLUE_SUPPORTS, RED_SUPPORTS, UnsupportedCardError, compile_battle,
                                                 compile_fighter)
from card_engine.simulator.reference import Options, simulate


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'supports.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def battle(self, red=(1, 1), blue=(0, 0), teams=((210, 18), (21, 104))):
        return compile_battle(self.catalog, teams, red_supports=red, blue_supports=blue)

    def test_red_support_tiers(self):
        # Desmond Of Despair (Seven Sins) at Galaxy: +256% stats, +414% for Seven Sins cards (IMG_0354).
        sin, other = 205, 18
        base_sin, base_other = compile_fighter(self.catalog, sin), compile_fighter(self.catalog, other)
        battle = self.battle(red=((27, 5), 0), teams=((sin, other), (21,)))
        self.assertAlmostEqual(battle.teams[0][0].attack / base_sin.attack, 5.14)
        self.assertAlmostEqual(battle.teams[0][1].attack / base_other.attack, 3.56)
        hp_only = self.battle(red=((2, 4), 0), teams=((other,), (21,)))  # General Sun Tzu at Ruby: +25% HP
        self.assertEqual((hp_only.teams[0][0].hp / base_other.hp, hp_only.teams[0][0].attack), (1.25, base_other.attack))

    def test_user_supplied_tiers(self):
        self.assertEqual(BLUE_SUPPORTS[4][4], 20)  # Phantom Galaxy
        self.assertEqual(BLUE_SUPPORTS[12], [4, 3, 2, 4, 1])  # Magical Elf: one Toy fewer per tier, Ruby 4
        swordsman, other = self.battle(red=((14, 5), 0), teams=((10, 3), (21,))).teams[0]  # Arthur is a Swordsman
        self.assertAlmostEqual(swordsman.attack / compile_fighter(self.catalog, 10).attack, 3.34)
        self.assertAlmostEqual(other.attack / compile_fighter(self.catalog, 3).attack, 2.17)
        bear = self.battle(blue=(12, 0), teams=((261, 262, 263, 264), (21,))).teams[0][0]
        self.assertEqual((bear.entry_fallen_toy_stat_bonus, bear.entry_attack_multiplier), (1.0, 1))
        self.assertEqual(BLUE_SUPPORTS[1][3], BLUE_SUPPORTS[1][0])  # Ruby blue supports use their base values (user)

    def test_every_known_support_tier_compiles_and_engines_agree(self):
        options = Options(mode='sample', seed=3, max_steps=200)
        for sid, (_, _, values) in RED_SUPPORTS.items():
            for tier, value in enumerate(values, 1):
                if value is None or sid == 14:
                    continue
                with self.subTest(red=sid, tier=tier):
                    self.compare(self.battle(red=((sid, tier), 1)), options)
        for sid, values in BLUE_SUPPORTS.items():
            for tier, value in enumerate(values, 1):
                if value is None:
                    continue
                with self.subTest(blue=sid, tier=tier):
                    battle = self.battle(blue=((sid, tier), (sid, tier)))
                    self.compare(battle, options)
                    self.compare(battle, Options(mode='branch', max_steps=40, max_frontier=300, prune_probability=1e-4, rollouts=1))

    def test_blue_supports_set_their_mechanics(self):
        flame = self.battle(blue=((2, 5), 0)).teams[0][0]
        self.assertEqual(flame.hit_burn_chance, .5)
        storm = self.battle(blue=((11, 2), 0)).teams[0][0]
        self.assertEqual((storm.followup_multiplier, storm.followup_chance), (.5, .15))
        fate = self.battle(blue=((8, 2), 0), teams=((1,), (21,))).teams[0][0]  # Archer: 10% dodge; a failure is retried half the time
        self.assertAlmostEqual(fate.dodge_probability, .1 + .9 * .5 * .1)

    def test_magical_elf_awakens_toys(self):
        toys = (262, 263, 264, 18)
        plain = self.battle(blue=((12, 2), 0), teams=((262, 263, 18), (21,))).teams[0]
        self.assertEqual((plain[0].attacks_per_action, plain[0].awakened_toy), (3, 0))  # 2 unique Toys: no awakening
        car, jack, nutcracker, other = self.battle(blue=((12, 2), 0), teams=(toys, (21,))).teams[0]
        self.assertEqual((car.attacks_per_action, car.outgoing_multiplier), (2, 1))  # an attack per other Toy, full damage
        self.assertEqual((jack.enemy_entry_confusion_turns, jack.entry_confusion_turns), (3, 0))
        self.assertEqual((nutcracker.damage_cap_max_hp, nutcracker.damaged_toy_stat_gain), (.25, .1))
        self.assertEqual([f.awakened_toy for f in (car, jack, nutcracker, other)], [1, 1, 1, 0])
        self.assertFalse(any(f.awakened_toy for f in self.battle(blue=(12, 0), teams=(toys, (21,))).teams[0]))  # base: 4 needed

    def test_awakened_toys_and_end_times_engines_agree(self):
        cases = [((262, 263, 264, 18), (21, 104), ((12, 2), 0)), ((21, 104), (262, 263, 264), (0, (12, 2))),
                 ((210, 18), (1, 104, 21), ((14, 5), (8, 2))), ((1, 104), (264, 210), ((14, 3), (12, 2)))]
        for seed in range(4):
            for a, b, blue in cases:
                with self.subTest(seed=seed, teams=(a, b)):
                    self.compare(self.battle(blue=blue, teams=(a, b)), Options(mode='sample', seed=seed, max_steps=300))
        for a, b, blue in cases:
            with self.subTest(branch=(a, b)):
                self.compare(self.battle(blue=blue, teams=(a, b)),
                             Options(mode='branch', max_steps=30, max_frontier=400, prune_probability=1e-4, rollouts=1))

    def test_end_times_fails_abilities(self):
        # Archer's 10% dodge fails 25% of the time against Galaxy End Times.
        from card_engine.simulator.reference import _hit_chances, _initial
        battle = self.battle(blue=((14, 5), 0), teams=((21,), (1,)))
        dodge, _ = _hit_chances(battle, _initial(battle, Options(), None))
        self.assertAlmostEqual(dodge, battle.teams[1][0].dodge_probability * .75)

    def test_awakened_toy_bear(self):
        # User: 1,943 ATK -> 3,886 / 5,829 with 1 / 2 fallen Toys (Car and Jack die at once here).
        from dataclasses import replace as _replace
        from card_engine.simulator.reference import Battle
        for fallen, tier in ((1, 5), (2, 3)):
            battle = self.battle(blue=((12, tier), 0), teams=((262, 263)[:fallen] + (261,), (210,)))
            team = tuple(_replace(f, hp=1.0) if i < fallen else f for i, f in enumerate(battle.teams[0]))
            battle = Battle((team, battle.teams[1]), battle.first_side, battle.lineup, battle.pool)
            trace = simulate(battle, Options(mode='sample', seed=1, max_steps=400), trace=True).trace
            entry = next(e for e in trace if e.get('phase') == 'ENTRY' and e['side'] == 0 and e['slot'] == fallen)
            self.assertEqual(entry['attack'], 1943 * (1 + fallen))
            self.compare(battle, Options(mode='sample', seed=1, max_steps=400))

    def test_santa_and_developer_classes(self):
        self.assertEqual((compile_fighter(self.catalog, 260).hp, compile_fighter(self.catalog, 260).attack), (7772, 3886))
        demons = [c.id for c in self.catalog.cards if 'demon' in c.classes]
        self.assertTrue({11, 16, 33, 46, 60, 109, 110, 111, 120, 122, 211} <= set(demons))  # developer's list, added

    def compare(self, battle, options):
        keys = ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states')
        py, c = simulate(battle, options), native.simulate(battle, options, library_path=self.library)
        for key in keys:
            self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=key)


if __name__ == '__main__':
    unittest.main()
