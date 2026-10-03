"""Counter search: the stress test behind its ranking."""

import unittest

from card_engine.catalog import load_catalog
from card_engine.training.counter import _side, _spec, _stressed
from card_engine.training.labels import compile_spec


class StressTests(unittest.TestCase):
    def test_stress_scales_the_enemy_and_strips_one_ability(self):
        catalog = load_catalog()
        player = _side([91, 4, 1, 164], [12, 1, 1, 1], ["None"] * 4, (0, 0), (0, 0))  # Vampire Lord, Useless Seer, Archer, True Prophet
        enemy = _side([136, 167, 79, 36], [1] * 4, ["None"] * 4, (0, 0), (0, 0))
        battle = compile_spec(catalog, _spec(player, enemy))
        stressed = _stressed(battle, 1, 2.0, 0)
        for before, after in zip(battle.teams[1], stressed.teams[1]):
            self.assertEqual((after.hp, after.attack), (before.hp * 2, before.attack * 2))
        lord, plain = battle.teams[0][0], stressed.teams[0][0]
        self.assertEqual((plain.hp, plain.attack, plain.spare), (lord.hp, lord.attack, 0))
        self.assertTrue(lord.hit_hp_gain_fraction and not plain.hit_hp_gain_fraction)  # the steal is gone
        self.assertEqual(stressed.teams[0][1:4], battle.teams[0][1:4])  # the other cards are untouched
        self.assertIs(_stressed(battle, 1, 1, None).teams[1], battle.teams[1])


if __name__ == "__main__":
    unittest.main()
