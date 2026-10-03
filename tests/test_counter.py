"""Counter search: the stress test behind its ranking."""

import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import drago
from card_engine.training.counter import _side, _spec


class StressTests(unittest.TestCase):
    def test_stress_scales_the_enemy_and_strips_one_ability(self):
        catalog = load_catalog()
        player = _side([91, 4, 1, 164], [12, 1, 1, 1], ["None"] * 4, (0, 0), (0, 0))  # Vampire Lord, Useless Seer, Archer, True Prophet
        enemy = _side([136, 167, 79, 36], [1] * 4, ["None"] * 4, (0, 0), (0, 0))
        spec = _spec(player, enemy)
        battle = drago.initial_cards(catalog, spec)
        stressed = drago.initial_cards(catalog, spec, scale=(1, 2.0), strip=(0, 0))
        for before, after in zip(battle[1], stressed[1]):
            self.assertAlmostEqual(after[0], before[0] * 2)
            self.assertAlmostEqual(after[1], before[1] * 2)
        lord, plain = battle[0][0], stressed[0][0]
        self.assertEqual((plain[:2], plain[2]), (lord[:2], None))
        self.assertTrue(lord[2])  # Vampire Lord had an ability
        self.assertEqual(stressed[0][1:], battle[0][1:])  # the other cards are untouched
        fixed = drago.initial_stats(catalog, spec, fixed=(1, [(10, 3)] * 4))
        self.assertEqual(fixed[1], [[10, 3]] * 4)


if __name__ == "__main__":
    unittest.main()
