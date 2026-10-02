"""The restricted deck (player-base availability) and its commands."""

import unittest

from card_engine import restricted
from card_engine.catalog import load_catalog


class RestrictedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()

    def rules(self):
        return restricted.load("/nonexistent")  # the defaults

    def test_default_rules(self):
        rules = self.rules()
        cards = set(restricted.cards(rules, self.catalog))
        self.assertTrue({240, 228, 231, 227, 230} <= cards)  # Limited exceptions (user)
        self.assertFalse({213, 260, 270, 289} & cards)  # other Limited and seasonal cards
        self.assertNotIn(240, restricted.cards(rules, self.catalog, limited=False))
        self.assertEqual(restricted.borders(rules, 206), list(range(1, 16)))  # no perfect gem (GaRuCrPl)

    def test_per_card_changes(self):
        rules = self.rules()
        rules["added_cards"].append(260)  # Santa Claus back in
        rules["removed_cards"].append(37)  # Pandora out
        rules["border_overrides"]["206"] = {"enable": [16], "disable": [1]}
        cards = restricted.cards(rules, self.catalog)
        self.assertIn(260, cards)
        self.assertNotIn(37, cards)
        self.assertEqual(restricted.borders(rules, 206), list(range(2, 17)))
        self.assertIn((206, 16), restricted.entries(rules, self.catalog))


if __name__ == "__main__":
    unittest.main()
