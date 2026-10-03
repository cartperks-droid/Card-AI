"""Team parsing, the matchup classifier and the generator's slot space."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from card_engine.catalog import load_catalog
from card_engine.model import BattleModel, save_checkpoint
from card_engine.teams import describe, parse_card, parse_side, parse_support, spec
from card_engine.training import generate
from card_engine.training.predict import Classifier


class TeamTests(unittest.TestCase):
    def test_names_borders_mutations_and_supports(self):
        catalog = load_catalog()
        self.assertEqual(parse_card(catalog, "Vampire Lord@GaPl"), (91, 10, 0, 0))
        self.assertEqual(parse_card(catalog, "Good Boy@Pl/Storm"), (3, 2, 1, 0))
        self.assertEqual(parse_card(catalog, "Astraeus+Virgo@Pl"), (56, 2, 0, 3))
        with self.assertRaises(SystemExit):
            parse_card(catalog, "Astraeus")  # each art is its own card
        self.assertEqual(parse_support(catalog, "red", "Stormcaller@Ruby"), (4, 4))
        self.assertEqual(parse_support(catalog, "blue", "Guardian Angel"), (15, 1))
        team = parse_side(catalog, ["Vampire Lord@GaPl", "Set", "Good Boy@Pl/Storm", "Archer"], "Stormcaller@Galaxy")
        self.assertEqual(describe(catalog, team), {"cards": ["Vampire Lord@GaPl", "Set", "Good Boy@Pl/Storm", "Archer"],
                                                   "red": "Stormcaller@Galaxy"})
        with self.assertRaises(SystemExit):
            parse_card(catalog, "Pandora/Storm")  # a weather card cannot mutate


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        model = BattleModel()
        for name, parameter in model.named_parameters():  # trained-like: identities and the stat MLP are not zero
            if "identity_embedding" in name or "stat_" in name:
                torch.nn.init.normal_(parameter, std=0.02)
        path = Path(cls.temp.name) / "model.checkpoint"
        save_checkpoint(model, path)
        cls.catalog = load_catalog()
        cls.classifier = Classifier(path, "cpu")
        cls.pool = generate.make_pool(cls.catalog, "all", borders=[1, 2, 9])
        cls.space = generate.SlotSpace(cls.classifier, cls.pool)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_slot_tokens_reproduce_the_classifier(self):
        rng = np.random.default_rng(3)
        teams = [generate.random_team(self.pool, rng) for _ in range(5)]
        teams.append(parse_side(self.catalog, ["Astraeus+Gemini", "Astraeus+Virgo@Pl", "Archer", "Set"]))  # art identities
        with torch.no_grad():
            logits = self.space.logits(self.space.fixed(teams[:3]), self.space.fixed(teams[3:]))
        expected = self.classifier.win_a([spec(a, b) for a, b in zip(teams[:3], teams[3:])])
        np.testing.assert_allclose(logits.softmax(-1)[:, 0].numpy(), expected, atol=1e-5)

    def test_counters_are_decoded_from_the_pool(self):
        enemy = parse_side(self.catalog, ["Immortal Witch", "Archer", "Good Boy", "Set"])
        found = generate.counters(self.space, enemy, count=3, restarts=4,
                                  settings=generate.Settings(steps=30, rechecks=1, nearest=2))
        allowed = {tuple(e) for e in self.pool.entries.tolist()}
        self.assertTrue(found)
        for team, wins, blur in found:
            self.assertTrue(all(e in allowed for e in zip(team["cards"], team["borders"], team["mutations"], team["arts"])))
            self.assertTrue((team["red"], team["red_tier"]) in self.pool.reds)
            np.testing.assert_allclose(wins, self.classifier.ally_win([(team, enemy)])[0], atol=1e-5)
        scores = [sum(wins) for _, wins, _ in found]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_own_pool_respects_copy_counts(self):
        deck = {"cards": [{"card": 1, "border": 1, "mutation": "None", "count": 1},
                          {"card": 3, "border": 2, "mutation": "Storm", "count": 3}],
                "supports": [{"color": "red", "support": 4, "tier": 5, "count": 1, "name": "Stormcaller"}]}
        path = Path(self.temp.name) / "deck.json"
        path.write_text(__import__("json").dumps(deck))
        pool = generate.make_pool(self.catalog, "own", deck_path=path)
        self.assertEqual((pool.entries.tolist(), pool.copies.tolist()), ([[1, 1, 0, 0], [3, 2, 1, 0]], [1, 3]))
        self.assertEqual((pool.reds, pool.blues), ([(4, 5)], [(0, 0)]))
        space = generate.SlotSpace(self.classifier, pool)
        self.assertFalse(generate._allowed(space, [0, 0, 1, 1]))
        self.assertTrue(generate._allowed(space, [0, 1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
