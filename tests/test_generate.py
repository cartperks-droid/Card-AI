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

    def test_slot_tokens_reproduce_classifiers_with_stats_apart_from_the_card(self):
        from card_engine.model.config import StrategicConfig
        for layout in ({"stat_width": 64}, {"stat_tokens": True}, {"stat_tokens": True, "stat_pairs": True}):
            torch.manual_seed(1)
            model = BattleModel(strategic_config=StrategicConfig(layers=2, pack_embedding=False, mutation_embedding=False,
                                                                 **layout))
            path = Path(self.temp.name) / "layout.checkpoint"
            save_checkpoint(model, path)
            classifier = Classifier(path, "cpu")
            space = generate.SlotSpace(classifier, self.pool)
            rng = np.random.default_rng(4)
            teams = [generate.random_team(self.pool, rng) for _ in range(6)]
            with torch.no_grad():
                logits = space.logits(space.fixed(teams[:3]), space.fixed(teams[3:]))
            expected = classifier.win_a([spec(a, b) for a, b in zip(teams[:3], teams[3:])])
            np.testing.assert_allclose(logits.softmax(-1)[:, 0].numpy(), expected, atol=1e-5, err_msg=str(layout))

    def test_fixed_enemy_stats_reach_the_classifier(self):
        rng = np.random.default_rng(5)
        allies = [generate.random_team(self.pool, rng) for _ in range(3)]
        enemy = generate.random_team(self.pool, rng)
        stats = (2.5e6, 4e5, True)  # HP times each card's multiplier, as on tower floors
        with torch.no_grad():
            logits = self.space.logits(self.space.fixed(allies), self.space.fixed([enemy] * 3, stats))
        wins = self.classifier.ally_win([(ally, enemy) for ally in allies], stats)
        np.testing.assert_allclose(logits.softmax(-1)[:, 0].numpy(), wins[:, 0], atol=1e-5)
        self.assertFalse(np.allclose(wins, self.classifier.ally_win([(ally, enemy) for ally in allies])))

    def test_single_copy_cards_appear_once(self):
        pool = generate.make_pool(self.catalog, "all", borders=[1, 2])
        space = generate.SlotSpace(self.classifier, pool)
        fate = [i for i, e in enumerate(pool.entries.tolist()) if e[0] == 240]
        other = [i for i, e in enumerate(pool.entries.tolist()) if e[0] == 3]
        self.assertFalse(generate._allowed(space, [fate[0], fate[1], other[0], other[1]]))  # two Fate Seamstress
        self.assertTrue(generate._allowed(space, [fate[0], other[0], other[1], other[0]]))

    def test_counters_are_decoded_from_the_pool(self):
        enemy = parse_side(self.catalog, ["Immortal Witch", "Archer", "Good Boy", "Set"])
        allowed = {tuple(e) for e in self.pool.entries.tolist()}
        for column, role in enumerate(generate.ROLES):  # each role on its own: attacking first, or defending
            found = generate.counters(self.space, enemy, count=3, restarts=4,
                                      settings=generate.Settings(steps=30, rechecks=1, nearest=2, role=role))
            self.assertTrue(found)
            for team, win, blur in found:
                self.assertTrue(all(e in allowed for e in zip(team["cards"], team["borders"], team["mutations"], team["arts"])))
                self.assertTrue((team["red"], team["red_tier"]) in self.pool.reds)
                self.assertAlmostEqual(win, self.classifier.ally_win([(team, enemy)])[0][column], places=5)
            scores = [win for _, win, _ in found]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_engine_search_stays_in_the_pool_and_walks_from_the_model_teams(self):
        enemy = parse_side(self.catalog, ["Immortal Witch", "Archer", "Good Boy", "Set"])
        allowed = {tuple(e) for e in self.pool.entries.tolist()}
        start = generate.random_team(self.pool, np.random.default_rng(3))
        found = generate.engine_search(self.space, enemy, [start], evaluations=40, enemy_stats=(5e5, 2e5, False),
                                       workers=2, population=8, parents=2, children=4, catalog=self.catalog)
        self.assertGreaterEqual(len(found), 8)
        self.assertIn(generate._key(start), {generate._key(team) for team, _ in found})
        for team, win in found:
            self.assertTrue(all(e in allowed for e in zip(team["cards"], team["borders"], team["mutations"], team["arts"])))
            self.assertTrue(-1e-9 <= win <= 1 + 1e-9, win)  # summed branch probabilities
        self.assertEqual([w for _, w in found], sorted((w for _, w in found), reverse=True))

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

    def test_masks_apply_to_every_pool(self):
        borders, mutations, tiers = generate.masks(["none"], ["None"], ["base"])  # the defaults: rare options off
        self.assertEqual((borders, mutations, tiers), ([1], [0], [1]))
        self.assertEqual(generate.masks(["all"], ["All"], ["all"]), (None, None, None))
        self.assertEqual(generate.masks(["Pl", "none"], ["Storm"], ["Platinum", "5"]), ([1, 2], [1], [2, 5]))
        self.assertEqual(generate.masks(["none,", "Pl,", "Cr"], ["None,Storm"], ["all"]), ([1, 2, 3], [0, 1], None))
        pool = generate.make_pool(self.catalog, "all", borders=[1, 2], mutations=[0, 1], tiers=[1])
        self.assertTrue(set(pool.entries[:, 1]) == {1, 2} and set(pool.entries[:, 2]) == {0, 1})
        self.assertTrue(all(tier == 1 for _, tier in pool.reds + pool.blues))
        deck = {"cards": [{"card": 1, "border": 1, "mutation": "None", "count": 1},
                          {"card": 3, "border": 2, "mutation": "Storm", "count": 3}],
                "supports": [{"color": "red", "support": 4, "tier": 5, "count": 1, "name": "Stormcaller"}]}
        path = Path(self.temp.name) / "custom.json"
        path.write_text(__import__("json").dumps(deck))
        # a deck overrides the masks: the custom pool is used as it is
        pool = generate.make_pool(self.catalog, "custom", deck_path=path, borders=[1], mutations=[0], tiers=[1],
                                  max_rarity=10)
        self.assertEqual((pool.entries.tolist(), pool.copies.tolist()), ([[1, 1, 0, 0], [3, 2, 1, 0]], [1, 3]))
        self.assertEqual(pool.reds, [(4, 5)])
        common = generate.make_pool(self.catalog, "restricted", borders=None, max_rarity=1e10)  # card x border rarity
        total = [self.catalog.card(int(c)).rarity * self.catalog.border(int(b)).rarity for c, b in common.entries[:, :2]]
        self.assertTrue(len(total) and max(total) <= 1e10 and {1, 2, 3} <= set(common.entries[:, 1].tolist()))
        with self.assertRaises(SystemExit):
            generate.make_pool(self.catalog, "all", max_rarity=1)  # every card is rarer


if __name__ == "__main__":
    unittest.main()
