import tempfile
import unittest
from pathlib import Path
from unittest import mock

from card_engine import web


TEAM = {"cards": ["Parallax", "Judgement Day", "Judgement Day", "Judgement Day"], "blue": "Fate", "red": ""}


class WebTests(unittest.TestCase):
    def test_battle_command(self):
        args = web.command("battle", {"team": TEAM, "enemy_mode": "tower", "floor": 65, "difficulty": "Impossible"})
        self.assertEqual(args[0], "card_engine.training.predict")
        self.assertIn("--simulate", args)
        self.assertEqual(args[args.index("--tower") + 1:args.index("--tower") + 3], ["65", "Impossible"])
        self.assertEqual(args[args.index("--ally-blue") + 1], "Fate")
        self.assertNotIn("--ally-red", args)

    def test_depths_commands(self):
        args = web.command("depths_improve", {"team": TEAM, "pool": "progression", "rolls": 46e6, "luck": 100,
                                              "bans": ["Piccolo"], "optimize_bans": 7})
        self.assertEqual(args[:2], ["card_engine.depths", "improve"])
        self.assertEqual(args[args.index("--bans") + 1], "Piccolo")
        self.assertIn("--no-limited", args)
        self.assertEqual(web.command("depths_stats", {"objective": "depth"})[-1], "expected_floors")

    def test_counters_against_generated_enemies_at_fixed_stats(self):
        args = web.command("counters", {"enemy_mode": "broad_tower", "floor": 104, "difficulty": "Impossible", "enemies": 3})
        self.assertEqual(args[args.index("--enemies"):args.index("--enemies") + 5], ["--enemies", "3", "--tower", "104", "Impossible"])
        args = web.command("counters", {"enemy_mode": "broad_stats", "hp": 2e6, "atk": 1e6})
        self.assertEqual(args[args.index("--enemy-stats") + 1:args.index("--enemy-stats") + 3], ["2000000.0", "1000000.0"])
        self.assertNotIn("--tower", web.command("counters", {"enemy_mode": "broad"}))

    def test_annotate_rarity_and_stats(self):
        parallax, plain, unknown = web.annotate([{"name": "Parallax", "border": "GaPl", "mutation": "None"},
                                                 {"name": "Parallax", "border": "none", "mutation": "None"}, {"name": "?"}])
        self.assertGreater(parallax["rarity"], plain["rarity"])
        self.assertGreater(parallax["hp"], plain["hp"])
        self.assertEqual(unknown, {"name": "?"})

    def test_rejects_bad_input(self):
        bad = dict(TEAM, cards=["--checkpoint", "x", "y", "z"])
        for tool, params in [("battle", {"team": bad, "enemy_mode": "tower"}),
                             ("battle", {"team": dict(TEAM, cards=TEAM["cards"][:3]), "enemy_mode": "tower"}),
                             ("battle", {"team": TEAM, "enemy_mode": "tower", "floor": 106}),
                             ("depths_run", {"team": TEAM, "workers": 100}),
                             ("shell", {})]:
            with self.assertRaises(ValueError):
                web.command(tool, params)

    def test_deck_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.multiple("card_engine.deck", DECK_FILE=tmp / "deck.json", DECK_DOC=tmp / "deck.md"):
                saved = web.deck_save("own", {
                    "cards": [{"name": "Parallax", "border": "GaPl", "mutation": "None", "count": 2},
                              {"name": "Parallax", "border": "GaPl", "mutation": "None", "count": 1},
                              {"name": "Good Boy", "border": "none", "mutation": "Storm", "count": 1}],
                    "supports": [{"color": "blue", "name": "Fate", "tier": "Crystal", "count": 1}]})
                self.assertEqual(web.deck_view("own"), saved)
                self.assertIn(("Parallax", "GaPl", "None", 3), [(e["name"], e["border"], e["mutation"], e["count"]) for e in saved["cards"]])
                self.assertEqual(saved["supports"], [{"color": "blue", "name": "Fate", "tier": "Crystal", "count": 1}])
                self.assertTrue((tmp / "deck.md").exists())
                for bad in ({"cards": [{"name": "No Such Card"}]}, {"cards": [{"name": "Parallax", "border": "Xx"}]},
                            {"supports": [{"color": "red", "name": "Fate"}]},
                            {"cards": [{"name": "Judgement Day", "mutation": "Storm"}]}):
                    with self.assertRaises(ValueError):
                        web.deck_save("own", bad)
                self.assertEqual(web.deck_view("own"), saved)
        with self.assertRaises(ValueError):
            web.deck_view("../../etc")


if __name__ == "__main__":
    unittest.main()
