import unittest

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

    def test_rejects_bad_input(self):
        bad = dict(TEAM, cards=["--checkpoint", "x", "y", "z"])
        for tool, params in [("battle", {"team": bad, "enemy_mode": "tower"}),
                             ("battle", {"team": dict(TEAM, cards=TEAM["cards"][:3]), "enemy_mode": "tower"}),
                             ("battle", {"team": TEAM, "enemy_mode": "tower", "floor": 106}),
                             ("depths_run", {"team": TEAM, "workers": 100}),
                             ("shell", {})]:
            with self.assertRaises(ValueError):
                web.command(tool, params)


if __name__ == "__main__":
    unittest.main()
