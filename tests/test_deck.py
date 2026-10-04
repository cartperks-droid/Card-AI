"""The player's deck file and its commands."""

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from card_engine import deck as deckmod
from card_engine.catalog import load_catalog


class DeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()

    def test_add_remove_and_render(self):
        deck = {"cards": [], "supports": []}
        deckmod.add_card(deck, self.catalog, "malik", deckmod._border("RuCrPl"))
        deckmod.add_card(deck, self.catalog, "206", deckmod._border("RuCrPl"))  # same entry: count 2
        deckmod.add_card(deck, self.catalog, "Sekhmet", deckmod._border("GaPl"), count=4)
        self.assertEqual([(e["card"], e["border"], e["count"]) for e in deck["cards"]], [(206, 8, 2), (70, 10, 4)])
        with self.assertRaises(SystemExit):
            deckmod.add_card(deck, self.catalog, "spirit", 1)  # ambiguous: Forest, Wind, Mist, Volcano Spirit...
        with self.assertRaises(SystemExit):
            deckmod.add_card(deck, self.catalog, "Sekhmet", 1, "Storm")  # weather cards cannot be mutated
        deckmod.add_support(deck, self.catalog, "Desmond", deckmod._tier("Platinum"))
        deckmod.add_support(deck, self.catalog, "Fate", deckmod._tier("ga"))
        self.assertEqual(deckmod.owned_supports(deck), ([(27, 2)], [(8, 5)]))
        deckmod.remove_card(deck, self.catalog, "Malik", count=1)
        self.assertEqual(deck["cards"][0]["count"], 1)
        with tempfile.TemporaryDirectory() as temp:
            path, doc = Path(temp) / "deck.json", Path(temp) / "deck.md"
            deckmod.save(deck, self.catalog, path, doc)
            self.assertEqual(deckmod.load(path), json.loads(path.read_text()))
            text = doc.read_text()
            self.assertIn("| Egypt | Sekhmet | 70 | GaPl | None | 4 |", text)
            self.assertIn("| red | Desmond Of Despair | 27 | Platinum | 1 |", text)
        self.assertEqual(deckmod.owned_entries(deck), [(70, 10, "None"), (206, 8, "None")])  # saved sorted by ID

    def test_remove_one_mutation_and_some_supports(self):
        deck = {"cards": [], "supports": []}
        deckmod.add_card(deck, self.catalog, "Good Boy", 2, "Storm", count=2)
        deckmod.add_card(deck, self.catalog, "Good Boy", 2)
        with self.assertRaises(SystemExit):
            deckmod.remove_card(deck, self.catalog, "Good Boy", 2)  # two mutations at that border: say which
        deckmod.remove_card(deck, self.catalog, "Good Boy", 2, count=1, mutation="Storm")
        self.assertEqual(deckmod.owned_entries(deck), [(3, 2, "Storm"), (3, 2, "None")])
        self.assertEqual(deck["cards"][0]["count"], 1)
        deckmod.remove_card(deck, self.catalog, "Good Boy", mutation="None")
        self.assertEqual(deckmod.owned_entries(deck), [(3, 2, "Storm")])
        deckmod.add_support(deck, self.catalog, "Fate", 5, count=3)
        deckmod.remove_support(deck, self.catalog, "Fate", count=2)
        self.assertEqual(deck["supports"][0]["count"], 1)
        deckmod.remove_support(deck, self.catalog, "Fate")
        self.assertEqual(deck["supports"], [])

    def test_custom_pool_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            files = {name: Path(temp) / name for name in ("deck.json", "deck.md", "custom.json", "custom.md")}
            deckmod.save({"cards": [], "supports": []}, self.catalog, files["deck.json"], files["deck.md"])
            with unittest.mock.patch.multiple(deckmod, DECK_FILE=files["deck.json"], DECK_DOC=files["deck.md"],
                                              CUSTOM_FILE=files["custom.json"], CUSTOM_DOC=files["custom.md"]):
                deckmod.main(["add", "Malik", "--border", "RuCrPl"])
                deckmod.main(["--custom", "add", "Sekhmet", "--count", "2"])
                self.assertEqual(deckmod.owned_entries(deckmod.load(files["custom.json"])), [(70, 1, "None")])
                deckmod.main(["--custom", "copy-deck"])
                self.assertEqual(deckmod.owned_entries(deckmod.load(files["custom.json"])), [(206, 8, "None")])
                self.assertIn("# Custom pool", files["custom.md"].read_text())
                deckmod.main(["--custom", "reset"])
                self.assertEqual(deckmod.load(files["custom.json"]), {"cards": [], "supports": []})
                with self.assertRaises(SystemExit):
                    deckmod.main(["reset"])  # only the custom pool can be reset
                self.assertEqual(deckmod.owned_entries(deckmod.load(files["deck.json"])), [(206, 8, "None")])


if __name__ == "__main__":
    unittest.main()
