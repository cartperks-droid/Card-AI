"""Per-entity label invalidation (training.flags)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from card_engine.catalog import load_catalog
from card_engine.training import flags


class FlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.current = flags.entity_hashes(cls.catalog)

    def rows(self):
        z = lambda *shape: np.zeros(shape, dtype=np.int16)
        cards = np.array([[[139, 3, 4, 5], [6, 7, 8, 9]],      # Buddha
                          [[10, 11, 12, 13], [14, 15, 16, 17]],
                          [[37, 3, 4, 5], [6, 7, 8, 9]]], dtype=np.int16)  # Pandora (random abilities)
        red = np.array([[27, 0], [0, 0], [0, 0]], dtype=np.int16)
        tier = np.array([[5, 0], [0, 0], [0, 0]], dtype=np.int16)
        return {"cards": cards, "borders": np.ones((3, 2, 4), np.int16), "mutations": z(3, 2, 4), "arts": z(3, 2, 4),
                "red": red, "red_tier": tier, "blue": z(3, 2), "blue_tier": z(3, 2)}

    def check(self, old, changes=()):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(flags, "SNAPSHOT_DIR", Path(temp)):
            (Path(temp) / "old.json").write_text(json.dumps(old))
            return flags.valid_rows(self.rows(), "old", self.current, self.catalog, list(changes)).tolist()

    def test_unchanged_rules_keep_everything(self):
        self.assertEqual(self.check(dict(self.current)), [True, True, True])

    def test_a_card_change_drops_its_rows_and_random_ability_rows_on_ability_changes(self):
        old = dict(self.current, **{"card:139": "different"})
        self.assertEqual(self.check(old), [False, True, True])
        old["pool"] = "different"  # an ability change also changes the pool every random card draws from
        self.assertEqual(self.check(old), [False, True, False])

    def test_support_changes(self):
        self.assertEqual(self.check(dict(self.current, **{"support:red27:5": "x"})), [False, True, True])
        self.assertEqual(self.check(dict(self.current, **{"support:red27:4": "x"})), [True, True, True])  # other tier
        self.assertEqual(self.check(dict(self.current, support_logic="x")), [False, True, True])

    def test_core_changes_wait_for_a_declaration(self):
        old = dict(self.current, core="oldcore")
        self.assertEqual(self.check(old), [False, False, False])  # undeclared: held back
        declared = {"from_core": "oldcore", "to_core": self.current["core"], "all": False,
                    "affects": ["card:10"], "note": "test"}
        self.assertEqual(self.check(old, [declared]), [True, False, True])
        self.assertEqual(self.check(old, [dict(declared, affects=[])]), [True, True, True])
        self.assertEqual(self.check(old, [dict(declared, all=True)]), [False, False, False])

    def test_declarations_chain_across_engine_versions(self):
        old = dict(self.current, core="v1")
        first = {"from_core": "v1", "to_core": "v2", "all": False, "affects": ["card:10"], "note": "test"}
        second = {"from_core": "v2", "to_core": self.current["core"], "all": False, "affects": [], "note": "test"}
        self.assertEqual(self.check(old, [first]), [False, False, False])  # v2 -> current undeclared: held back
        self.assertEqual(self.check(old, [first, second]), [True, False, True])  # v1 -> v2 -> current
        self.assertEqual(self.check(old, [dict(first, all=True), second]), [False, False, False])


if __name__ == "__main__":
    unittest.main()
