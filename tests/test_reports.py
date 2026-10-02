"""Review exports preserve source identity and expose their uncertainty."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from card_engine.reports import DEFAULT_DATASET, build_reports, export_reports


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = build_reports()
        cls.dataset = json.loads(DEFAULT_DATASET.read_text())

    def test_csv_roundtrip_preserves_order_names_descriptions_and_inference_status(self):
        rows = list(csv.DictReader(io.StringIO(self.artifacts["catalog_overview.csv"])))
        self.assertEqual([int(row["card_id"]) for row in rows], list(range(1, 290)))
        for row, card, description in zip(rows, self.dataset["cards"], self.dataset["descriptions"], strict=True):
            self.assertEqual(row["name"], card["name"])
            self.assertEqual(row["refined_description"], description["refined"])
            self.assertEqual(row["source_rarity"], card["rarity_lexical"])
            self.assertEqual(row["Card Modifier"], card["card_modifier_lexical"])
            self.assertEqual(row["class_assignment_status"], "user_verified")
            self.assertEqual(row["gameplay_classes_verified"], "true")
            self.assertEqual(row["opponent_availability"], "training_dummy_available")
            self.assertIn("not fully validated", row["stat_advisory"])
        self.assertEqual(rows[2]["availability_confidence"], "confirmed")
        self.assertEqual(rows[182]["player_availability"], "unavailable")
        self.assertEqual(rows[77]["player_availability"], "unknown")
        for index in (32, 37):
            self.assertEqual((rows[index]["borderless_hp_formula"], rows[index]["borderless_atk_formula"]), ("2560", "1280"))

    def test_coverage_is_conservative_for_every_card(self):
        rows = list(csv.DictReader(io.StringIO(self.artifacts["simulator_coverage.csv"])))
        self.assertEqual([int(row["card_id"]) for row in rows], list(range(1, 290)))
        self.assertEqual(sum(r["status"] == "experimental_complete_ability_subset" for r in rows), 289)
        self.assertEqual(sum(r["status"] == "unsupported" for r in rows), 0)
        self.assertTrue(all(r["training_labels_allowed"] == "false" for r in rows))
        supports = list(csv.DictReader(io.StringIO(self.artifacts["support_coverage.csv"])))
        self.assertEqual(len(supports), 43)
        self.assertEqual([(r["color"], r["support_id"]) for r in supports if r["status"] != "unsupported"], [('red', '1'), ('red', '2'), ('red', '3'), ('red', '4'), ('red', '5'), ('red', '6'), ('red', '7'), ('red', '8'), ('red', '9'), ('red', '10'), ('red', '11'), ('red', '12'), ('red', '13'), ('red', '14'), ('red', '15'), ('red', '16'), ('red', '17'), ('red', '18'), ('red', '19'), ('red', '20'), ('red', '21'), ('red', '22'), ('red', '23'), ('red', '24'), ('red', '25'), ('red', '26'), ('red', '27'), ('red', '28'), ('blue', '1'), ('blue', '2'), ('blue', '3'), ('blue', '4'), ('blue', '5'), ('blue', '6'), ('blue', '7'), ('blue', '8'), ('blue', '9'), ('blue', '10'), ('blue', '11'), ('blue', '12'), ('blue', '13'), ('blue', '14'), ('blue', '15')])

    def test_conflicts_keep_canonical_values_and_exact_screenshot_readings(self):
        document = json.loads(self.artifacts["source_conflicts.json"])
        self.assertFalse(document["raw_sources_changed"])
        rows = document["conflicts"]
        self.assertEqual([(r["card_id"], r["canonical_value"], r["screenshot_value"]) for r in rows],
                         [(48, 6660000, 6660666), (132, 400000000, 450000000)])
        self.assertTrue(all(r["status"] == "unresolved" for r in rows))
        self.assertTrue(all(r["canonical_source"]["file"] == "Cards-2.xml" for r in rows))

    def test_export_is_reproducible_and_does_not_modify_dataset(self):
        before = DEFAULT_DATASET.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_reports(output_dir=Path(temporary))
            first = {p.name: p.read_bytes() for p in paths}
            export_reports(output_dir=Path(temporary))
            self.assertEqual(first, {p.name: p.read_bytes() for p in paths})
            self.assertEqual(set(first), set(self.artifacts))
        self.assertEqual(DEFAULT_DATASET.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "raw sources"):
            export_reports(output_dir=DEFAULT_DATASET.parent.parent / "raw/review")


if __name__ == "__main__":
    unittest.main()
