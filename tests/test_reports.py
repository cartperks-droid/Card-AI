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

    def test_every_card_and_support_maps_to_the_engine(self):
        rows = list(csv.DictReader(io.StringIO(self.artifacts["engine_mapping.csv"])))
        self.assertEqual([int(r["id"]) for r in rows if r["kind"] == "card"], list(range(1, 290)))
        self.assertEqual([len([r for r in rows if r["kind"] == kind]) for kind in ("red", "blue")], [28, 15])
        self.assertTrue(all(r["supported"] == "true" and r["engine_name"] for r in rows))

    def test_conflicts_keep_canonical_values_and_exact_screenshot_readings(self):
        document = json.loads(self.artifacts["source_conflicts.json"])
        self.assertFalse(document["raw_sources_changed"])
        rows = document["conflicts"]
        self.assertEqual([(r["card_id"], r["canonical_value"], r["screenshot_value"]) for r in rows],
                         [(48, 6660666, 6660666), (132, 450000000, 450000000)])
        # DaddyDrago's rarities (adopted 2026-10-03) match both screenshot readings.
        self.assertTrue(all(r["status"] == "consistent" for r in rows))
        self.assertTrue(all(r["canonical_source"]["file"] == "../../card_engine/data_corrections.py" for r in rows))

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
