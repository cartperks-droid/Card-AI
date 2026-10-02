"""Measured mutation snapshots constrain the fit and its integration."""

import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from card_engine.catalog import load_catalog
from card_engine.mutations import (
    ANNOTATIONS, MULTIPLIERS, export_fit, fit_intervals, generate_fit,
    mutation_multiplier,
)
from card_engine.stats import base_stats


class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.annotations = json.loads(ANNOTATIONS.read_text())

    def test_all_28_measured_pairs_match_formula_and_preserve_report_order(self):
        rows = generate_fit(self.catalog)
        self.assertEqual([row["card_id"] for row in rows], [69] * 8 + [67] * 9 + [10] * 11)
        self.assertEqual([(row["reported_attack"], row["reported_hp"]) for row in rows], [
            (142,284), (156,312), (170,340), (185,369), (255,510), (284,567), (496,992), (638,1275),
            (65,130), (72,143), (78,156), (98,195), (85,169), (117,234), (130,260), (163,325), (195,390),
            (33,65), (36,72), (39,78), (43,85), (49,98), (59,117), (65,130), (82,163), (98,195), (130,260), (147,293),
        ])
        self.assertEqual([row["mutation"] for row in rows[11:13]], ["Shroud", "Aurora"])
        self.assertEqual([row["mutation"] for row in rows[20:22]], ["Aurora", "Shroud"])
        for row in rows:
            with self.subTest(card=row["card_name"], mutation=row["mutation"]):
                stats = base_stats(self.catalog, row["card_id"], mutation=row["mutation"])
                self.assertEqual((stats.attack, stats.hp), (row["reported_attack"], row["reported_hp"]))
                self.assertEqual(stats.mutation_multiplier, row["candidate_multiplier"])

    def test_mutation_is_applied_before_base_stat_rounding(self):
        unmutated = base_stats(self.catalog, 69)
        storm = base_stats(self.catalog, 69, mutation="Storm")
        self.assertEqual((unmutated.attack, storm.attack), (142, 156))
        self.assertEqual(math.ceil(unmutated.attack * 1.1), 157)

    def test_joint_rounding_intervals_contain_candidates_but_do_not_prove_uniqueness(self):
        intervals = fit_intervals(generate_fit(self.catalog))
        self.assertEqual({row["mutation"] for row in intervals}, set(MULTIPLIERS))
        self.assertEqual(sum(row["observation_count"] for row in intervals), 28)
        for row in intervals:
            self.assertLess(row["lower_exclusive"], row["candidate_multiplier"])
            self.assertLessEqual(row["candidate_multiplier"], row["upper_inclusive"])
            self.assertGreater(row["upper_inclusive"], row["lower_exclusive"])
            self.assertFalse(row["unique_factor_proven"])

    def test_unknown_mutations_and_unobserved_intrinsic_stacking_fail(self):
        for name in ("Rapture", "storm", "Storm ", None, True, 1, [], {}):
            with self.subTest(name=name), self.assertRaises(ValueError):
                mutation_multiplier(name)
        for card_id in (33, 38):
            with self.subTest(card_id=card_id), self.assertRaisesRegex(ValueError, "stacking"):
                base_stats(self.catalog, card_id, mutation="Storm")

    def test_conflicting_stat_or_factor_annotation_fails_without_replacing_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            output = temporary / "fit.csv"
            export_fit(self.catalog, output)
            before = output.read_bytes()
            for mutation in (lambda d: d["observations"][0].update(hp=285),
                             lambda d: d["candidate_multipliers"][1].update(multiplier=1.11)):
                bad = copy.deepcopy(self.annotations)
                mutation(bad)
                annotation_path = temporary / "bad.json"
                annotation_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    export_fit(self.catalog, output, annotation_path)
                self.assertEqual(output.read_bytes(), before)

    def test_fit_csv_is_reproducible_and_roundtrips_all_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fit.csv"
            export_fit(self.catalog, output)
            before = output.read_bytes()
            export_fit(self.catalog, output)
            self.assertEqual(output.read_bytes(), before)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["observation_id"]) for row in rows], list(range(1,29)))
            self.assertTrue(all(row["matches"] == "true" for row in rows))
            for row, observation in zip(rows, self.annotations["observations"], strict=True):
                self.assertEqual((int(row["reported_attack"]), int(row["reported_hp"])),
                                 (observation["attack"], observation["hp"]))


if __name__ == "__main__":
    unittest.main()
