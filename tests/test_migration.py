"""Regression checks for the recovered source and its normalized representation."""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from card_engine.import_snap import import_project, USER_DESCRIPTION_EDITS
from card_engine.schema import SnapImportError, validate_dataset
from card_engine.catalog import load_catalog
from card_engine.stats import base_stats


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"


def source_value(element):
    """Independent decoder for the literal value forms in this specific snapshot."""
    if element.tag == "l":
        return element.text or ""
    if element.tag != "list":
        raise AssertionError(f"Unexpected source value {element.tag}")
    if element.get("struct") == "atomic":
        return next(csv.reader(io.StringIO(element.text or "")), [])
    return [source_value(item[0]) for item in element]


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "clean"
        cls.raw_hashes = hashes(RAW)
        cls.report = import_project(RAW, cls.output)
        cls.dataset = json.loads((cls.output / "dataset.json").read_text())
        tree = ET.parse(RAW / "Cards-2.xml")
        cls.source = {v.get("name"): source_value(v[0])
                      for group in tree.iter("variables") for v in group}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_import_preserves_source_bytes_and_is_reproducible(self):
        self.assertEqual(hashes(RAW), self.raw_hashes)
        before = hashes(self.output)
        import_project(RAW, self.output)
        self.assertEqual(hashes(self.output), before)

    def test_card_and_border_order_and_exact_descriptions(self):
        d = self.dataset
        self.assertEqual([c["card_id"] for c in d["cards"]], list(range(1, 290)))
        self.assertEqual([c["name"] for c in d["cards"]], self.source["Cards"])
        from card_engine.data_corrections import DRAGO_STATS
        expected_rarities = list(self.source["Card Rarities"])
        for card_id, (_, rarity, _, _) in DRAGO_STATS.items():  # DaddyDrago's data (user: more accurate), 2026-10-03.
            if int(expected_rarities[card_id - 1]) != rarity:
                expected_rarities[card_id - 1] = str(rarity)
        self.assertEqual([c["rarity_lexical"] for c in d["cards"]], expected_rarities)
        expected_refined = list(self.source["data-5_refined"])
        expected_refined[119] = expected_refined[119].replace("on the 3rd turn", "after 2 turns")
        for index, (source_text, clean_text, _) in USER_DESCRIPTION_EDITS.items():  # User-authorized 2026-09-30.
            self.assertEqual(expected_refined[index], source_text)
            expected_refined[index] = clean_text
        self.assertEqual([c["refined"] for c in d["descriptions"]], expected_refined)
        self.assertEqual([c["original"] for c in d["descriptions"]], self.source["Card Descriptions"][:-1])
        self.assertEqual([b["name"] for b in d["borders"]], self.source["Borders"])
        self.assertEqual(d["borders"][0]["name"], "")
        self.assertEqual(d["borders"][-1]["rarity"], 100000000000000000)
        self.assertTrue(all(len(c["packs"]) == 1 for c in d["cards"]))
        self.assertTrue(all(c["metadata"]["gameplay_classes_verified"] for c in d["cards"]))  # User-verified 2026-09-30.

    def test_overrides_use_all_values_and_only_authorized_weather_tail(self):
        weather = next(csv.reader(io.StringIO((RAW / "data-10.csv").read_text(encoding="utf-8-sig"))))
        damage = next(csv.reader(io.StringIO((RAW / "data-9.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(len(weather), 290)
        self.assertEqual(weather[-1], "1")
        expected_weather = list(map(int, weather[:-1]))
        expected_weather[35] = 1  # User corrected Bad Boys; no neighboring shift.
        expected_weather[36] = 9  # Explicit Pandora correction: Eclipse, not Rapture.
        self.assertEqual([c["weather_id"] for c in self.dataset["cards"]], expected_weather)
        expected_modifiers = list(damage)
        halloween = [c["card_id"] for c in self.dataset["cards"] if c["packs"][0] == "Halloween 2025"
                     and c["weather_id"] == 1 and c["card_id"] != 288]  # User: Halloween 2025 multipliers.
        for card_id in (*range(205, 213), 288, *halloween):
            expected_modifiers[card_id - 1] = {284: "5", 289: "3"}.get(card_id, "2")
        from card_engine.data_corrections import OBSERVED_MODIFIERS
        for card_id, (_, value) in OBSERVED_MODIFIERS.items():  # Observed in user videos.
            expected_modifiers[card_id - 1] = str(value)
        expected_modifiers[260] = str(2 / 3)  # Fit while preserving confirmed Snow.
        from card_engine.data_corrections import DRAGO_STATS
        for card_id, (_, _, multiplier, weather_multiplier) in DRAGO_STATS.items():  # DaddyDrago's data, 2026-10-03.
            if float(expected_modifiers[card_id - 1]) != multiplier / weather_multiplier:
                expected_modifiers[card_id - 1] = str(multiplier / weather_multiplier)
        self.assertEqual([c["card_modifier_lexical"] for c in self.dataset["cards"]], expected_modifiers)
        corrections = json.loads((self.output / "corrections.json").read_text())
        edits = [c for c in corrections if c["kind"] == "user_authorized_stat_metadata_correction"]
        drago = [e for e in edits if e["reason"].startswith("DaddyDrago")]  # 4 rarities + 34 multipliers, 2026-10-03.
        self.assertEqual(len(drago), 38)
        self.assertEqual(len(edits), 12 + len(OBSERVED_MODIFIERS) + len(halloween) + len(drago))  # Plus the Halloween multipliers (user, 2026-09-30).
        for edit in edits:
            if edit in drago:
                continue
            values = weather if edit["field"] == "weather_id" else damage
            self.assertEqual(edit["original_value_lexical"], values[edit["card_id"] - 1])
            self.assertEqual(edit["superseded_source"]["column_1"], edit["card_id"])
        for edit in drago:  # rarities replace the XML values; multipliers replace the CSV or an earlier correction
            if edit["field"] == "rarity":
                self.assertEqual(edit["original_value_lexical"], self.source["Card Rarities"][edit["card_id"] - 1])

    def test_user_corrected_baselines_and_confirmed_weather_eligibility(self):
        catalog = load_catalog(self.output / "dataset.json")
        for card_id, expected in {36: (320, 640), 205: (160, 320), 210: (2078, 4156),
                                  261: (1943, 3886), 288: (4788, 9575)}.items():
            with self.subTest(card_id=card_id):
                stats = base_stats(catalog, card_id)
                self.assertEqual((stats.attack, stats.hp), expected)
        for card_id, weather in {36: "Base", 37: "Eclipse", 38: "Rapture", 57: "Armageddon", 58: "Time Storm",
                                 258: "Snow", 260: "Aurora", 261: "Snow",
                                 265: "Snow", 280: "Blood Rain"}.items():
            self.assertEqual(catalog.weather(catalog.card(card_id).weather_id).name, weather)
        self.assertTrue(base_stats(catalog, 36, mutation="Storm").mutation_eligible)
        with self.assertRaises(ValueError):
            base_stats(catalog, 261, mutation="Storm")
        bosses = [card for card in catalog.cards if card.packs == ("Bosses",)]
        self.assertEqual([card.id for card in bosses], list(range(205, 213)))
        self.assertTrue(all(card.card_modifier == 2 for card in bosses))
        self.assertEqual(catalog.card(261).rarity, 400000000)
        self.assertEqual(catalog.card(94).rarity, 1500000)

    def test_support_names_and_nonduplicate_descriptions_preserved(self):
        for color, count in (("red", 28), ("blue", 15)):
            rows = [s for s in self.dataset["supports"] if s["color"] == color]
            title = color.title()
            self.assertEqual([r["support_id"] for r in rows], list(range(1, count + 1)))
            self.assertEqual([r["name"] for r in rows], self.source[f"Support Cards ({title})"])
            descriptions = self.source[f"Support Card Descriptions ({title})"]
            if color == "blue":
                self.assertEqual(descriptions[-1], descriptions[-2])
                descriptions = descriptions[:-1]
            else:
                # IMG_0350-0354: five red pairs carry each other's texts in the source; The Sequel lacks a '%'.
                from card_engine.import_snap import RED_SUPPORT_DESCRIPTION_SWAPS, RED_SUPPORT_TEXT_FIXES
                descriptions = list(descriptions)
                for a, b in RED_SUPPORT_DESCRIPTION_SWAPS:
                    descriptions[a - 1], descriptions[b - 1] = descriptions[b - 1], descriptions[a - 1]
                for sid, (_, fixed) in RED_SUPPORT_TEXT_FIXES.items():
                    descriptions[sid - 1] = fixed
            self.assertEqual([r["description"] for r in rows], descriptions)

    def test_canonical_tokens_roundtrip_every_character(self):
        tokens = self.dataset["tokens"]
        vocabulary = {v["token_id"]: v["token"] for v in tokens["vocabulary"]}
        self.assertEqual(tokens["special_ids"]["PAD"], 0)
        special_ids = set(tokens["special_ids"].values())
        lexical_ids = set(vocabulary) - special_ids
        self.assertGreater(tokens["special_ids"]["BOS"], max(lexical_ids))
        self.assertGreater(tokens["special_ids"]["CARD"], max(lexical_ids))
        for row, description in zip(tokens["rows"], self.dataset["descriptions"], strict=True):
            ids = row["token_ids"]
            self.assertEqual((ids[0], ids[-1]), (tokens["special_ids"]["BOS"], tokens["special_ids"]["CARD"]))
            self.assertTrue(set(ids[1:-1]).issubset(lexical_ids))
            decoded = row["separators"][0]
            for token, gap in zip(ids[1:-1], row["separators"][1:], strict=True):
                decoded += vocabulary[token] + gap
            self.assertEqual(decoded, description["refined"])
        self.assertIn("after 2 turns", self.dataset["descriptions"][119]["refined"])
        self.assertNotIn("3rd turn", self.dataset["descriptions"][119]["refined"])

    def test_supplied_tokens_change_only_by_authorized_rd_reindex(self):
        path = RAW / "user_2026-09-27"
        original = [[int(t) for t in row] for row in csv.reader((path / "data-6.csv").read_text(encoding="utf-8-sig").splitlines())]
        expected = [[t-1 if t>248 else t for t in row] for row in original]
        matrix = self.dataset["tokens"]["padded_matrix"]
        self.assertEqual([row for i, row in enumerate(matrix) if i not in USER_DESCRIPTION_EDITS],
                         [row for i, row in enumerate(expected) if i not in USER_DESCRIPTION_EDITS])
        # Edited rows (Academy Student 40 -> 30; True Prophet) are re-encoded with existing tokens only.
        self.assertEqual(sum(a != b for a, b in zip(matrix[87], expected[87])), 1)
        supplied_vocab = next(csv.reader(io.StringIO((path / "data-7.csv").read_text(encoding="utf-8-sig"))))
        lexical = [r["token"] for r in self.dataset["tokens"]["vocabulary"] if r["kind"] == "lexical"]
        self.assertEqual(lexical, supplied_vocab)
        self.assertNotIn("rd", lexical)
        self.assertEqual(self.dataset["tokens"]["special_ids"], {"PAD": 0, "BOS": 439, "CARD": 440})

    def test_complete_source_archive_matches_each_variable(self):
        archive = json.loads((self.output / "archive/persistent_variables.json").read_text())
        self.assertEqual(len(archive["variables"]), 41)
        self.assertEqual({r["name"]: r["value"] for r in archive["variables"]}, self.source)
        self.assertEqual(len(self.source["Rarity Tokens"]), 32768)

    def test_output_cannot_pollute_raw_inventory(self):
        with self.assertRaises(SnapImportError):
            import_project(RAW, RAW / "forbidden_generated_output")

    def test_pack_boundaries_and_availability_are_evidenced(self):
        self.assertEqual(len(self.dataset["packs"]), 14)
        self.assertEqual(self.dataset["cards"][64]["packs"], ["Era 1"])
        self.assertEqual(self.dataset["cards"][65]["packs"], ["Egypt"])
        self.assertEqual(self.dataset["cards"][119]["packs"], ["Immortal"])
        self.assertEqual(self.dataset["cards"][273]["packs"], ["Halloween 2025"])
        for card in self.dataset["cards"]:
            self.assertTrue((RAW / card["metadata"]["screenshot"]["file"]).is_file())
            self.assertEqual(card["metadata"]["availability"]["opponent_use"], "training_dummy_available")
        self.assertEqual(self.dataset["cards"][2]["metadata"]["availability"]["basis"], "user_battle_observation")

    def test_card_modifier_scales_both_confirmed_rapture_stats(self):
        catalog = load_catalog(self.output / "dataset.json")
        for card_id in (33, 38):
            stats = base_stats(catalog, card_id)
            self.assertEqual((stats.hp, stats.attack), (2560, 1280))
            self.assertEqual(stats.intrinsic_weather_multiplier, 3)
        # Platinum changes rarity by x100, hence the pre-rounding stat factor by x4.
        stats = base_stats(catalog, 1, 2)
        self.assertEqual(stats.effective_rarity, 200)
        self.assertEqual((stats.hp, stats.attack), (50, 25))

    def test_loader_rejects_non_numeric_and_non_finite_modifiers(self):
        for value in ("oops", float("nan"), 0, True):
            bad = copy.deepcopy(self.dataset)
            bad["cards"][0]["card_modifier"] = value
            with self.subTest(value=value), self.assertRaises(SnapImportError):
                validate_dataset(bad)

    def test_registry_covers_all_cards_but_is_not_executable(self):
        self.assertFalse(self.dataset["simulation_ready"])
        registry = self.dataset["registry"]
        self.assertEqual(len(registry), 372)
        self.assertEqual({h["card_id"] for h in registry}, set(range(1, 290)))
        self.assertTrue(all(h["verification_status"] == "unverified" for h in registry))

    def test_source_token_errors_remain_observable(self):
        vocabulary = self.source["data-5_refined_vocab_ordered"]
        matrix = self.source["data-5_refined_tokenized_2d"]
        self.assertEqual(len(vocabulary), 440)
        self.assertEqual(max(int(t) for row in matrix for t in row), 441)
        self.assertEqual(vocabulary[int(matrix[1][1]) - 1], "reveal")
        self.assertTrue(self.dataset["descriptions"][1]["refined"].startswith("deduct"))

    def test_typed_catalog_rejects_python_negative_indices_and_wrong_namespace(self):
        catalog = load_catalog(self.output / "dataset.json")
        self.assertEqual(catalog.card(33).name, "Hell's Army")
        self.assertEqual(catalog.border(1).code, "")
        self.assertEqual(catalog.support("red", 1).name, "Adventurer")
        for bad_id in (0, -1, 290, True, 1.0, "1"):
            with self.subTest(identifier=bad_id), self.assertRaises(ValueError):
                catalog.card(bad_id)
        with self.assertRaises(ValueError):
            catalog.support("green", 1)

    def test_validation_rejects_misaligned_ids_and_fabricated_readiness(self):
        for mutate in (lambda d: d["cards"][0].update(card_id=2),
                       lambda d: d["cards"][0].update(weather_id=99),
                       lambda d: d.update(simulation_ready=True),
                       lambda d: d["tokens"]["rows"][0]["token_ids"].__setitem__(1, 999999)):
            bad = copy.deepcopy(self.dataset)
            mutate(bad)
            with self.assertRaises(SnapImportError):
                validate_dataset(bad)

    def test_invalid_external_list_never_replaces_successful_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "raw"
            shutil.copytree(RAW, source)
            destination = root / "clean"
            import_project(source, destination)
            before = hashes(destination)
            (source / "data-9.csv").write_text("1,1\n")
            with self.assertRaises(SnapImportError):
                import_project(source, destination)
            self.assertEqual(hashes(destination), before)


if __name__ == "__main__":
    unittest.main()
