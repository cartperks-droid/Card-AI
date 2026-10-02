"""Reproducible review exports; inference and simulation limits stay visible.

Run ``python -m card_engine.reports`` after importing the source dataset.
This module reads the canonical dataset and writes review artifacts.
It never changes the source values when screenshot evidence differs.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from .catalog import load_catalog
from .simulator.catalog_rules import coverage, support_coverage
from .stats import base_stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/clean/dataset.json"
DEFAULT_OUTPUT = ROOT / "data/review"
STAT_ADVISORY = (
    "Generic dataset formula before abilities/supports; card-specific stat "
    "exceptions are not fully validated"
)

# Read directly from the user-supplied screenshots. These observations are
# evidence for review, not an authorization to alter the original XML rarity.
SCREENSHOT_RARITIES = (
    (48, 6660666, "6,660,666", 2, 4, 6),
    (132, 450000000, "450,000,000", 7, 3, 1),
)

CATALOG_FIELDS = (
    "card_id", "name", "pack_id", "pack", "visual_tags",
    "visual_inference_status", "gameplay_class_candidates",
    "class_assignment_status", "gameplay_classes_verified",
    "sourced_class_memberships", "player_availability",
    "availability_basis", "availability_confidence", "availability_as_of",
    "test_setup_requires_confirmation", "opponent_availability",
    "source_rarity", "rarity_conflict_status", "screenshot_rarity_if_reviewed",
    "weather_id", "intrinsic_weather", "intrinsic_weather_multiplier",
    "Card Modifier", "borderless_hp_formula", "borderless_atk_formula",
    "stat_advisory", "refined_description", "screenshot_file",
    "screenshot_number", "screenshot_card_ordinal",
)
COVERAGE_FIELDS = (
    "card_id", "name", "status", "training_labels_allowed", "reason",
)


def _cell(value):
    if type(value) is bool:
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _csv_text(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)
    return output.getvalue()


def source_conflicts(dataset):
    """Return both observed values with locations, leaving resolution explicit."""
    result = []
    for card_id, screenshot_value, display, image_number, row, column in SCREENSHOT_RARITIES:
        card = dataset["cards"][card_id - 1]
        screenshot = card["metadata"]["screenshot"]
        if screenshot["image_number"] != image_number:
            raise ValueError(f"Card {card_id} screenshot alignment changed; review rarity evidence")
        source = card["source"]["rarity"]
        result.append({
            "card_id": card_id,
            "name": card["name"],
            "field": "rarity",
            "status": "unresolved" if card["rarity"] != screenshot_value else "consistent",
            "canonical_value": card["rarity"],
            "canonical_value_lexical": card["rarity_lexical"],
            "canonical_source": source,
            "screenshot_value": screenshot_value,
            "screenshot_display": display,
            "screenshot_source": {
                "file": screenshot["file"], "image_number": image_number,
                "tile_row_1": row, "tile_column_1": column,
                "observation": "manual_visual_reading",
            },
            "resolution": "Retain canonical source value; no correction authorized",
        })
    return result


def build_reports(dataset_path=DEFAULT_DATASET):
    """Build all output text in memory before writing any review file."""
    dataset_path = Path(dataset_path)
    source_bytes = dataset_path.read_bytes()
    dataset = json.loads(source_bytes)
    catalog = load_catalog(dataset_path)
    conflicts = source_conflicts(dataset)
    conflicts_by_id = {row["card_id"]: row for row in conflicts}
    rows = []
    stat_rows = []
    coverage_rows = coverage(catalog)
    implemented = {row["card_id"] for row in coverage_rows if row["status"] != "unsupported"}
    for card in dataset["cards"]:
        card_id = card["card_id"]
        metadata = card["metadata"]
        availability = metadata["availability"]
        screenshot = metadata["screenshot"]
        weather = dataset["weathers"][card["weather_id"] - 1]
        stats = base_stats(catalog, card_id)
        stat_rows.append({"id": card_id, "name": card["name"], "attack": stats.attack, "hp": stats.hp,
                          "intrinsic_weather": weather["name"], "ability_implemented": card_id in implemented,
                          "verification": stats.verification})
        conflict = conflicts_by_id.get(card_id)
        # Preserve the provenance/status of each sourced class instead of
        # collapsing external claims into verified gameplay membership.
        sourced = [f"{r['class_name']}:{r.get('status', 'unverified')}"
                   for r in metadata["sourced_class_memberships"]]
        rows.append({
            "card_id": card_id,
            "name": card["name"],
            "pack_id": card["pack_id"],
            "pack": metadata["pack"],
            "visual_tags": " | ".join(metadata["visual_tags"]),
            "visual_inference_status": metadata["visual_inference_status"],
            "gameplay_class_candidates": " | ".join(metadata["gameplay_class_candidates"]),
            "class_assignment_status": card["class_assignment_status"],
            "gameplay_classes_verified": metadata["gameplay_classes_verified"],
            "sourced_class_memberships": " | ".join(sourced),
            "player_availability": availability["player_use"],
            "availability_basis": availability["basis"],
            "availability_confidence": availability["confidence"],
            "availability_as_of": availability["as_of"],
            "test_setup_requires_confirmation": availability["requires_confirmation_for_test_setup"],
            "opponent_availability": availability["opponent_use"],
            "source_rarity": card["rarity_lexical"],
            "rarity_conflict_status": conflict["status"] if conflict else "not_reviewed",
            "screenshot_rarity_if_reviewed": conflict["screenshot_value"] if conflict else "",
            "weather_id": weather["weather_id"],
            "intrinsic_weather": weather["name"],
            "intrinsic_weather_multiplier": weather["multiplier_lexical"],
            "Card Modifier": card["card_modifier_lexical"],
            "borderless_hp_formula": stats.hp,
            "borderless_atk_formula": stats.attack,
            "stat_advisory": STAT_ADVISORY + "; " + stats.verification,
            "refined_description": catalog.card(card_id).description,
            "screenshot_file": screenshot["file"],
            "screenshot_number": screenshot["image_number"],
            "screenshot_card_ordinal": screenshot["card_ordinal_excluding_clipped_repeats"],
        })
    conflict_document = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scope": "Two manually reviewed rarity discrepancies; not an exhaustive screenshot audit",
        "raw_sources_changed": False,
        "conflicts": conflicts,
    }
    stat_lines = ["# Baseline card stats for verification", "",
                  "No border, no mutation, no supports; before battle abilities. Intrinsic weather multipliers remain included.",
                  "Values are predictions unless separately confirmed. Toy Bear and Jason use fitted modifiers; variant scaling is unverified.",
                  "", "| ID | Card | ATK | HP |", "|---|---|---:|---:|"]
    stat_lines.extend(f"| {r['id']} | {r['name']} | {r['attack']:,} | {r['hp']:,} |" for r in stat_rows)
    return {
        "stats_to_verify.csv": _csv_text(stat_rows, tuple(stat_rows[0])),
        "stats_to_verify.md": "\n".join(stat_lines) + "\n",
        "catalog_overview.csv": _csv_text(rows, CATALOG_FIELDS),
        "simulator_coverage.csv": _csv_text(coverage_rows, COVERAGE_FIELDS),
        "support_coverage.csv": _csv_text(support_coverage(catalog), ("color", "support_id", "name", "status", "reason", "training_labels_allowed")),
        "source_conflicts.json": json.dumps(conflict_document, ensure_ascii=False, indent=2) + "\n",
    }


def export_reports(dataset_path=DEFAULT_DATASET, output_dir=DEFAULT_OUTPUT):
    """Write reproducible reports outside the raw source tree."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    raw_dir = dataset_path.resolve().parent.parent / "raw"
    if output_dir.resolve().is_relative_to(raw_dir):
        raise ValueError("Review reports cannot be written into raw sources")
    artifacts = build_reports(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (output_dir / name).write_text(content, encoding="utf-8", newline="")
    return tuple(output_dir / name for name in artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in export_reports(args.dataset, args.output):
        print(path)


if __name__ == "__main__":
    main()
