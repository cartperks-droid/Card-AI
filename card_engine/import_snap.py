"""Import the supplied Snap! project without executing it or inferring battle rules.

The raw XML remains authoritative for persistence. Scalar strings are never
coerced in the archive; clean numeric fields always retain lexical companions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .schema import DATASET_SCHEMA, SCHEMA_VERSION, SnapImportError, require, validate_dataset

TOKEN_PATTERN = r"\d+(?:\.\d+)?|[^\W\d_]+(?:['’][^\W\d_]+)*|[^\s]"
TOKEN_RE = re.compile(TOKEN_PATTERN)
SOURCE_TOKEN_PATTERN = r"\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?|[^\w\s']"
SOURCE_TOKEN_RE = re.compile(SOURCE_TOKEN_PATTERN)
INTEGER_RE = re.compile(r"[+-]?\d+\Z")
NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def read_csv_text(text: str, *, label: str = "CSV") -> list[list[str]]:
    """CSV escaping matters: a comma token is serialized as `","`."""
    try:
        return list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise SnapImportError(f"{label}: malformed CSV: {exc}") from exc


def parse_value(element: ET.Element):
    """Decode Snap atomic CSV and item lists while retaining lexical scalars."""
    if element.tag == "list":
        if element.get("struct") == "atomic":
            require(not len(element), "Atomic Snap list unexpectedly has child elements")
            rows = read_csv_text(element.text or "", label="Snap atomic list")
            require(len(rows) <= 1, "Atomic Snap list contains multiple CSV records")
            return rows[0] if rows else []
        values = []
        for item in element:
            require(item.tag == "item", f"Unexpected Snap list child: {item.tag}")
            require(len(item) == 1, "Snap list item must contain exactly one serialized value")
            values.append(parse_value(item[0]))
        return values
    if element.tag in ("l", "bool") and len(element) == 0:
        return element.text or ""
    # Preserve unsupported Snap values rather than silently discarding children.
    return {"opaque_xml": ET.tostring(element, encoding="unicode")}


def _leaves(value):
    if isinstance(value, list):
        for child in value:
            yield from _leaves(child)
    elif isinstance(value, dict):
        yield value
    else:
        yield value


def _dimensions(value):
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    children = [_dimensions(child) for child in value]
    depth = max(map(len, children))
    return [len(value)] + [
        children[0][i] if all(len(c) > i and c[i] == children[0][i] for c in children) else None
        for i in range(depth)
    ]


def _dtype(value):
    kinds = set()
    for leaf in _leaves(value):
        if not isinstance(leaf, str):
            kinds.add("opaque")
        elif INTEGER_RE.fullmatch(leaf):
            kinds.add("integer")
        elif NUMBER_RE.fullmatch(leaf):
            kinds.add("decimal")
        else:
            kinds.add("string")
    if not kinds:
        return "empty"
    if kinds <= {"integer", "decimal"}:
        return "decimal" if "decimal" in kinds else "integer"
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def parse_project(xml_path: Path | str) -> list[dict]:
    """Return all persistent variables with scope, exact values and provenance.

    Each record contains `scope`, `name`, `value`, `source`, and serialization
    metadata. IDs distinguish scene-global variables from sprite-local ones.
    """
    path = Path(xml_path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise SnapImportError(f"{path.name}: invalid XML: {exc}") from exc
    require(root.tag == "project", "Expected a Snap project root")
    result = []

    def walk(element, xpath, scene=None, sprite=None, owner="project"):
        if element.tag == "scene":
            scene, sprite, owner = element.get("name", ""), None, "scene"
        elif element.tag == "stage":
            owner = "stage"
        elif element.tag == "sprite":
            sprite, owner = element.get("name", ""), "sprite"
        if element.tag == "variables":
            for index, variable in enumerate(element, 1):
                require(variable.tag == "variable", "Unexpected persistent variable tag")
                require(len(variable) == 1, "Persistent variable must have exactly one value")
                name = variable.get("name")
                require(name is not None, "Unnamed persistent variable")
                scope = {"kind": owner, "scene": scene, "sprite": sprite}
                variable_path = f"{xpath}/variable[{index}]"
                value = parse_value(variable[0])
                result.append({
                    "variable_id": f"{owner}:{scene or ''}:{sprite or ''}:{name}",
                    "scope": scope, "name": name, "value": value,
                    "source": {"file": path.name, "xml_path": variable_path,
                               "snap_value_id": variable[0].get("id")},
                    "serialization": {"tag": variable[0].tag,
                                      "attributes": dict(variable[0].attrib),
                                      "raw_text": variable[0].text or "",
                                      "xml": ET.tostring(variable[0], encoding="unicode")},
                })
            return
        counters = {}
        for child in element:
            counters[child.tag] = counters.get(child.tag, 0) + 1
            walk(child, f"{xpath}/{child.tag}[{counters[child.tag]}]", scene, sprite, owner)

    walk(root, "/project[1]")
    ids = [r["variable_id"] for r in result]
    require(len(ids) == len(set(ids)), "Duplicate persistent variable within a scope")
    return result


def _manifest(variables):
    return [{
        "variable_id": row["variable_id"], "scope": row["scope"], "name": row["name"],
        "type": "list" if isinstance(row["value"], list) else "scalar" if isinstance(row["value"], str) else "opaque",
        "dimensions": _dimensions(row["value"]),
        "leaf_element_count": sum(1 for _ in _leaves(row["value"])),
        "dtype": _dtype(row["value"]), "storage_dtype": "lexical_string",
        "populated": any(leaf != "" for leaf in _leaves(row["value"])),
        "source": row["source"],
        "runtime_role": "archive_only_learned_table" if row["name"] == "Rarity Tokens" else "preserved_source",
    } for row in variables]


def _integer(text, label, *, minimum=None, maximum=None):
    require(isinstance(text, str) and INTEGER_RE.fullmatch(text), f"{label}: expected integer, got {text!r}")
    value = int(text)
    require(minimum is None or value >= minimum, f"{label}: below minimum {minimum}")
    require(maximum is None or value <= maximum, f"{label}: above maximum {maximum}")
    return value


def _number(text, label):
    require(isinstance(text, str) and NUMBER_RE.fullmatch(text), f"{label}: expected finite number, got {text!r}")
    value = int(text) if INTEGER_RE.fullmatch(text) else float(text)
    require(math.isfinite(value), f"{label}: non-finite number")
    return value


def _expect_length(value, count, label):
    require(isinstance(value, list), f"{label}: expected a list")
    require(len(value) == count, f"{label}: expected {count} entries, got {len(value)}")
    return value


# User (2026-09-30): Academy Student was nerfed to 30%; True Prophet's observed ability is an on-death
# grant that dodges the next attack, lethal or not. Token-efficient, existing tokens only.
# Red support pairs whose descriptions are swapped in the source (Stormcaller/Khione, Dragon King/Avian King, Myths/Gamer,
# Elohim/Mangeka, Dinosaur King/Desmond Of Despair), per the binder videos IMG_0350-0354.
# Border rarities that break the multiplicative pattern of combined borders (user deck screenshots, 2026-10-02).
BORDER_RARITY_CORRECTIONS = {6: ("RuPl", 1_000_000, 10_000_000), 12: ("GaCrPl", 10_000_000_000_000, 1_000_000_000_000)}
RED_SUPPORT_DESCRIPTION_SWAPS = ((4, 5), (13, 15), (20, 21), (22, 23), (26, 27))
RED_SUPPORT_TEXT_FIXES = {19: ("+22 Stats. +36% for Era 2 cards.", "+22% Stats. +36% for Era 2 cards.")}

USER_DESCRIPTION_EDITS = {
    110: ("deal 2 x damage, gain extra turn, and + 20 % stats on kill", "deal 1.5 x damage, gain extra turn, and + 20 % stats on kill",
          "User (2026-10-06): Shuten-dōji was nerfed from 2x to 1.5x damage (DaddyDrago's engine already deals 1.5x)"),
    87: ("each enemy loses hp equal to 40 % of damage dealt", "each enemy loses hp equal to 30 % of damage dealt",
         "User: Academy Student was nerfed to 30 %"),
    163: ("the next card will dodge a lethal attack", "on death, the next card will dodge an attack",
          "User: True Prophet's grant fires on death and dodges the next attack, lethal or not"),
    179: ("attacks on entry; attacks reduce the target's damage by 30 %", "attacks on entry; attacks reduce the target's damage by 10 %",
          "User: Slum Dweller was nerfed to 10 % (IMG_0330: Malik 655,360 -> 589,824 -> 530,842)"),
    285: ("on entry, the enemy is stunned in 3 turns, both active cards die", "on entry, the enemy is stunned; in 3 turns, both active cards die",
          "Card text in IMG_0330/IMG_0341: 'On entry, the enemy is stunned. In 3 turns, both active cards die.'"),
}


def _split_tokens(text):
    matches = list(TOKEN_RE.finditer(text))
    tokens = [m.group(0) for m in matches]
    gaps = []
    end = 0
    for match in matches:
        gaps.append(text[end:match.start()])
        end = match.end()
    gaps.append(text[end:])
    require(all(not gap or gap.isspace() for gap in gaps), "Tokenizer discarded non-whitespace")
    return tokens, gaps


def _audit_source_tokens(vocabulary, matrix, refined):
    """Audit the literal stored mapping; never infer an offset repair."""
    _expect_length(matrix, len(refined), "Legacy token rows")
    out_of_range, mismatch_rows, boundary_rows = [], [], []
    all_ids = []
    exact_count = 0
    for index in range(len(refined)):
        row = matrix[index]
        require(isinstance(row, list), f"Legacy token row {index + 1}: expected list")
        ids = [_integer(t, f"Legacy token row {index + 1}") for t in row]
        all_ids.extend(ids)
        if len(ids) < 2 or ids[0] != 1 or ids[-1] != 2:
            boundary_rows.append(index + 1)
        unknown = [{"position_1": j + 1, "token_id": t} for j, t in enumerate(ids)
                   if t < 1 or t > len(vocabulary)]
        if unknown:
            out_of_range.append({"card_id": index + 1, "tokens": unknown})
        decoded = [vocabulary[t - 1] if 1 <= t <= len(vocabulary) else f"<OUT_OF_RANGE:{t}>" for t in ids[1:-1]]
        expected, _ = _split_tokens(refined[index])
        if decoded == expected:
            exact_count += 1
        else:
            mismatch_rows.append({"card_id": index + 1, "expected_tokens": expected,
                                  "decoded_source_tokens": decoded})
    valid = not (out_of_range or mismatch_rows or boundary_rows)
    return {
        "status": "valid" if valid else "quarantined_invalid",
        "source_vocabulary_size": len(vocabulary),
        "source_boundary_tokens": vocabulary[:2],
        "observed_token_min": min(all_ids) if all_ids else None,
        "observed_token_max": max(all_ids) if all_ids else None,
        "exact_lexical_reconstruction_count": exact_count,
        "mismatched_row_count": len(mismatch_rows), "mismatched_rows": mismatch_rows,
        "out_of_range_rows": out_of_range, "boundary_error_rows": boundary_rows,
        "repair_policy": "Literal old XML mapping remains archived. Canonical tokens follow the new CSVs with the user's rd deletion/reindex and row119 wording edit.",
    }


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _import_authorized_tokens(vocabulary, matrix, refined, corrections):
    """Apply the user's rd deletion to IDs, preserving the supplied tokenization."""
    require(len(vocabulary) == 438 and "rd" not in vocabulary, "Expected the user-supplied vocabulary with rd removed")
    require(len(set(vocabulary)) == len(vocabulary), "Duplicate supplied vocabulary entries")
    rows, padded, remapped_count = [], [], 0
    specials = {"PAD": 0, "BOS": 439, "CARD": 440}
    for index, source_row in enumerate(matrix):
        _expect_length(source_row, 32, f"Supplied token row {index+1}")
        source_ids = [_integer(t, f"Supplied token row {index+1}", minimum=0, maximum=439) for t in source_row]
        require(248 not in source_ids, "Deleted rd token is still used; cannot shift its ID silently")
        first_pad = source_ids.index(0) if 0 in source_ids else 32
        require(all(t == 0 for t in source_ids[first_pad:]), "Padding must be a trailing suffix")
        ids = [t-1 if t > 248 else t for t in source_ids]
        remapped_count += sum(t > 248 for t in source_ids)
        matches = list(SOURCE_TOKEN_RE.finditer(refined[index]))
        decoded = [vocabulary[t-1] for t in ids[:first_pad]]
        require(decoded == [m.group() for m in matches], f"Supplied tokens do not reconstruct canonical lexemes for card {index+1}")
        # Preserve ignored standalone possessive apostrophes in sidecar separators.
        # These separators are evidence, not additional neural-network tokens.
        separators, end = [], 0
        for match in matches:
            separators.append(refined[index][end:match.start()])
            end = match.end()
        separators.append(refined[index][end:])
        rows.append({"card_id": index+1, "token_ids": [specials["BOS"]] + ids[:first_pad] + [specials["CARD"]],
                     "separators": separators, "lexical_text": " ".join(decoded)})
        padded.append(ids)
    vocab = [{"token_id": 0, "token": "<PAD>", "kind": "special", "source_token_id": None}]
    vocab += [{"token_id": i+1, "token": word, "kind": "lexical", "source_token_id": i+1 if i+1<248 else i+2}
              for i, word in enumerate(vocabulary)]
    vocab += [{"token_id": specials[name], "token": f"<{name}>", "kind": "special", "source_token_id": None} for name in ("BOS", "CARD")]
    corrections.append({"kind": "user_authorized_rd_deletion_reindex", "removed_legacy_lexical_id": 248,
                        "rule": "Leave PAD and IDs below 248 unchanged; decrement IDs above 248 by one",
                        "changed_token_occurrences": remapped_count, "vocabulary_order_preserved": True,
                        "source": "user_2026-09-27/data-6.csv + data-7.csv", "lexical_rows_validated": 289})
    return {"tokenizer": {"pattern": SOURCE_TOKEN_PATTERN, "source": "user supplied untampered token CSVs with authorized rd reindex",
                          "padding": "32 columns; PAD=0; runtime BOS/CARD enclose real tokens before padding",
                          "separators": "Exact text gaps preserve whitespace and ignored standalone possessive apostrophes"},
            "lexical_count": 438, "special_ids": specials, "vocabulary": vocab, "rows": rows, "padded_matrix": padded}


def _write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not records:
            return
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(v, (dict, list)) else v for k, v in record.items()})


def import_project(raw_dir: Path | str = Path("data/raw"), output_dir: Path | str = Path("data/clean")) -> dict:
    """Validate all data before writing deterministic, reviewable clean exports."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    require(not output_dir.resolve().is_relative_to(raw_dir.resolve()), "Output directory must not be inside raw source directory")
    xml_path = raw_dir / "Cards-2.xml"
    require(xml_path.is_file(), f"Missing source: {xml_path}")
    variables = parse_project(xml_path)
    by_name = {}
    for row in variables:
        by_name.setdefault(row["name"], []).append(row)

    def record(name):
        require(name in by_name and len(by_name[name]) == 1, f"Missing or ambiguous source variable {name!r}")
        return by_name[name][0]

    def value(name):
        return record(name)["value"]

    def ref(name, item=None):
        return dict(record(name)["source"], variable=name, **({"item_1": item} if item is not None else {}))

    source_files, csv_sources = [], []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue  # Finder metadata such as .DS_Store is not a source.
        blob = path.read_bytes()
        relative_name = path.relative_to(raw_dir).as_posix()
        source_files.append({"file": relative_name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
        if path.suffix.lower() == ".csv":
            csv_sources.append({"file": relative_name, "utf8_bom": blob.startswith(b"\xef\xbb\xbf"),
                                "records": read_csv_text(blob.decode("utf-8-sig"), label=path.name)})
    csv_by_name = {row["file"]: row for row in csv_sources}

    def csv_vector(name):
        require(name in csv_by_name, f"Missing required CSV: {name}")
        rows = csv_by_name[name]["records"]
        require(len(rows) == 1, f"{name}: expected one CSV vector row, got {len(rows)}")
        return list(rows[0])

    corrections = []

    def trailing_duplicate(name, expected):
        source = value(name)
        require(isinstance(source, list), f"{name}: expected list")
        if len(source) == expected:
            return list(source)
        require(len(source) == expected + 1 and source[-1] == source[-2],
                f"{name}: expected {expected} entries or one provably duplicate trailing entry; got {len(source)}")
        corrections.append({"kind": "remove_proven_trailing_duplicate", "source": ref(name),
                            "original_count": len(source), "clean_count": expected,
                            "removed_item_1": expected + 1, "duplicate_of_item_1": expected,
                            "original_value": source[-1], "reason": "Identical final two descriptions and matching entity count"})
        return source[:-1]

    names = _expect_length(value("Cards"), 289, "Cards")
    rarities = _expect_length(value("Card Rarities"), 289, "Card Rarities")
    source_refined = _expect_length(value("data-5_refined"), 289, "Refined descriptions")
    refined = list(source_refined)
    require(refined[119].endswith("awaken on the 3rd turn"), "Unexpected source wording for user-authorized card index119 edit")
    refined[119] = refined[119].replace("awaken on the 3rd turn", "awaken after 2 turns")
    corrections.append({"kind": "user_authorized_description_edit", "card_id": 120, "card_index_0": 119,
                        "name": names[119], "original_value": source_refined[119], "clean_value": refined[119],
                        "reason": "User requests token-efficient after 2 turns wording and removal of rd"})
    for index, (source_text, clean_text, reason) in USER_DESCRIPTION_EDITS.items():
        require(refined[index] == source_text, f"Unexpected source wording for user-authorized card index{index} edit")
        refined[index] = clean_text
        corrections.append({"kind": "user_authorized_description_edit", "card_id": index + 1, "card_index_0": index,
                            "name": names[index], "original_value": source_refined[index], "clean_value": clean_text,
                            "reason": reason})
    original = trailing_duplicate("Card Descriptions", 289)
    weather = csv_vector("data-10.csv")
    require(len(weather) == 290 and weather[-1] == "1", "data-10.csv: expected exactly 290 values ending in authorized removable '1'")
    corrections.append({"kind": "user_authorized_trailing_weather_removal", "source": {"file": "data-10.csv", "record_1": 1},
                        "removed_item_1": 290, "original_value": "1", "original_count": 290, "clean_count": 289,
                        "reason": "User explicitly requested dropping exactly the final 1"})
    weather = weather[:-1]
    old_weather = value("Card Weather")
    corrections.append({"kind": "source_precedence", "field": "weather_id",
                        "source": {"file": "data-10.csv"}, "superseded_source": ref("Card Weather"),
                        "reason": "Latest user-supplied weather mapping supersedes embedded mapping",
                        "original_value": old_weather, "clean_value": weather})
    damage = _expect_length(csv_vector("data-9.csv"), 289, "Card modifiers")
    corrections.append({"kind": "source_precedence", "field": "card_modifier",
                        "source": {"file": "data-9.csv"}, "superseded_source": ref("Card Damage Modifiers"),
                        "original_value": value("Card Damage Modifiers"),
                        "reason": "User-supplied 289-element vector replaces scalar placeholder"})
    cards, descriptions = [], []
    for i in range(289):
        cards.append({"card_id": i + 1, "name": names[i],
                      "rarity": _integer(rarities[i], f"Card {i+1} rarity", minimum=1), "rarity_lexical": rarities[i],
                      "weather_id": _integer(weather[i], f"Card {i+1} weather", minimum=1, maximum=13),
                      "weather_id_lexical": weather[i],
                      "card_modifier": _number(damage[i], f"Card {i+1} card modifier"),
                      "card_modifier_lexical": damage[i], "packs": None, "classes": None,
                      "source": {"name": ref("Cards", i+1), "rarity": ref("Card Rarities", i+1),
                                 "weather": {"file": "data-10.csv", "record_1": 1, "column_1": i+1},
                                 "card_modifier": {"file": "data-9.csv", "record_1": 1, "column_1": i+1}}})
        descriptions.append({"card_id": i + 1, "original": original[i], "refined": refined[i],
                             "source": {"original": ref("Card Descriptions", i+1), "refined": ref("data-5_refined", i+1)}})
    supports = []
    for color, count in (("Red", 28), ("Blue", 15)):
        labels = _expect_length(value(f"Support Cards ({color})"), count, f"{color} supports")
        desc_name = f"Support Card Descriptions ({color})"
        text = trailing_duplicate(desc_name, count) if color == "Blue" else _expect_length(value(desc_name), count, desc_name)
        text = list(text)
        if color == "Red":
            # IMG_0350-0354 (the user's support binder): these source descriptions sit under the wrong names.
            for a, b in RED_SUPPORT_DESCRIPTION_SWAPS:
                text[a - 1], text[b - 1] = text[b - 1], text[a - 1]
                corrections.append({"kind": "video_verified_support_description_swap", "color": "red", "support_ids": [a, b],
                                    "reason": "IMG_0350-0354: support binder texts"})
            for sid, (old_text, new_text) in RED_SUPPORT_TEXT_FIXES.items():
                require(text[sid - 1] == old_text, f"Unexpected red support {sid} description")
                text[sid - 1] = new_text
                corrections.append({"kind": "video_verified_support_description_edit", "color": "red", "support_id": sid,
                                    "original_value": old_text, "clean_value": new_text, "reason": "IMG_0350: support binder text"})
        for i in range(count):
            supports.append({"support_id": i + 1, "color": color.lower(), "name": labels[i], "description": text[i],
                             "source": {"name": ref(f"Support Cards ({color})", i+1), "description": ref(desc_name, i+1)}})
    borders = []
    border_names = _expect_length(value("Borders"), 16, "Borders")
    border_rarities = _expect_length(value("Border Rarities"), 16, "Border Rarities")
    for i in range(16):
        borders.append({"border_id": i+1, "name": border_names[i],
                        "rarity": _integer(border_rarities[i], f"Border {i+1} rarity", minimum=1),
                        "rarity_lexical": border_rarities[i],
                        "source": {"name": ref("Borders", i+1), "rarity": ref("Border Rarities", i+1)}})
    for border_id, (name, original, corrected) in BORDER_RARITY_CORRECTIONS.items():
        row = borders[border_id - 1]
        require(row["name"] == name and row["rarity"] == original, f"Border {border_id} source changed; review rarity correction")
        row["rarity"], row["rarity_lexical"] = corrected, str(corrected)
        corrections.append({"kind": "user_evidence_border_rarity", "border_id": border_id, "name": name,
                            "original_value": original, "clean_value": corrected,
                            "reason": "Combined borders multiply their parts (Ru 1e5 x Pl 1e2; Ga 1e6 x Cr 1e4 x Pl 1e2): the user's "
                                      "deck shows A0-ON1 GaCrPl at 66qd and Deus Ex/Anubis/Set RuPl at 1e7 x card rarity, and the "
                                      "stats-sorted deck screenshot and an enemy GaCrPl Malik (2,621,440 HP) fit only these values"})
    weather_names = _expect_length(value("Weathers"), 13, "Weathers")
    multipliers = _expect_length(value("Weather Multiplers"), 13, "Weather Multiplers")
    weathers = [{"weather_id": i+1, "name": weather_names[i],
                 "multiplier": _number(multipliers[i], f"Weather {i+1} multiplier"), "multiplier_lexical": multipliers[i],
                 "source": {"name": ref("Weathers", i+1), "multiplier": ref("Weather Multiplers", i+1)}} for i in range(13)]

    source_registry = value("card_simulator_phase_registry")
    require(isinstance(source_registry, list) and len(source_registry) > 1, "Missing registry table")
    header = source_registry[0]
    require(len(header) == len(set(header)), "Duplicate registry columns")
    required_columns = {"handler_id", "card_index_0", "card_row_1", "clause_order", "phase_order", "parameters_json", "full_description"}
    require(required_columns <= set(header), "Missing required registry columns")
    registry = []
    for row_number in range(1, len(source_registry)):
        row = source_registry[row_number]
        require(len(row) == len(header), f"Registry row {row_number+1}: column count mismatch")
        entry = dict(zip(header, row))
        for column in ("handler_id", "card_index_0", "card_row_1", "clause_order", "phase_order"):
            entry[column] = _integer(entry[column], f"Registry row {row_number+1} {column}", minimum=0)
        require(entry["card_row_1"] == entry["card_index_0"] + 1, f"Registry row {row_number+1}: inconsistent card IDs")
        card_id = entry["card_row_1"]
        require(1 <= card_id <= 289, f"Registry row {row_number+1}: card out of range")
        require(entry["full_description"] == source_refined[card_id - 1], f"Registry row {row_number+1}: description/card alignment mismatch")
        entry["source_full_description"] = entry["full_description"]
        entry["full_description"] = refined[card_id-1]
        try:
            parameters = json.loads(entry["parameters_json"])
        except json.JSONDecodeError as exc:
            raise SnapImportError(f"Registry row {row_number+1}: invalid parameter JSON") from exc
        require(isinstance(parameters, dict), f"Registry row {row_number+1}: parameters must be object")
        entry.update(card_id=card_id, parameters=parameters, verification_status="unverified",
                     source=ref("card_simulator_phase_registry", row_number+1))
        registry.append(entry)
    require(len({r["handler_id"] for r in registry}) == len(registry), "Duplicate registry handler ID")
    require({r["card_id"] for r in registry} == set(range(1, 290)), "Registry does not cover all 289 cards")

    # Every training embedding remains available in the archive, never used as game rules.
    for name, count in (("Card Tokens", 289), ("Support Tokens (Red)", 28), ("Support Tokens (Blue)", 15), ("Rarity Tokens", 32768)):
        table = _expect_length(value(name), count, name)
        for i, row in enumerate(table, 1):
            _expect_length(row, 2, f"{name} row {i}")
            for cell in row:
                _number(cell, f"{name} row {i}")
    source_vocab = value("data-5_refined_vocab_ordered")
    source_tokens = value("data-5_refined_tokenized_2d")
    token_audit = _audit_source_tokens(source_vocab, source_tokens, source_refined)
    new_matrix_key = "user_2026-09-27/data-6.csv"
    new_vocab_key = "user_2026-09-27/data-7.csv"
    if new_matrix_key in csv_by_name or new_vocab_key in csv_by_name:
        require(new_matrix_key in csv_by_name and new_vocab_key in csv_by_name, "Both replacement token files are required")
        new_matrix = csv_by_name[new_matrix_key]["records"]
        new_vocab = csv_vector(new_vocab_key)
        _expect_length(new_matrix, 289, "New token matrix")
        unpadded = []
        for i, row in enumerate(new_matrix, 1):
            _expect_length(row, 32, f"New token row {i}")
            ids = [_integer(t, f"New token row {i}", minimum=0) for t in row]
            first_pad = ids.index(0) if 0 in ids else len(ids)
            require(all(t == 0 for t in ids[first_pad:]), f"New token row {i}: non-trailing padding")
            unpadded.append(ids[:first_pad])
        token_audit["new_user_files"] = {
            "matrix_file": new_matrix_key, "vocabulary_file": new_vocab_key,
            "matrix_shape": [289, 32], "vocabulary_size": len(new_vocab),
            "max_nonpadding_id": max(t for row in unpadded for t in row),
            "missing_used_vocabulary_ids": sorted({t for row in unpadded for t in row if t > len(new_vocab)}),
            "unused_id_gap": sorted(set(range(1, max(t for row in unpadded for t in row) + 1)) - {t for row in unpadded for t in row}),
            "matches_saved_tokens_without_specials": unpadded == [[int(t)-2 for t in row[1:-1]] for row in source_tokens],
            "matches_saved_lexical_vocabulary": new_vocab == source_vocab[2:],
            "status": "quarantined_pending_vocabulary_alignment",
        }
    require(new_matrix_key in csv_by_name and new_vocab_key in csv_by_name, "Latest user token CSVs are required")
    for index in USER_DESCRIPTION_EDITS:
        # Re-encode user-edited rows with existing vocabulary only (source-ID space: IDs from 248 shift up by one).
        words = [m.group() for m in SOURCE_TOKEN_RE.finditer(refined[index])]
        require(all(w in new_vocab for w in words), f"Card {index+1} edit needs a token outside the vocabulary")
        ids = [new_vocab.index(w) + 1 for w in words]
        ids = [t + 1 if t >= 248 else t for t in ids]
        require(len(ids) <= 32, f"Card {index+1} edit exceeds 32 tokens")
        corrections.append({"kind": "user_authorized_token_row_edit", "card_id": index + 1,
                            "original_row": list(new_matrix[index]), "clean_row": ids + [0] * (32 - len(ids))})
        new_matrix[index] = [str(t) for t in ids] + ["0"] * (32 - len(ids))
    canonical_tokens = _import_authorized_tokens(new_vocab, new_matrix, refined, corrections)
    token_audit["new_user_files"]["status"] = "resolved_by_user_authorized_rd_reindex_and_wording_edit"
    token_audit["resolution"] = {"deleted_unused_lexical_id": 248, "lexical_ids": [1,438], "all_rows_validated": True,
                                "description_edit_card_id": 120}
    corrections.append({"kind": "quarantine_source_token_matrix", "source": ref("data-5_refined_tokenized_2d"),
                        "original_value": "archive/persistent_variables.json", "status": token_audit["status"],
                        "reason": "Stored token IDs fail range/reconstruction checks; no source vocabulary offset is guessed"})

    entry_attack_annotations = []
    if "data-8.csv" in csv_by_name:
        entry_names = csv_vector("data-8.csv")
        for column, name in enumerate(entry_names, 1):
            matches = [i+1 for i, card_name in enumerate(names) if card_name == name]
            require(len(matches) == 1, f"data-8.csv column {column}: unknown or ambiguous card name {name!r}")
            entry_attack_annotations.append({"card_id": matches[0], "name": name,
                                             "annotation": "user_supplied_entry_attack_list", "verification_status": "unverified",
                                             "source": {"file": "data-8.csv", "record_1": 1, "column_1": column}})

    from .metadata import annotate
    packs, metadata = annotate(cards, descriptions, raw_dir)
    from .data_corrections import apply_user_corrections
    apply_user_corrections(cards, corrections)
    from .card_keys import assign_identity_keys
    for card, key in zip(cards, assign_identity_keys([card["name"] for card in cards]), strict=True):
        card["identity_key"] = key  # Permanent: unaffected by cards inserted into earlier packs.
    correction_code = Path(__file__).with_name("data_corrections.py").read_bytes()
    source_files.append({"file": "../../card_engine/data_corrections.py", "bytes": len(correction_code),
                         "sha256": hashlib.sha256(correction_code).hexdigest(), "role": "user_authorized_stat_metadata_corrections"})
    annotation_file = raw_dir.parent / "annotations/sourced_classes.json"
    if annotation_file.exists():
        blob = annotation_file.read_bytes()
        source_files.append({"file": "../annotations/sourced_classes.json", "bytes": len(blob),
                             "sha256": hashlib.sha256(blob).hexdigest(), "role": "curated_web_class_evidence"})
    metadata_code = Path(__file__).with_name("metadata.py").read_bytes()
    source_files.append({"file": "../../card_engine/metadata.py", "bytes": len(metadata_code),
                         "sha256": hashlib.sha256(metadata_code).hexdigest(), "role": "curated_screenshot_annotations"})
    unresolved = [
        {"id": "gameplay_class_membership", "status": "inferred", "affected_cards": 289,
         "detail": "Packs follow supplied screenshots; visual classes are candidates and sourced positives remain version-sensitive. Never silently use inferred classes as confirmed simulator predicates."},
        {"id": "registry_semantics", "status": "unverified", "affected_handlers": len(registry),
         "detail": "All proposed handlers need true-logic verification, including source confidence HIGH. Source confidence is preserved, not trusted."},
        {"id": "original_and_refined_text_divergence", "status": "preserved",
         "detail": "Original and refined descriptions retained independently; no typo or semantic rewriting applied."},
        {"id": "legacy_token_matrix", "status": token_audit["status"],
         "detail": "Old mapping is archived. Current input follows the user's supplied CSVs and authorized rd reindex/after 2 turns edit; all 289 rows validate."},
        {"id": "rarity_tokens", "status": "archive_only", "dimensions": [32768, 2],
         "detail": "Learned table preserved losslessly. It is not a runtime rarity-to-stat rule."},
    ]
    dataset = {"schema_version": SCHEMA_VERSION, "simulation_ready": False,
               "cards": cards, "descriptions": descriptions, "supports": supports,
               "borders": borders, "weathers": weathers, "registry": registry,
               "entry_attack_annotations": entry_attack_annotations, "tokens": canonical_tokens, "packs": packs, "card_metadata": metadata,
               "unresolved": unresolved}
    validate_dataset(dataset)
    manifest = _manifest(variables)
    report = {"schema_version": SCHEMA_VERSION, "status": "clean_data_validated_rules_unverified", "simulation_ready": False,
              "counts": {"persistent_variables": len(variables), "cards": len(cards), "supports_red": 28,
                         "supports_blue": 15, "borders": len(borders), "weathers": len(weathers),
                         "registry_handlers": len(registry), "entry_attack_annotations": len(entry_attack_annotations),
                         "lexical_tokens": canonical_tokens["lexical_count"]},
              "checks": {"raw_files_hashed": True, "persistent_variables_archived": True,
                         "stable_1_based_snap_ids": True, "entity_alignment": True,
                         "canonical_token_ranges": True, "exact_refined_text_reconstruction": True,
                         "source_registry_description_alignment": True},
              "source_tokens": {k: v for k, v in token_audit.items() if k not in ("mismatched_rows",)},
              "unresolved": unresolved, "correction_count": len(corrections), "source_files": source_files}

    # No writes occur until the complete dataset passes all invariants.
    _write_json(output_dir / "archive/persistent_variables.json", {"schema_version": SCHEMA_VERSION, "variables": variables})
    _write_json(output_dir / "archive/csv_sources.json", csv_sources)
    _write_json(output_dir / "manifest.json", {"sources": source_files, "variables": manifest})
    _write_csv(output_dir / "manifest.csv", manifest)
    _write_json(output_dir / "schema.json", DATASET_SCHEMA)
    _write_json(output_dir / "dataset.json", dataset)
    for name in ("cards", "descriptions", "supports", "borders", "weathers", "registry", "entry_attack_annotations", "packs", "card_metadata"):
        _write_json(output_dir / f"{name}.json", dataset[name])
        _write_csv(output_dir / f"{name}.csv", dataset[name])
    _write_json(output_dir / "tokens.json", canonical_tokens)
    _write_csv(output_dir / "tokens.csv", canonical_tokens["rows"])
    with (output_dir / "description_tokens_32.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(canonical_tokens["padded_matrix"])
    with (output_dir / "vocabulary_ordered.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow([r["token"] for r in canonical_tokens["vocabulary"] if r["kind"] == "lexical"])
    _write_json(output_dir / "vocabulary.json", canonical_tokens["vocabulary"])
    _write_csv(output_dir / "vocabulary.csv", canonical_tokens["vocabulary"])
    _write_json(output_dir / "token_audit.json", token_audit)
    _write_json(output_dir / "corrections.json", corrections)
    _write_json(output_dir / "validation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/clean"))
    args = parser.parse_args(argv)
    try:
        report = import_project(args.raw_dir, args.output_dir)
    except (SnapImportError, OSError, UnicodeError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "source_token_status": report["source_tokens"]["status"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
