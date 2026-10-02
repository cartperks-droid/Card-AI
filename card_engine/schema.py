"""Portable schema for clean card data; Snap IDs are always one-based."""

import math

SCHEMA_VERSION = "1.0.0"

DATASET_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:card-engine:clean-dataset:1",
    "title": "Organized card source data, not a verified battle simulator",
    "type": "object",
    "required": ["schema_version", "simulation_ready", "cards", "descriptions",
                 "supports", "borders", "weathers", "registry", "tokens"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "simulation_ready": {"const": False},
        "cards": {"type": "array", "minItems": 289, "maxItems": 289,
                  "items": {"$ref": "#/$defs/card"}},
        "descriptions": {"type": "array", "minItems": 289, "maxItems": 289,
                         "items": {"$ref": "#/$defs/description"}},
        "supports": {"type": "array", "minItems": 43, "maxItems": 43,
                     "items": {"$ref": "#/$defs/support"}},
        "borders": {"type": "array", "minItems": 16, "maxItems": 16,
                    "items": {"$ref": "#/$defs/border"}},
        "weathers": {"type": "array", "minItems": 13, "maxItems": 13,
                     "items": {"$ref": "#/$defs/weather"}},
        "registry": {"type": "array", "items": {"type": "object",
                     "required": ["card_id", "verification_status", "source"],
                     "properties": {"card_id": {"$ref": "#/$defs/id"},
                                    "verification_status": {"const": "unverified"}}}},
        "tokens": {"type": "object", "required": ["vocabulary", "rows", "special_ids", "tokenizer"]},
    },
    "$defs": {
        "id": {"type": "integer", "minimum": 1},
        "nullable_strings": {"type": ["array", "null"], "items": {"type": "string"}},
        "card": {
            "type": "object", "required": ["card_id", "name", "rarity", "rarity_lexical",
                "weather_id", "card_modifier", "card_modifier_lexical", "packs", "classes", "source"],
            "properties": {"card_id": {"$ref": "#/$defs/id"}, "name": {"type": "string"},
                "rarity": {"type": "integer", "minimum": 1}, "rarity_lexical": {"type": "string"},
                "weather_id": {"type": "integer", "minimum": 1, "maximum": 13},
                "card_modifier": {"type": "number"}, "card_modifier_lexical": {"type": "string"},
                "packs": {"$ref": "#/$defs/nullable_strings"}, "classes": {"$ref": "#/$defs/nullable_strings"},
                "source": {"type": "object"}},
        },
        "description": {"type": "object", "required": ["card_id", "original", "refined", "source"],
                        "properties": {"card_id": {"$ref": "#/$defs/id"},
                                       "original": {"type": "string"}, "refined": {"type": "string"}}},
        "support": {"type": "object", "required": ["support_id", "color", "name", "description", "source"],
                    "properties": {"support_id": {"$ref": "#/$defs/id"},
                                   "color": {"enum": ["red", "blue"]}, "name": {"type": "string"},
                                   "description": {"type": "string"}}},
        "border": {"type": "object", "required": ["border_id", "name", "rarity", "rarity_lexical", "source"],
                   "properties": {"border_id": {"$ref": "#/$defs/id"}, "name": {"type": "string"},
                                  "rarity": {"type": "integer", "minimum": 1}, "rarity_lexical": {"type": "string"}}},
        "weather": {"type": "object", "required": ["weather_id", "name", "multiplier", "multiplier_lexical", "source"],
                    "properties": {"weather_id": {"$ref": "#/$defs/id"}, "name": {"type": "string"},
                                   "multiplier": {"type": "number"}, "multiplier_lexical": {"type": "string"}}},
    },
}


class SnapImportError(ValueError):
    """Source or clean data violates an explicit import invariant."""


def require(condition, message):
    if not condition:
        raise SnapImportError(message)


def validate_dataset(dataset):
    """Check relational invariants in addition to the exported JSON Schema."""
    require(dataset.get("schema_version") == SCHEMA_VERSION, "Unknown schema version")
    require(dataset.get("simulation_ready") is False, "Unverified registry cannot be simulation-ready")
    for key, id_key, count in (("cards", "card_id", 289), ("descriptions", "card_id", 289),
                               ("borders", "border_id", 16), ("weathers", "weather_id", 13)):
        rows = dataset[key]
        require(len(rows) == count, f"{key}: expected {count} rows; got {len(rows)}")
        require([row[id_key] for row in rows] == list(range(1, count + 1)), f"{key}: unstable Snap IDs")
    weather_ids = {row["weather_id"] for row in dataset["weathers"]}
    for card in dataset["cards"]:
        require(type(card["card_id"]) is int and type(card["rarity"]) is int and card["rarity"] > 0, "Invalid card integer field")
        require(type(card["weather_id"]) is int, "Invalid weather ID type")
        require(type(card["card_modifier"]) in (int, float) and math.isfinite(card["card_modifier"]) and card["card_modifier"] > 0,
                "Card Modifier must be a finite positive number")
        require(card["weather_id"] in weather_ids, f"Card {card['card_id']}: invalid weather")
    for border in dataset["borders"]:
        require(type(border["rarity"]) is int and border["rarity"] > 0, "Invalid border rarity")
    for weather in dataset["weathers"]:
        require(type(weather["multiplier"]) in (int, float) and math.isfinite(weather["multiplier"]) and weather["multiplier"] > 0,
                "Invalid intrinsic weather multiplier")
    require(len(dataset["supports"]) == 43, "Expected exactly 43 support records")
    require(all(r.get("color") in ("red", "blue") for r in dataset["supports"]), "Unknown support namespace")
    for color, count in (("red", 28), ("blue", 15)):
        rows = [r for r in dataset["supports"] if r["color"] == color]
        require([r["support_id"] for r in rows] == list(range(1, count + 1)), f"{color} support IDs")
    for row in dataset["registry"]:
        require(1 <= row["card_id"] <= 289, "Registry card reference out of range")
        require(row["verification_status"] == "unverified", "Registry needs verification")
    tokens = dataset["tokens"]
    require(tokens["special_ids"] == {"PAD": 0, "BOS": 439, "CARD": 440}, "Invalid canonical special IDs")
    vocab = {row["token_id"]: row["token"] for row in tokens["vocabulary"]}
    require(len(tokens["rows"]) == 289, "Token/card alignment mismatch")
    require(len(vocab) == len(tokens["vocabulary"]), "Duplicate token IDs")
    for index, row in enumerate(tokens["rows"]):
        require(row["card_id"] == index + 1, "Token IDs are not aligned")
        ids = row["token_ids"]
        require(len(ids) >= 2, f"Empty or incomplete token sequence: card {index + 1}")
        require(ids[0] == tokens["special_ids"]["BOS"] and ids[-1] == tokens["special_ids"]["CARD"],
                f"Token boundary mismatch: card {index + 1}")
        require(all(i in vocab for i in ids), f"Out-of-range token ID: card {index + 1}")
        gaps = row["separators"]
        require(len(gaps) == len(ids) - 1, f"Whitespace alignment mismatch: card {index + 1}")
        reconstructed = gaps[0] + "".join(vocab[t] + gaps[j + 1] for j, t in enumerate(ids[1:-1]))
        require(reconstructed == dataset["descriptions"][index]["refined"],
                f"Exact token reconstruction failed: card {index + 1}")
    return True
