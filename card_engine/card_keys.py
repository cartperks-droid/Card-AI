"""Permanent card identity keys for the learned card-identity embedding.

Card IDs are positions in the game's card list, so a card inserted into a pack
shifts every later ID. Identity keys never move: they are assigned once per card
name, append-only, and a new card simply takes the next unused key. Keys of
removed cards are retired, never reused. A renamed card gets a new key unless an
alias is recorded in the key file.
"""

import json
from pathlib import Path

KEY_FILE = Path(__file__).resolve().parents[1] / "data" / "annotations" / "card_keys.json"


def _normalize(name):
    return " ".join(name.split()).casefold()


def assign_identity_keys(names, key_file=KEY_FILE):
    """Return one permanent key (>= 1) per name, extending the key file for new cards."""
    key_file = Path(key_file)
    table = json.loads(key_file.read_text()) if key_file.exists() else {"keys": {}, "aliases": {}}
    keys, aliases = table["keys"], table.get("aliases", {})
    if len({_normalize(n) for n in names}) != len(names):
        raise ValueError("Card names must be unique to assign identity keys")
    if len(set(keys.values())) != len(keys):
        raise ValueError("Identity key file has duplicate keys")
    result, next_key = [], max(keys.values(), default=0) + 1
    for name in names:
        normalized = _normalize(name)
        normalized = aliases.get(normalized, normalized)
        if normalized not in keys:
            keys[normalized] = next_key
            next_key += 1
        result.append(keys[normalized])
    table["keys"], table["aliases"] = keys, aliases
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    return result
