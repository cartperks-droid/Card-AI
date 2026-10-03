"""The restricted deck: which cards and borders the player base can realistically own, for team building.

Rules (user, 2026-10-02): every card except seasonal and Limited packs, plus a few Limited cards that are still
widely owned (Fate Seamstress, Eclipseborn Luminant, Supreme Ozzy, Frank, The Broken One); every border except
the perfect gem (GaRuCrPl). Cards and borders can be added back or removed per card later. Every change rewrites
data/restricted_deck.json and docs/restricted_deck.md.

    python -m card_engine.restricted show
    python -m card_engine.restricted add-card "Santa Claus"                 # e.g. a seasonal card back in
    python -m card_engine.restricted remove-card Pandora
    python -m card_engine.restricted enable-border "Malik" --border GaRuCrPl
    python -m card_engine.restricted disable-border "Odin" --border GaRuCr
    python -m card_engine.restricted reset-card Odin                        # back to the default rules
"""

import argparse
import json
from pathlib import Path

from .catalog import load_catalog
from .deck import BORDERS, _border, _match

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_FILE = ROOT / "data" / "restricted_deck.json"
RESTRICTED_DOC = ROOT / "docs" / "restricted_deck.md"
DEFAULT = {
    "excluded_packs": ["Christmas", "Halloween", "Halloween 2025", "Limited"],
    # Limited cards widely owned anyway (user); search with --no-limited to leave them out too.
    "limited_exceptions": [240, 228, 231, 227, 230],  # Fate Seamstress, Eclipseborn Luminant, Supreme Ozzy, Frank, The Broken One
    "disabled_borders": [16],  # the perfect gem, GaRuCrPl
    "added_cards": [],  # cards added back despite the pack rules
    "removed_cards": [],
    "border_overrides": {},  # card id -> {"enable": [borders], "disable": [borders]}
}


def load(path=RESTRICTED_FILE):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else json.loads(json.dumps(DEFAULT))


def save(rules, catalog, path=RESTRICTED_FILE, doc=RESTRICTED_DOC):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rules, indent=1) + "\n")
    Path(doc).write_text(render(rules, catalog))


def cards(rules, catalog, *, limited=True):
    """Card IDs in the restricted deck."""
    out = []
    for card in catalog.cards:
        if card.id in rules["removed_cards"]:
            continue
        excluded = card.packs[0] in rules["excluded_packs"]
        exception = card.id in rules["limited_exceptions"] and limited
        if card.id in rules["added_cards"] or not excluded or exception:
            out.append(card.id)
    return out


def borders(rules, card_id):
    override = rules["border_overrides"].get(str(card_id), {})
    enabled = {b for b in range(1, 17) if b not in rules["disabled_borders"]}
    return sorted((enabled | set(override.get("enable", []))) - set(override.get("disable", [])))


def entries(rules, catalog, *, limited=True):
    """(card, border) pairs available to the player base."""
    return [(card, border) for card in cards(rules, catalog, limited=limited) for border in borders(rules, card)]


def render(rules, catalog):
    names = {c.id: c.name for c in catalog.cards}
    included = cards(rules, catalog)
    lines = ["# Restricted deck (player-base availability)", "",
             "Cards and borders the player base can realistically own; team searches in `restricted` and `semi` mode draw "
             "from it. Edit with `python -m card_engine.restricted` (`--help`); this page is regenerated after every change.", "",
             f"- **Cards:** {len(included)}, with every pack except {', '.join(rules['excluded_packs'])}.",
             f"- **Limited cards kept** (`--no-limited` leaves them out): {', '.join(names[i] for i in rules['limited_exceptions'])}.",
             f"- **Borders disabled for every card:** {', '.join(BORDERS[b - 1] for b in rules['disabled_borders']) or 'none'}.",
             f"- **Cards added back:** {', '.join(names[i] for i in rules['added_cards']) or 'none'}.",
             f"- **Cards removed:** {', '.join(names[i] for i in rules['removed_cards']) or 'none'}.", "",
             "## Per-card border changes", ""]
    if rules["border_overrides"]:
        lines += ["| Card | Enabled | Disabled |", "|---|---|---|"]
        for card, override in sorted(rules["border_overrides"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {names[int(card)]} | {', '.join(BORDERS[b - 1] for b in override.get('enable', [])) or '-'} | "
                         f"{', '.join(BORDERS[b - 1] for b in override.get('disable', [])) or '-'} |")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m card_engine.restricted", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="print the restricted deck rules")
    for name, text in (("add-card", "add a card back"), ("remove-card", "remove a card"),
                       ("reset-card", "undo every change to one card")):
        commands.add_parser(name, help=text).add_argument("card")
    for name, text in (("enable-border", "allow a border for one card"), ("disable-border", "forbid a border for one card")):
        p = commands.add_parser(name, help=text)
        p.add_argument("card")
        p.add_argument("--border", required=True)
    args = parser.parse_args(argv)
    catalog = load_catalog()
    rules = load()
    if args.command == "show":
        print(render(rules, catalog), end="")
        return
    card_id, name = _match(args.card, [(c.id, c.name) for c in catalog.cards], "Card")
    key = str(card_id)
    if args.command == "add-card":
        rules["removed_cards"] = [i for i in rules["removed_cards"] if i != card_id]
        if card_id not in cards(rules, catalog):
            rules["added_cards"].append(card_id)
        message = f"{name} is in the restricted deck"
    elif args.command == "remove-card":
        rules["added_cards"] = [i for i in rules["added_cards"] if i != card_id]
        if card_id not in rules["removed_cards"]:
            rules["removed_cards"].append(card_id)
        message = f"{name} removed from the restricted deck"
    elif args.command == "reset-card":
        rules["added_cards"] = [i for i in rules["added_cards"] if i != card_id]
        rules["removed_cards"] = [i for i in rules["removed_cards"] if i != card_id]
        rules["border_overrides"].pop(key, None)
        message = f"{name} follows the default rules again"
    else:
        border = _border(args.border)
        override = rules["border_overrides"].setdefault(key, {"enable": [], "disable": []})
        add, drop = ("enable", "disable") if args.command == "enable-border" else ("disable", "enable")
        override[drop] = [b for b in override[drop] if b != border]
        if border not in override[add]:
            override[add].append(border)
        if not override["enable"] and not override["disable"]:
            rules["border_overrides"].pop(key)
        message = f"{name}: {BORDERS[border - 1]} {add}d"
    save(rules, catalog)
    print(message)


if __name__ == "__main__":
    main()
