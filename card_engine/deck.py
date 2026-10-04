"""The player's own deck: the cards and support cards they actually own, with borders, mutations and counts.

Searches read it instead of guessing the collection from screenshots. Every change rewrites the data file
(data/my_deck.json) and the readable copy (docs/my_deck.md).

    python -m card_engine.deck add "Malik The Sovereign" --border RuCrPl
    python -m card_engine.deck add 70 --border GaPl --count 4 --mutation Storm
    python -m card_engine.deck remove Malik --border RuCr
    python -m card_engine.deck remove 70 --border GaPl --mutation Storm --count 1
    python -m card_engine.deck add-support "Desmond Of Despair" --tier Platinum
    python -m card_engine.deck remove-support Desmond --tier Platinum
    python -m card_engine.deck list

The custom pool is a second list in the same format, for suggestions tailored to someone else's collection (the
generator's --pool custom). Put --custom before the command; reset empties it, copy-deck starts it from your deck:

    python -m card_engine.deck --custom reset
    python -m card_engine.deck --custom copy-deck
    python -m card_engine.deck --custom add "Vampire Lord" --border GaPl

Cards and supports can be named (any unambiguous part of the name, case-insensitive) or given by ID.
Borders: none, Pl, Cr, CrPl, Ru, RuPl, RuCr, RuCrPl, Ga, GaPl, GaCr, GaCrPl, GaRu, GaRuPl, GaRuCr, GaRuCrPl.
Support tiers: base, Platinum, Crystal, Ruby, Galaxy (or 1-5). Mutations: None, Storm, Snow, Aurora, Shroud,
Meteor Shower, Time Storm, Eclipse, Virus, Blood Rain, Armageddon, Manga.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from .catalog import load_catalog
from .mutations import MUTATION_NAMES

ROOT = Path(__file__).resolve().parents[1]
DECK_FILE = ROOT / "data" / "my_deck.json"
DECK_DOC = ROOT / "docs" / "my_deck.md"
CUSTOM_FILE = ROOT / "data" / "custom_pool.json"
CUSTOM_DOC = ROOT / "docs" / "custom_pool.md"
BORDERS = ("none", "Pl", "Cr", "CrPl", "Ru", "RuPl", "RuCr", "RuCrPl", "Ga", "GaPl", "GaCr", "GaCrPl",
           "GaRu", "GaRuPl", "GaRuCr", "GaRuCrPl")  # border ID = index + 1
TIERS = ("base", "Platinum", "Crystal", "Ruby", "Galaxy")  # tier = index + 1


def load(path=DECK_FILE):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {"cards": [], "supports": []}


def save(deck, catalog, path=DECK_FILE, doc=DECK_DOC, title="My deck"):
    deck["cards"].sort(key=lambda e: (e["card"], e["border"], e["mutation"]))
    deck["supports"].sort(key=lambda e: (e["color"], e["support"], e["tier"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(deck, indent=1, ensure_ascii=False) + "\n")
    Path(doc).write_text(render(deck, catalog, title))


def render(deck, catalog, title="My deck"):
    intro = ("These are the cards and supports you own. Searches use them as your collection. "
             "Edit with `python -m card_engine.deck`" if title == "My deck" else
             "A custom pool for suggestions tailored to one collection (`generate --pool custom`). "
             "Edit with `python -m card_engine.deck --custom`")
    lines = [f"# {title}", "",
             intro + " (see `python -m card_engine.deck --help`); this page is regenerated after every change.", "",
             f"## Cards ({sum(e['count'] for e in deck['cards'])})", "",
             "| Pack | Card | ID | Border | Mutation | Count |", "|---|---|---|---|---|---|"]
    pack_order = {name: i for i, name in enumerate(
        ("Era 1", "Egypt", "Anime", "Rising Sun", "Immortal", "Prehistoric", "Era 2", "Video Game", "Cryptid",
         "Bosses", "Limited", "Christmas", "Halloween", "Halloween 2025"))}
    for e in sorted(deck["cards"], key=lambda e: (pack_order.get(catalog.card(e["card"]).packs[0], 99),
                                                   catalog.card(e["card"]).rarity, e["card"], e["border"])):
        card = catalog.card(e["card"])
        lines.append(f"| {card.packs[0]} | {card.name} | {card.id} | {BORDERS[e['border'] - 1]} | {e['mutation']} | {e['count']} |")
    lines += ["", f"## Supports ({sum(e['count'] for e in deck['supports'])})", "",
              "| Colour | Support | ID | Tier | Count |", "|---|---|---|---|---|"]
    for e in deck["supports"]:
        lines.append(f"| {e['color']} | {e['name']} | {e['support']} | {TIERS[e['tier'] - 1]} | {e['count']} |")
    return "\n".join(lines) + "\n"


def _fold(text):
    """Case, punctuation and spacing ignored: "Sable, The Envious" reads as "sable the envious"."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _match(query, items, kind):
    """items: [(id, name)]. An ID, an exact name, or a unique substring; case and punctuation are ignored."""
    if query.isdigit():
        hits = [item for item in items if item[0] == int(query)]
    else:
        q = _fold(query)
        hits = [item for item in items if _fold(item[1]) == q] or [item for item in items if q in _fold(item[1])]
    if len(hits) != 1:
        options = ", ".join(f"{name} ({i})" for i, name in hits[:12]) or "none"
        raise SystemExit(f"{kind} '{query}' matches {len(hits)}: {options}")
    return hits[0]


def _border(text):
    lookup = {name.casefold(): i + 1 for i, name in enumerate(BORDERS)} | {"base": 1, "no": 1, "": 1}
    if text.isdigit() and 1 <= int(text) <= 16:
        return int(text)
    if text.casefold() not in lookup:
        raise SystemExit(f"Unknown border '{text}'; use one of {', '.join(BORDERS)}")
    return lookup[text.casefold()]


def _tier(text):
    if text.isdigit() and 1 <= int(text) <= 5:
        return int(text)
    lookup = {name.casefold(): i + 1 for i, name in enumerate(TIERS)} | {"pl": 2, "cr": 3, "ru": 4, "ga": 5, "none": 1}
    if text.casefold() not in lookup:
        raise SystemExit(f"Unknown tier '{text}'; use one of {', '.join(TIERS)} or 1-5")
    return lookup[text.casefold()]


def _mutation(text):
    hits = [m for m in MUTATION_NAMES if m.casefold() == text.casefold()]
    if not hits:
        raise SystemExit(f"Unknown mutation '{text}'; use one of {', '.join(MUTATION_NAMES)}")
    return hits[0]


def add_card(deck, catalog, query, border, mutation="None", count=1):
    card_id, name = _match(query, [(c.id, c.name) for c in catalog.cards], "Card")
    if mutation != "None" and catalog.card(card_id).weather_id != 1:
        raise SystemExit(f"{name} is a weather card; weather cards cannot be mutated")
    for e in deck["cards"]:
        if (e["card"], e["border"], e["mutation"]) == (card_id, border, mutation):
            e["count"] += count
            return e
    entry = {"card": card_id, "name": name, "border": border, "mutation": mutation, "count": count}
    deck["cards"].append(entry)
    return entry


def remove_card(deck, catalog, query, border=None, count=None, mutation=None):
    card_id, name = _match(query, [(c.id, c.name) for c in catalog.cards], "Card")
    hits = [e for e in deck["cards"] if e["card"] == card_id and (border is None or e["border"] == border)
            and (mutation is None or e["mutation"] == mutation)]
    if not hits:
        raise SystemExit(f"{name} ({'any border' if border is None else BORDERS[border - 1]}, "
                         f"{'any mutation' if mutation is None else mutation}) is not in the deck")
    if len(hits) > 1:
        options = ", ".join(f"--border {BORDERS[e['border'] - 1]} --mutation {e['mutation']!r}" for e in hits)
        raise SystemExit(f"{name} is in the deck {len(hits)} ways; give one: {options}")
    entry = hits[0]
    entry["count"] -= entry["count"] if count is None else count
    if entry["count"] <= 0:
        deck["cards"].remove(entry)
    return name


def _supports(catalog):
    data = json.loads((ROOT / "data" / "clean" / "dataset.json").read_text())
    return [(s["color"], s["support_id"], s["name"]) for s in data["supports"]]


def add_support(deck, catalog, query, tier, count=1):
    items = _supports(catalog)
    if query.isdigit():
        raise SystemExit("Red and blue support IDs overlap; give the support's name instead")
    index, _ = _match(query, [(i, n) for i, (_, _, n) in enumerate(items)], "Support")
    color, support_id, name = items[index]
    for e in deck["supports"]:
        if (e["color"], e["support"], e["tier"]) == (color, support_id, tier):
            e["count"] += count
            return e
    entry = {"color": color, "support": support_id, "name": name, "tier": tier, "count": count}
    deck["supports"].append(entry)
    return entry


def remove_support(deck, catalog, query, tier=None, count=None):
    items = _supports(catalog)
    index, _ = _match(query, [(i, n) for i, (_, _, n) in enumerate(items)], "Support")
    color, support_id, name = items[index]
    hits = [e for e in deck["supports"] if (e["color"], e["support"]) == (color, support_id) and (tier is None or e["tier"] == tier)]
    if len(hits) != 1:
        raise SystemExit(f"{name}: {len(hits)} matching entries; give --tier" if hits else f"{name} is not in the deck")
    entry = hits[0]
    entry["count"] -= entry["count"] if count is None else count
    if entry["count"] <= 0:
        deck["supports"].remove(entry)
    return name


def owned_entries(deck):
    """(card, border, mutation) the player owns, for searches."""
    return [(e["card"], e["border"], e["mutation"]) for e in deck["cards"]]


def owned_supports(deck):
    """([(red id, tier)], [(blue id, tier)]), for searches."""
    red = sorted({(e["support"], e["tier"]) for e in deck["supports"] if e["color"] == "red"})
    blue = sorted({(e["support"], e["tier"]) for e in deck["supports"] if e["color"] == "blue"})
    return red, blue


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m card_engine.deck", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--custom", action="store_true", help="edit the custom pool instead of your deck")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("add", help="add a card")
    p.add_argument("card")
    p.add_argument("--border", default="none")
    p.add_argument("--mutation", default="None")
    p.add_argument("--count", type=int, default=1)
    p = commands.add_parser("remove", help="remove a card (all copies unless --count)")
    p.add_argument("card")
    p.add_argument("--border")
    p.add_argument("--mutation")
    p.add_argument("--count", type=int)
    p = commands.add_parser("add-support", help="add a support card")
    p.add_argument("support")
    p.add_argument("--tier", default="base")
    p.add_argument("--count", type=int, default=1)
    p = commands.add_parser("remove-support", help="remove a support card")
    p.add_argument("support")
    p.add_argument("--tier")
    p.add_argument("--count", type=int)
    commands.add_parser("list", help="print the deck")
    commands.add_parser("reset", help="empty the custom pool (--custom only)")
    commands.add_parser("copy-deck", help="replace the custom pool with your deck (--custom only)")
    args = parser.parse_args(argv)
    path, doc, title = (CUSTOM_FILE, CUSTOM_DOC, "Custom pool") if args.custom else (DECK_FILE, DECK_DOC, "My deck")
    if args.command in ("reset", "copy-deck") and not args.custom:
        parser.error(f"{args.command} only applies to the custom pool: add --custom")
    catalog = load_catalog()
    deck = load(path)
    if args.command == "add":
        e = add_card(deck, catalog, args.card, _border(args.border), _mutation(args.mutation), args.count)
        message = f"Added {args.count} x {e['name']} ({BORDERS[e['border'] - 1]}, {e['mutation']}); now {e['count']}"
    elif args.command == "remove":
        name = remove_card(deck, catalog, args.card, _border(args.border) if args.border else None, args.count,
                           _mutation(args.mutation) if args.mutation else None)
        message = f"Removed {name}"
    elif args.command == "add-support":
        e = add_support(deck, catalog, args.support, _tier(args.tier), args.count)
        message = f"Added {e['name']} ({TIERS[e['tier'] - 1]}); now {e['count']}"
    elif args.command == "remove-support":
        name = remove_support(deck, catalog, args.support, _tier(args.tier) if args.tier else None, args.count)
        message = f"Removed {name}"
    elif args.command == "reset":
        deck = {"cards": [], "supports": []}
        message = "Custom pool emptied"
    elif args.command == "copy-deck":
        deck = load(DECK_FILE)
        message = f"Custom pool set to your deck ({sum(e['count'] for e in deck['cards'])} cards)"
    else:
        sys.stdout.write(render(deck, catalog, title))
        return
    save(deck, catalog, path, doc, title)
    print(message)


if __name__ == "__main__":
    main()
