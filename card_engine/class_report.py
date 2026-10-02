"""Write docs/classes.md: every class membership and where it came from, in in-game index order (pack, rarity)."""

from pathlib import Path

from .catalog import load_catalog
from .data_corrections import CHANCE_BASED_RNG, CLASS_ADDITIONS, DEVELOPER_CLASSES, SEVEN_SINS, USER_SCANNED_CLASSES
from .metadata import PACKS

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "classes.md"
CLASSES = ("demon", "dragon", "avian", "swordsmen", "sin", "rng", "undead", "toy", "friendship")


def _rarity(value):
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= scale:
            whole = int(value / scale * 100) / 100  # truncated to two decimals
            return f"1 in {whole:g}{suffix}"
    return f"1 in {value:g}"


def _source(card, name):
    if name in USER_SCANNED_CLASSES:
        return f"Confirmed (user scan {USER_SCANNED_CLASSES[name]})"
    if card.id in DEVELOPER_CLASSES.get(name, {}):
        return "Confirmed (developer list)"
    if name == "sin" and card.id in SEVEN_SINS:
        return "Confirmed (user: the seven sins)"
    if name == "rng" and card.id in CHANCE_BASED_RNG:
        return "Confirmed (user's RNG rule and list)"
    if name == "friendship":
        return "Confirmed (user: A0-ON1 / AK4-ON1)"
    if CLASS_ADDITIONS.get(card.id, (None, None))[1] == name:
        return "Confirmed (user addition)"
    return "Reviewed (art-inferred)"


def write(path=OUTPUT):
    catalog = load_catalog()
    pack_order = {name: index for index, (name, _, _) in enumerate(PACKS)}
    lines = ["# Card classes", "",
             "This is every class membership the simulator and model use, with where each one came from. "
             "**Confirmed** = from the developer's list, a list you gave explicitly, or your in-game scan of the whole class. "
             "**Reviewed** = inferred from the card art, then accepted when you said on 2026-10-02 that the classes are verified. "
             "If you're scanning cards in-game, only the *Reviewed* entries still need checking.", "",
             "Rows follow the in-game index: by card pack, then rarity. Regenerate with "
             "`python -m card_engine.class_report` after a class change.", ""]
    for name in CLASSES:
        members = sorted((c for c in catalog.cards if name in c.classes),
                         key=lambda c: (pack_order[c.packs[0]], c.rarity, c.id))
        sources = [_source(c, name) for c in members]
        confirmed = sum(s.startswith("Confirmed") for s in sources)
        lines += [f"## {name.capitalize()}: {len(members)} cards ({confirmed} confirmed, {len(members) - confirmed} reviewed)", "",
                  "| Pack | Rarity | ID | Card | Source |", "|---|---|---|---|---|"]
        lines += [f"| {c.packs[0]} | {_rarity(c.rarity)} | {c.id} | {c.name} | {s} |" for c, s in zip(members, sources)]
        lines.append("")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    write()
