"""User-authorized metadata corrections, applied after source extraction.

Keep original files and per-field provenance in the import correction ledger.
These are identity-checked edits, not a shift of adjacent weather assignments.
"""

from .schema import require


WEATHER_CORRECTIONS = {36: ("Bad Boys", 9, 1), 37: ("Pandora", 6, 9)}
CONFIRMED_WEATHERS = {
    38: ("Heaven's Armor", 6),
    57: ("Surtr", 12), 58: ("Cronus", 8),
    258: ("Wandering Snowman", 3), 260: ("Santa Claus", 5),
    261: ("Toy Bear", 3), 265: ("Santa Claws", 3), 280: ("The Blood Countess", 11),
}
# Stats observed in user videos that differ from the formula regardless of border/support (user: apply them).
OBSERVED_MODIFIERS = {
    250: ("Time Lord Stryx", 1.5),       # IMG_0310: 15,360 ATK borderless (formula 10,240).
    139: ("Buddha", 1.02344),            # IMG_0303 335,361 and IMG_0311 5,241 ATK (formula 327,680 / 5,120).
    247: ("Legends", 1.2),               # IMG_0319: 6,144 ATK (formula 5,120).
    157: ("Tyrannosaurus Rex", 1.2),     # IMG_0314: 2,873 ATK (formula 2,394).
    159: ("Tyrannodon", 1.5),            # IMG_0314: 8,678 ATK before its fossil boost (formula 5,785).
}
HALLOWEEN_MODIFIERS = {284: 5, 289: 3}  # User: Hecate 5x, Glamour 3x; other Halloween 2025 cards 2x.
BOSS_NAMES = (
    "Chronus The Hoarder", "Malik The Sovereign", "Gideon The Insatiable",
    "Lilith the Enchantress", "Morpheus the Slumberer", "Raze The Destroyer",
    "Sable The Envious", "Mother of Beasts",
)


# User (2026-09-29): these art-inferred classes are wrong; the art/wiki sources are unreliable.
CLASS_REMOVALS = {152: ("Marrowclaw", "undead"),  # user's undead scan, 2026-10-02
                  65: ("Parallax", "demon"), 219: ("Sisyphus", "avian"), 246: ("Memories", "avian"),
                  280: ("The Blood Countess", "undead"), 285: ("Abomination", "undead"), 286: ("Sleep Paralysis", "undead")}
CLASS_ADDITIONS = {233: ("Anubis & Hades", "undead"),
                   242: ("War", "undead"), 270: ("Bloody Mary", "undead")}  # user's undead scan, 2026-10-02
# User (2026-09-30): the seven sins are every Bosses card except the last (Mother of Beasts).
SEVEN_SINS = {205: "Chronus The Hoarder", 206: "Malik The Sovereign", 207: "Gideon The Insatiable",
              208: "Lilith the Enchantress", 209: "Morpheus the Slumberer", 210: "Raze The Destroyer", 211: "Sable The Envious"}


# Classes the user scanned card by card in-game: every membership is confirmed (docs/classes.md).
USER_SCANNED_CLASSES = {"undead": "2026-10-02"}

# Developer's class lists (user, 2026-10-01): added to each class. Existing members stay until the
# user verifies them.
DEVELOPER_CLASSES = {
    "demon": {11: "Beelzebub", 16: "Tartarus", 120: "Demon Cultivator", 122: "Heavenly Demon", 60: "Fafnir", 46: "Vicious",
              110: "AK4-ON1", 109: "A0-ON1", 111: "Shuten-dōji", 33: "Hell's Army", 211: "Sable The Envious"},
    "swordsmen": {10: "Arthur", 55: "Siegfried", 18: "Arthur of Excalibur", 2: "Shining Armor", 14: "Knightmare",
                  113: "Sasaki Kojiro", 116: "Susanoo", 106: "Samurai", 20: "Conqueror", 23: "Ixion", 29: "Sciron",
                  87: "Artoria of Excalibur", 33: "Hell's Army"},
    "dragon": {177: "Juggernoid", 61: "Ragon", 60: "Fafnir", 17: "Greedy Belly", 210: "Raze The Destroyer", 112: "River Dragon",
               134: "Dōng Fāng Qīng Lóng", 238: "Ragon & Fafnir", 65: "Parallax", 234: "Longmu"},
    "avian": {15: "Arcane Avian", 31: "Phoenix", 32: "Hoarfrost Phoenix", 22: "Three-Legged Golden Crow",
              135: "Nán Fāng Zhū Què", 229: "Turkey"},
}


# User-verified RNG class (2026-09-29): mostly abilities that can fail by chance, plus the
# random-outcome cards the user listed (Pandora, Chaos, Gambler, Nuwa, Shay, Old Man Winter).
# Not RNG (user): The Sack, Ultimate Brawler, Loki, The Curse, Brachiosaurus, The Awakened One,
# Jersey Devil, Kraken, Marionette, Flying Dutchman, Glamour; Black Cat and Night Witch only
# target RNG cards; status inflicters (confusion, frostbite) are not RNG themselves.
CHANCE_BASED_RNG = frozenset((
    1, 15, 18, 21, 22, 27, 29, 31, 32, 37, 40, 50, 53, 64, 83, 93, 122, 130, 136, 148, 150, 152, 194, 201,
    216, 218, 220, 228, 232, 252, 260, 268, 270, 281))


def apply_user_corrections(cards, corrections):
    source = {"file": "../../card_engine/data_corrections.py",
              "evidence": "user stat/weather corrections and all-Bosses clarification, 2026-09-28"}

    def change(card, field, original, value, reason):
        require(card[field] == original, f"Card {card['card_id']} {field} source changed; review correction")
        key = "weather" if field == "weather_id" else field
        corrections.append({
            "kind": "user_authorized_stat_metadata_correction", "card_id": card["card_id"],
            "name": card["name"], "field": field, "original_value": original,
            "original_value_lexical": card[field + "_lexical"], "clean_value": value,
            "superseded_source": card["source"][key], "source": source, "reason": reason,
        })
        card[field] = value
        card[field + "_lexical"] = str(value)
        card["source"][key] = dict(source)

    additions = list(CLASS_ADDITIONS.items()) + [(card_id, (name, "sin")) for card_id, name in SEVEN_SINS.items()]
    for card_id, (name, missing) in additions:
        card = cards[card_id - 1]
        require(card["name"] == name and missing not in card["classes"], f"Card {card_id} class source changed; review class addition")
        card["classes"] = sorted(card["classes"] + [missing])
        card["metadata"]["gameplay_class_candidates"] = list(card["classes"])
        corrections.append({"kind": "user_authorized_class_addition", "card_id": card_id, "name": name, "added_class": missing,
                            "source": source, "reason": "User: class missing from the inferred data"})
    for card_id, (name, wrong) in CLASS_REMOVALS.items():
        card = cards[card_id - 1]
        require(card["name"] == name and wrong in card["classes"], f"Card {card_id} class source changed; review class removal")
        card["classes"] = [c for c in card["classes"] if c != wrong]
        card["metadata"]["gameplay_class_candidates"] = list(card["classes"])
        corrections.append({"kind": "user_authorized_class_removal", "card_id": card_id, "name": name, "removed_class": wrong,
                            "source": source, "reason": "User: inferred class is wrong; art/wiki class sources are unreliable"})
    for card in cards:
        was, now = "rng" in card["classes"], card["card_id"] in CHANCE_BASED_RNG
        if was != now:
            card["classes"] = sorted(set(card["classes"]) - {"rng"} | ({"rng"} if now else set()))
            card["metadata"]["gameplay_class_candidates"] = list(card["classes"])
            corrections.append({"kind": "user_rule_rng_class", "card_id": card["card_id"], "name": card["name"],
                                "rng": now, "source": source, "reason": "User: chance-based abilities qualify as RNG"})
    for class_name, members in DEVELOPER_CLASSES.items():
        for card in cards:
            if card["card_id"] in members:
                require(card["name"] == members[card["card_id"]], f"Card {card['card_id']} identity changed; review developer class")
            if card["card_id"] in members and class_name not in card["classes"]:
                card["classes"] = sorted(set(card["classes"]) | {class_name})
                card["metadata"]["gameplay_class_candidates"] = list(card["classes"])
                corrections.append({"kind": "developer_class_list", "card_id": card["card_id"], "name": card["name"],
                                    "class": class_name, "member": True, "source": source,
                                    "reason": "User: the developer's class list"})
    for card in cards:
        # User reviewed every class list (2026-09-30, "contradictions only"), so memberships are verified.
        card["metadata"]["gameplay_classes_verified"] = True
        card["class_assignment_status"] = "user_verified"
    corrections.append({"kind": "user_verified_class_memberships", "cards": len(cards), "source": source,
                        "reason": "User reviewed all class lists and supplied only contradictions"})
    for card_id, (name, original, weather_id) in WEATHER_CORRECTIONS.items():
        card = cards[card_id - 1]
        require(card["name"] == name, f"Card {card_id} identity changed; review weather correction")
        change(card, "weather_id", original, weather_id,
               "User confirms Base weather for Bad Boys and Eclipse for adjacent Pandora; no global index shift")
    for card_id, (name, weather_id) in CONFIRMED_WEATHERS.items():
        card = cards[card_id - 1]
        require((card["name"], card["weather_id"]) == (name, weather_id),
                f"Card {card_id} confirmed weather changed; review source alignment")
    for card_id, name, rarity in ((261, "Toy Bear", 400000000), (94, "Immortal Witch", 1500000)):
        require((cards[card_id - 1]["name"], cards[card_id - 1]["rarity"]) == (name, rarity),
                f"Card {card_id} user-confirmed rarity changed; review source")
    bosses = [card for card in cards if card["packs"] == ["Bosses"]]
    require(tuple(card["name"] for card in bosses) == BOSS_NAMES,
            "Bosses pack membership changed; review multiplier scope")
    for card in bosses:
        change(card, "card_modifier", 1, 2, "User confirms 2x stat multiplier for every card in Bosses pack")
    for card_id, (name, value) in OBSERVED_MODIFIERS.items():
        card = cards[card_id - 1]
        require(card["name"] == name, f"Card {card_id} identity changed; review observed modifier")
        change(card, "card_modifier", 1, value, f"Observed stats x{value} in user videos (see OBSERVED_MODIFIERS)")
    for card in cards:
        # User (2026-09-30): Halloween 2025 cards have 2x stats (Hecate 5x, Glamour 3x); its weather cards use
        # their weather multiplier instead, and the older Halloween pack has normal stats. Jason: own correction below.
        if (card["packs"][0] == "Halloween 2025" and card["card_modifier"] == 1
                and card["weather_id"] == 1 and card["card_id"] != 288):
            value = HALLOWEEN_MODIFIERS.get(card["card_id"], 2)
            change(card, "card_modifier", 1, value, f"User: Halloween 2025 card stats x{value} (IMG_0310/0325 Walking Dead 520, The Hanged Man 320)")
    change(cards[260], "card_modifier", 1, 2 / 3,
           "Inferred shared modifier with confirmed Snow weather (1.5x) fits user baseline 1943 ATK / 3886 HP; exact modifier not directly observed")
    jason = cards[287]
    require(jason["name"] == "Jason", "Card 288 identity changed; review stat correction")
    change(jason, "card_modifier", 1, 2,
           "Fitted 2x shared modifier before rounding matches user baseline 4788 ATK / 9575 HP; cause not specified")
