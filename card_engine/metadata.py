"""Curated screenshot annotations. Inferences are never simulator predicates."""

import json
from pathlib import Path

from .schema import require

PACKS = (
    ("Era 1", 1, 65), ("Egypt", 66, 79), ("Anime", 80, 104),
    ("Rising Sun", 105, 119), ("Immortal", 120, 140),
    ("Prehistoric", 141, 159), ("Era 2", 160, 182),
    ("Video Game", 183, 194), ("Cryptid", 195, 204),
    ("Bosses", 205, 212), ("Limited", 213, 250),
    ("Christmas", 251, 265), ("Halloween", 266, 273),
    ("Halloween 2025", 274, 289),
)

# Screenshot order is the order supplied by the user. Repeated clipped tiles are
# omitted; a card is assigned its clearest occurrence.
SCREENSHOTS = (
    ("6.30.20", 1, 24), ("6.30.31", 25, 48), ("6.30.52", 49, 65),
    ("6.31.00", 66, 79), ("6.31.07", 80, 103), ("6.31.22", 104, 119),
    ("6.31.56", 120, 140), ("6.32.18", 141, 159), ("6.32.23", 160, 182),
    ("6.32.30", 183, 200), ("6.32.40", 201, 212),
    ("6.33.01", 213, 236), ("6.33.14", 237, 256),
    ("6.33.22", 257, 273), ("6.33.32", 274, 289),
)

# Each line corresponds to consecutive source IDs within the named pack. Tags
# describe visible art/name cues, not a claim about hidden game classifications.
VISUAL_TAGS = {
    "Era 1": """
humanlike
humanlike,armored
animal,canine
humanlike
undead,skeletal
humanlike,mage
animal,deer
humanlike,warrior
humanlike
humanlike,warrior
demon,insectoid
humanlike,divine
humanlike,demon
armored,humanoid
humanoid,avian
humanoid,demon,armored
dragon,reptile
humanlike,warrior
humanlike,spirit
humanlike,hooded
humanlike,ice
avian,fire
armored,humanoid
dragon,aquatic
humanlike,warrior
humanlike,hooded
humanoid,undead
humanoid,armored
construct,armored
spirit,hooded
avian,fire
avian,ice
humanlike,demon
humanlike,divine
equine,winged
spirit,group
humanlike
armored,divine
humanlike,armored
humanoid,fungal
humanlike,spirit
humanlike,warrior
humanlike,horned
humanlike,divine
abstract
spirit
dragon,armored
humanlike,demon
humanlike,divine
divine,winged
humanlike,divine
humanoid,plant
humanoid,antlered
humanlike,hooded
humanlike,warrior
armored,divine
humanoid,fire
humanlike,divine
humanlike,warrior
dragon
dragon
aquatic,reptile
animal,mammal,ice
spirit,humanoid
humanoid,demon
""",
    "Egypt": """
undead,hooded
humanlike
humanlike
animal,feline
humanlike,divine
humanlike
animal,feline,divine
avian,humanoid,divine
humanlike,divine
animal,humanoid,divine
canine,humanoid,divine
humanoid,insectoid
unknown
avian,divine
""",
    "Anime": """
humanlike
humanlike,demon
humanlike,warrior
humanlike
humanlike
humanlike,horned
humanlike
humanlike,warrior
humanlike
humanlike
humanlike
humanlike,undead
humanlike,ice
humanlike
humanlike
humanlike
humanlike
humanlike
humanlike,warrior
humanlike
humanlike
humanlike,feline
humanlike
humanlike
humanoid,alien
""",
    "Rising Sun": """
humanlike,warrior
humanlike,warrior
humanlike,mage
animal,canine,spirit
construct,armored
construct,armored
demon,humanoid
dragon,aquatic
humanlike,warrior
animal,canine,spirit
reptile,serpent
humanlike,divine
humanlike,divine
humanlike,divine
humanlike,spirit
""",
    "Immortal": """
humanlike,demon
humanlike,warrior
humanlike,demon
humanlike,undead
humanlike
humanlike
humanlike
animal,mythical
humanlike,divine
humanlike,divine
humanlike,divine
dragon
animal,feline,divine
animal,reptile,divine
dragon,divine
avian,divine,fire
humanlike,divine
humanlike,divine
animal,primate,divine
humanoid,divine
humanoid,divine
""",
    "Prehistoric": """
object,egg
dinosaur
dinosaur
dinosaur
reptile,winged
dinosaur
animal,mammal
dinosaur,shadow
animal,feline
animal,equine,mythical
dinosaur,fire
dinosaur,undead
dinosaur
dinosaur
dinosaur,construct
dinosaur
dinosaur
aquatic,shark
dinosaur
""",
    "Era 2": """
humanoid,armored
humanlike,warrior
humanoid,goblin
animal,feline
humanoid,hooded
divine,winged
humanlike
spirit,fire
humanlike,warrior
humanlike
humanlike,hooded
animal,reptile
humanoid,divine,winged
construct
humanlike,undead
humanlike,mage
undead,skeletal
animal,armored
dragon,undead
humanlike,warrior
humanlike
divine,winged
animal,feline
""",
    "Video Game": """
humanlike,construct
humanoid,insectoid
humanlike,armored
humanoid
humanlike,warrior
humanlike,warrior
humanlike,warrior
humanoid
humanlike,construct
humanoid,armored,ice
humanoid,divine
humanlike,demon
""",
    "Cryptid": """
cryptid,humanoid
cryptid,aquatic,reptile
cryptid,amphibian
cryptid,animal
cryptid,winged
cryptid,insectoid,winged
cryptid,primate
cryptid,antlered
cryptid,humanoid
cryptid,aquatic
""",
    "Bosses": """
humanlike
humanlike
humanoid,ogre
humanlike,demon
animal,spirit
dragon
humanlike
humanoid,antlered
""",
    "Limited": """
humanlike,group
humanoid,serpent
humanlike
humanlike
animal,primate
humanlike,spirit
avian,shadow
humanlike
humanoid,group
humanlike,divine
humanlike
humanlike
object
humanlike,divine
animal,feline
humanoid,armored
avian
construct,toy
animal,feline
humanoid,goblin
humanoid,group,divine
humanlike
aquatic
abstract
undead,hooded
dragon,group
humanoid,aquatic
humanlike
undead,hooded
humanoid,armored
humanoid,group,divine
humanlike,mage
undead,hooded
avian,group
humanoid,armored
avian
dragon,aquatic
humanlike,divine
""",
    "Christmas": """
construct,snowman
animal,mammal,ice
animal,deer
construct,toy
humanlike,divine
humanoid
demon,horned
construct,snowman
humanlike
humanlike
construct,toy,animal
construct,toy,object
construct,toy
construct,toy,humanoid
humanoid
""",
    "Halloween": """
undead,humanoid
spirit,undead
humanlike,mage
construct,plant
humanlike,spirit
undead,skeletal
humanlike,undead
construct,toy,humanoid
""",
    "Halloween 2025": """
unknown
humanlike,undead
humanlike,mage
humanoid,shadow
undead,humanoid
undead,plant
humanlike,undead
humanoid,undead
humanlike,undead
object,spirit
humanlike,mage
humanoid,undead
humanoid,undead
humanlike,masked
humanlike,masked
humanoid,demon
""",
}

UNLIT = {62, 64, 65, 156, 158, *range(183, 205), *range(212, 290)}
UNCLEAR_AVAILABILITY = {78}


def annotate(cards, descriptions, raw_dir: Path):
    """Apply versioned annotations while preserving card names, IDs and rarities."""
    packs, annotations = [], []
    source_classes_path = raw_dir.parents[0] / "annotations/sourced_classes.json"
    source_classes = json.loads(source_classes_path.read_text()) if source_classes_path.exists() else []
    if isinstance(source_classes, dict):
        source_classes = source_classes.get("memberships", source_classes.get("rows", []))
    for pack_id, (name, start, end) in enumerate(PACKS, 1):
        tags = [line.strip().split(",") for line in VISUAL_TAGS[name].strip().splitlines()]
        require(len(tags) == end-start+1, f"Visual annotation count for {name}")
        packs.append({"pack_id": pack_id, "name": name, "first_card_id": start, "last_card_id": end,
                      "basis": "User-specified pack order and screenshot section boundaries", "status": "user_screenshot_supported"})
        for card_id, visual_tags in zip(range(start, end+1), tags, strict=True):
            image_id, (time, first, last) = next((i, s) for i, s in enumerate(SCREENSHOTS, 1) if s[1] <= card_id <= s[2])
            image_path = f"user_2026-09-27/Screenshot 2026-09-26 at {time}\u202fPM.png"
            require((raw_dir / image_path).is_file(), f"Missing screenshot evidence {image_path}")
            candidates = sorted(set(visual_tags) & {"dragon", "avian", "demon", "undead", "toy"})
            if card_id in (109, 110):
                candidates.append("friendship")
            memberships = [row for row in source_classes if row.get("card_id") == card_id]
            candidates = sorted(set(candidates) | {row["class_name"] for row in memberships})
            available = "unavailable" if card_id in UNLIT else "unknown" if card_id in UNCLEAR_AVAILABILITY else "available"
            annotation = {
                "card_id": card_id, "name": cards[card_id-1]["name"], "pack_id": pack_id, "pack": name,
                "screenshot": {"file": image_path, "image_number": image_id,
                               "card_ordinal_excluding_clipped_repeats": card_id-first+1},
                "visual_tags": visual_tags,
                "visual_inference_status": "unresolved_blank_art" if visual_tags == ["unknown"] else "inferred_from_art_and_name",
                "gameplay_class_candidates": candidates,
                "sourced_class_memberships": memberships,
                "gameplay_classes_verified": False,
                "availability": {"player_use": available, "opponent_use": "training_dummy_available",
                                 "basis": "user_battle_observation" if card_id == 3 else "screenshot_brightness_inference",
                                 "confidence": "confirmed" if card_id == 3 else "low" if available == "unknown" else "medium",
                                 "as_of": "2026-09-26", "requires_confirmation_for_test_setup": card_id != 3},
            }
            cards[card_id-1].update(pack_id=pack_id, packs=[name], classes=candidates,
                                   class_assignment_status="inferred_not_verified", metadata=annotation)
            annotations.append(annotation)
    require([r["card_id"] for r in annotations] == list(range(1, 290)), "Metadata ID alignment")
    return packs, annotations
