"""Immutable metadata with explicit one-based IDs and no inferred combat rules."""

from dataclasses import dataclass
import json
from pathlib import Path

from .schema import validate_dataset


@dataclass(frozen=True, slots=True)
class CardDefinition:
    id: int
    name: str
    rarity: int
    weather_id: int
    card_modifier: float
    description: str
    original_description: str
    description_tokens: tuple[int, ...]
    packs: tuple[str, ...] | None
    classes: tuple[str, ...] | None
    class_assignment_status: str
    screenshot_source: str
    player_availability: str


@dataclass(frozen=True, slots=True)
class SupportDefinition:
    id: int
    color: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class BorderDefinition:
    id: int
    code: str
    rarity: int


@dataclass(frozen=True, slots=True)
class WeatherDefinition:
    id: int
    name: str
    multiplier: float


def _lookup(rows, identifier):
    # Reject bools, floats, zero, and negative Python indices explicitly.
    if type(identifier) is not int or not 1 <= identifier <= len(rows):
        raise ValueError(f"Expected a one-based integer ID in 1..{len(rows)}")
    return rows[identifier - 1]


@dataclass(frozen=True, slots=True)
class Catalog:
    cards: tuple[CardDefinition, ...]
    red_supports: tuple[SupportDefinition, ...]
    blue_supports: tuple[SupportDefinition, ...]
    borders: tuple[BorderDefinition, ...]
    weathers: tuple[WeatherDefinition, ...]
    pad_id: int
    bos_id: int
    card_token_id: int
    vocabulary: tuple[str, ...]

    def card(self, identifier: int) -> CardDefinition:
        return _lookup(self.cards, identifier)

    def border(self, identifier: int) -> BorderDefinition:
        return _lookup(self.borders, identifier)

    def weather(self, identifier: int) -> WeatherDefinition:
        return _lookup(self.weathers, identifier)

    def support(self, color: str, identifier: int) -> SupportDefinition:
        if color not in ("red", "blue"):
            raise ValueError("Support color must be 'red' or 'blue'")
        return _lookup(self.red_supports if color == "red" else self.blue_supports, identifier)


def load_catalog(path: str | Path | None = None) -> Catalog:
    """Load validated metadata; never interpret registry rows as executable abilities."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data/clean/dataset.json"
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(dataset)
    descriptions = dataset["descriptions"]
    tokens = dataset["tokens"]
    vocabulary_rows = sorted(tokens["vocabulary"], key=lambda row: row["token_id"])
    if [r["token_id"] for r in vocabulary_rows] != list(range(len(vocabulary_rows))):
        raise ValueError("Vocabulary IDs must be contiguous from PAD=0")
    cards = tuple(
        CardDefinition(
            id=row["card_id"], name=row["name"], rarity=row["rarity"],
            weather_id=row["weather_id"], card_modifier=row["card_modifier"],
            description=descriptions[index]["refined"],
            original_description=descriptions[index]["original"],
            description_tokens=tuple(tokens["rows"][index]["token_ids"]),
            packs=None if row["packs"] is None else tuple(row["packs"]),
            classes=None if row["classes"] is None else tuple(row["classes"]),
            class_assignment_status=row["class_assignment_status"],
            screenshot_source=row["metadata"]["screenshot"]["file"],
            player_availability=row["metadata"]["availability"]["player_use"],
        )
        for index, row in enumerate(dataset["cards"])
    )
    supports = tuple(SupportDefinition(r["support_id"], r["color"], r["name"], r["description"])
                     for r in dataset["supports"])
    return Catalog(
        cards=cards,
        red_supports=tuple(s for s in supports if s.color == "red"),
        blue_supports=tuple(s for s in supports if s.color == "blue"),
        borders=tuple(BorderDefinition(r["border_id"], r["name"], r["rarity"])
                      for r in dataset["borders"]),
        weathers=tuple(WeatherDefinition(r["weather_id"], r["name"], r["multiplier"])
                       for r in dataset["weathers"]),
        pad_id=tokens["special_ids"]["PAD"],
        bos_id=tokens["special_ids"]["BOS"],
        card_token_id=tokens["special_ids"]["CARD"],
        vocabulary=tuple(r["token"] for r in vocabulary_rows),
    )
