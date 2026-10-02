"""Mutation factors fitted to the user's rounded stat observations.

These factors are simple exact candidates within the observed rounding bounds.
They are not a proof of unique values. Weather cards retain their separate
weather multiplier and are ineligible for mutations.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from types import MappingProxyType

from .catalog import Catalog, load_catalog


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data/annotations/mutations.json"
DEFAULT_FIT = ROOT / "data/review/mutation_fit.csv"
MULTIPLIERS = MappingProxyType({
    "None": 1.0,
    "Storm": 1.1,
    "Snow": 1.2,
    "Aurora": 1.3,
    "Shroud": 1.5,
    "Meteor Shower": 1.8,
    "Time Storm": 2.0,
    "Eclipse": 2.5,
    "Virus": 3.0,
    "Blood Rain": 3.5,
    "Armageddon": 4.0,
    "Manga": 4.5,
})

MUTATION_NAMES = tuple(MULTIPLIERS)


def effective_mutation(weather_name: str, selected: str = "None") -> str:
    """Weather scaling is separate; weather cards are ineligible for mutations."""
    if not isinstance(weather_name, str) or weather_name not in (*MUTATION_NAMES, "Rapture"):
        raise ValueError(f"Unknown weather {weather_name!r}")
    mutation_multiplier(selected)
    if weather_name != "None" and selected != "None":
        raise ValueError("Weather cards are ineligible for mutations; stacking is not allowed")
    return selected


def mutation_multiplier(name: str) -> float:
    """Return a fitted factor for an exact display name, including ``'None'``.

    Unknown names, non-string values, and unobserved Rapture mutations fail
    explicitly instead of borrowing an intrinsic-weather multiplier.
    """
    if not isinstance(name, str) or name not in MULTIPLIERS:
        raise ValueError(f"Unknown mutation {name!r}; expected one of {tuple(MULTIPLIERS)}")
    return MULTIPLIERS[name]


def generate_fit(catalog: Catalog, annotations_path=ANNOTATIONS) -> list[dict]:
    """Verify all ordered observations and return their individual fit bounds.

    The reported cards have baseline weather and Card Modifier 1. This function
    rejects other baselines rather than assuming unconfirmed stacking behavior.
    """
    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    annotated_factors = {row["mutation"]: row["multiplier"]
                         for row in annotations["candidate_multipliers"]}
    if annotated_factors != dict(MULTIPLIERS):
        raise ValueError("Mutation annotations and fitted-factor API disagree")
    observations = annotations["observations"]
    if len(observations) != 28:
        raise ValueError("Mutation evidence must retain all 28 ordered observations")
    rows = []
    for index, observation in enumerate(observations, 1):
        if observation["observation_id"] != index:
            raise ValueError("Mutation observation order/IDs changed")
        card = catalog.card(observation["card_id"])
        if card.name != observation["card_name"]:
            raise ValueError(f"Mutation observation {index} card name/ID mismatch")
        weather = catalog.weather(card.weather_id)
        if weather.multiplier != 1 or card.card_modifier != 1:
            raise ValueError(f"{card.name}: unverified mutation/intrinsic multiplier stacking")
        for stat in ("hp", "attack"):
            if type(observation[stat]) is not int or observation[stat] <= 0:
                raise ValueError(f"Mutation observation {index}: {stat} must be a positive integer")
        mutation = observation["mutation"]
        multiplier = mutation_multiplier(mutation)
        factor = 2 ** math.log10(card.rarity)
        hp_base, attack_base = 10 * factor, 5 * factor
        predicted_hp, predicted_attack = math.ceil(hp_base * multiplier), math.ceil(attack_base * multiplier)
        if (predicted_hp, predicted_attack) != (observation["hp"], observation["attack"]):
            raise ValueError(f"Mutation observation {index} does not match: "
                             f"predicted HP/ATK {predicted_hp}/{predicted_attack}, "
                             f"reported {observation['hp']}/{observation['attack']}")
        # ceil(x*m)=y implies (y-1)/x < m <= y/x. Intersect the two
        # displayed-stat intervals without mistaking them for exact factors.
        lower = max((observation["hp"] - 1) / hp_base,
                    (observation["attack"] - 1) / attack_base)
        upper = min(observation["hp"] / hp_base,
                    observation["attack"] / attack_base)
        rows.append({
            "observation_id": index,
            "card_id": card.id,
            "card_name": card.name,
            "mutation": mutation,
            "reported_attack": observation["attack"],
            "reported_hp": observation["hp"],
            "pre_round_base_attack": attack_base,
            "pre_round_base_hp": hp_base,
            "candidate_multiplier": multiplier,
            "predicted_attack": predicted_attack,
            "predicted_hp": predicted_hp,
            "lower_exclusive": lower,
            "upper_inclusive": upper,
            "matches": True,
            "interpretation": "simple_exact_candidate_consistent_with_rounded_observations",
        })
    expected_counts = {69: 8, 67: 9, 10: 11}
    actual_counts = {card_id: sum(r["card_id"] == card_id for r in rows)
                     for card_id in {r["card_id"] for r in rows}}
    if actual_counts != expected_counts:
        raise ValueError("Mutation observation card counts changed")
    return rows


def fit_intervals(rows: list[dict]) -> list[dict]:
    """Intersect bounds across all observations sharing a mutation name."""
    intervals = []
    for name, multiplier in MULTIPLIERS.items():
        observations = [row for row in rows if row["mutation"] == name]
        if not observations:
            continue
        lower = max(row["lower_exclusive"] for row in observations)
        upper = min(row["upper_inclusive"] for row in observations)
        if not lower < multiplier <= upper:
            raise ValueError(f"{name}: candidate is outside the joint rounding interval")
        intervals.append({
            "mutation": name,
            "candidate_multiplier": multiplier,
            "lower_exclusive": lower,
            "upper_inclusive": upper,
            "observation_count": len(observations),
            "unique_factor_proven": False,
        })
    return intervals


def export_fit(catalog: Catalog, output_path=DEFAULT_FIT, annotations_path=ANNOTATIONS):
    rows = generate_fit(catalog, annotations_path)
    intervals = {row["mutation"]: row for row in fit_intervals(rows)}
    for row in rows:
        interval = intervals[row["mutation"]]
        row["joint_lower_exclusive"] = interval["lower_exclusive"]
        row["joint_upper_inclusive"] = interval["upper_inclusive"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: "true" if value is True else value for key, value in row.items()}
                         for row in rows)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIT)
    args = parser.parse_args()
    print(export_fit(load_catalog(), args.output))


if __name__ == "__main__":
    main()
