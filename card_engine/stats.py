"""Dataset stat formula, before support and ability effects.

Weather is intrinsic. Card Modifier scales both HP and ATK. A border changes
effective rarity. This does not assert that every card lacks special stat rules.
"""

from dataclasses import dataclass
import math

from .catalog import Catalog


# Separate HP/ATK ratios are not replacements for the shared Card Modifier.
# Borderless Witch ATK matches IMG_0289.MOV; user confirmed 868 HP outside battle.
# Shu: user-read 1230 HP / 362 ATK, "an abnormality for its weather multiplier".
# Fitted as HP weather 1.7 instead of 1.75 and no weather on ATK; any HP ratio
# rounding to 1230 fits equally, so the exact factor is unverified.
STAT_RATIO_PROFILES = {
    # DaddyDrago's engine data, 2026-10-03 (user: his stats are more accurate): HP multiplier, ATK unchanged.
    29: ("Sciron", 1.1, 1, "DaddyDrago engine data", "HP x1.1 per his card data"),
    63: ("Yeti", 1.5, 1, "DaddyDrago engine data", "HP x1.5 per his card data"),
    91: ("Vampire Lord", 1.5, 1, "DaddyDrago engine data", "HP x1.5 per his card data; ATK matches user videos IMG_0336/0338"),
    217: ("Gorilla", 1.1, 1, "DaddyDrago engine data", "HP x1.1 per his card data"),
    94: ("Immortal Witch", 1.2, .8, "https://card-rng.fandom.com/wiki/Immortal_Witch",
         "experimental stat ratio: borderless 868 HP user-confirmed; 290 ATK matches user video; ratios sourced from wiki"),
    74: ("Shu", 1.7 / 1.75, 1 / 1.75, "user report 2026-09-28",
         "experimental stat ratio fitted to user-read borderless 1230 HP / 362 ATK"),
    # User: Piccolo's card shows 3,072 ATK out of battle (formula: 2,560), unmutated.
    # HP is not displayed; the same 1.2 ratio is assumed (6,144 HP).
    104: ("Piccolo", 1.2, 1.2, "user report 2026-09-29",
          "experimental stat ratio: user-read 3,072 ATK out of battle; HP ratio assumed equal"),
    # User: 100 Men's 100x HP is part of its baseline before battle, not an entry boost.
    213: ("100 Men", 100, 1, "user report 2026-09-29", "baseline HP boosted 100x before battle (user-confirmed)"),
    # User: Yamato 446 ATK / 4,455 HP and Pangu 5,270 ATK / 21,077 HP, both borderless. ATK fits
    # the formula; HP is x5 and x2 of the formula value before rounding (ceil(5 x 890.9), ceil(2 x 10538.x)).
    # User (2026-09-30): borderless stats that differ from the formula.
    70: ("Sekhmet", 1.7, 1, "user report 2026-09-30", "user-read borderless 4,039 HP / 1,188 ATK"),
    116: ("Susanoo", 1.2, 0.6, "user report 2026-09-30", "user-read borderless 3,331 HP / 833 ATK"),
    283: ("Flying Dutchman", 2, 1, "user report 2026-09-30", "user-read borderless 7,127 HP / 1,782 ATK (Halloween 2025 x2 included)"),
    # User: Priest has 30% more HP than the formula.
    89: ("Priest", 1.3, 1, "user report 2026-09-30", "user-reported +30% HP"),
    115: ("Yamato no Orochi", 5, 1, "user report 2026-09-29", "user-read borderless 4,455 HP / 446 ATK"),
    140: ("Pangu", 2, 1, "user report 2026-09-29", "user-read borderless 21,077 HP / 5,270 ATK"),
}


@dataclass(frozen=True, slots=True)
class BaseStats:
    hp: int
    attack: int
    effective_rarity: int
    intrinsic_weather_multiplier: float
    card_modifier: float
    mutation: str = "None"
    mutation_multiplier: float = 1.0
    mutation_eligible: bool = True
    hp_ratio_multiplier: float = 1.0
    attack_ratio_multiplier: float = 1.0
    stat_ratio_source: str | None = None
    rule: str = "ceil(10 or 5 × 2^log10(card rarity × border rarity) × weather × Card Modifier × mutation × HP/ATK ratio); mutations require Base weather"
    verification: str = "dataset_formula; confirmed 2560/1280 for borderless cards 33 and 38"


# User (2026-10-01): Santa Claus has a 2x stat multiplier instead of Aurora's 1.75x.
CARD_WEATHER_MULTIPLIERS = {260: ("Santa Claus", 2.0)}


def base_stats(catalog: Catalog, card_id: int, border_id: int = 1, *, mutation: str = "None") -> BaseStats:
    """Round once, after all supported starting-stat factors.

    Weather cards keep their separate weather multiplier and are ineligible
    for mutations. Ordinary cards use the fitted selected mutation factor.
    """
    from .mutations import effective_mutation, mutation_multiplier
    card, border = catalog.card(card_id), catalog.border(border_id)
    weather = catalog.weather(card.weather_id)
    effective_name = effective_mutation(weather.name if weather.id != 1 else "None", mutation)
    mutation_factor = mutation_multiplier(effective_name)
    effective_rarity = card.rarity * border.rarity
    weather_multiplier = weather.multiplier
    if card_id in CARD_WEATHER_MULTIPLIERS:
        name, weather_multiplier = CARD_WEATHER_MULTIPLIERS[card_id]
        if card.name != name:
            raise ValueError(f"Card {card_id} identity changed; review weather multiplier")
    factor = 2 ** math.log10(effective_rarity) * weather_multiplier * card.card_modifier * mutation_factor
    hp_ratio, attack_ratio, source = 1.0, 1.0, None
    verification = ("dataset_formula; mutation scaling fits all 28 supplied base-card snapshots"
                    if weather.id == 1 and mutation != "None" else
                    "dataset_formula; confirmed 2560/1280 for borderless cards 33 and 38")
    if card_id in STAT_RATIO_PROFILES:
        name, hp_ratio, attack_ratio, source, verification = STAT_RATIO_PROFILES[card_id]
        if card.name != name:
            raise ValueError(f"Card {card_id} identity changed; review sourced stat ratios")
        if border_id != 1 or mutation != "None":
            verification += "; border/mutation scaling extrapolated, not independently verified"
    elif card_id in (261, 288):
        verification = "user-confirmed baseline; shared Card Modifier fitted to stats, underlying cause unverified"
        if border_id != 1 or mutation != "None":
            verification += "; border/mutation scaling extrapolated, not independently verified"
    return BaseStats(math.ceil(10 * factor * hp_ratio), math.ceil(5 * factor * attack_ratio), effective_rarity,
                     weather_multiplier, card.card_modifier, mutation=effective_name, mutation_multiplier=mutation_factor,
                     mutation_eligible=weather.id == 1,
                     hp_ratio_multiplier=hp_ratio, attack_ratio_multiplier=attack_ratio,
                     stat_ratio_source=source, verification=verification)
