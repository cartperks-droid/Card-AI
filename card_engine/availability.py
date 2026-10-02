"""How easy a card is to roll: its rarity, divided by the chance of rolling while its weather is active.

Weather availability (user, 2026-10-02): base (no weather) cards 0.9 (the usual rolling phase); a weather is
active about 0.1 of the time; Storm and Snow are the most common (1.0 of that), the other common weathers
(Shroud, Aurora, Rapture) assumed 0.5; rare weathers take their share from the weather-reroll table and last
~1.5 min instead of ~2 (x 0.75).
"""

DEFAULT_WEATHER = {
    "base": 0.9,
    "any_weather": 0.1,
    "common_rates": {"Storm": 1.0, "Snow": 1.0, "Shroud": 0.5, "Aurora": 0.5, "Rapture": 0.5},
    "rates": {"Meteor Shower": 0.1642, "Time Storm": 0.6629, "Eclipse": 0.0907, "Virus": 0.0374,
              "Blood Rain": 0.0263, "Armageddon": 0.0133, "Manga": 0.0067},
    "rare_duration_factor": 0.75,
}


def weather_chance(catalog, card, weather=DEFAULT_WEATHER):
    """Fraction of rolling time during which `card` can drop."""
    if weather is None:
        return 1.0
    name = next(w.name for w in catalog.weathers if w.id == catalog.card(card).weather_id)
    if name == "Base":
        return weather.get("base", 1.0)
    if name in weather["rates"]:
        return weather["any_weather"] * weather["rates"][name] * weather.get("rare_duration_factor", 1.0)
    return weather["any_weather"] * weather["common_rates"][name]


def roll_rarity(catalog, card, border=1, weather=DEFAULT_WEATHER):
    """Expected rolls to obtain `card` at `border`."""
    return catalog.card(card).rarity * catalog.border(border).rarity / weather_chance(catalog, card, weather)
