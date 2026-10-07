"""The C engine (sim_c) against DaddyDrago's TypeScript engine and search: identical answers, bit for bit."""

import random
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import drago, kernel
from card_engine.training import labels


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.pool = tuple(sorted(drago.supported(cls.catalog)[0]))

    def same(self, spec, seed, **tweaks):
        options = {"nodeBudget": 2000, "turns": True}  # the expected battle length too (depths speed)
        self.assertEqual(kernel.evaluate(self.catalog, spec, seed, **tweaks, **options),
                         drago.evaluate(self.catalog, spec, seed, **tweaks, **options))

    def test_random_battles(self):
        rng = random.Random(8)
        for seed in range(40):
            self.same(labels.random_spec(rng, self.catalog, cards=self.pool), seed + 1)

    def test_tweaks(self):
        rng = random.Random(9)
        for seed in range(6):
            spec = labels.random_spec(rng, self.catalog, cards=self.pool)
            self.same(spec, seed, fixed=(1, [(5000, 300)] * 4))
            self.same(spec, seed, scale=(0, 1.7))
            self.same(spec, seed, strip=(1, 2))

    def test_awakened_toys_and_draconian(self):
        rng = random.Random(10)
        cards, _, blue = drago.names(self.catalog)
        ids = {name: card for card, name in cards.items()}
        elf = next(s for s, name in blue.items() if name == "Magical Elf")
        toys = [ids[n] for n in ("Toy Bear", "Toy Car", "Toy Jack-in-the-Box", "Toy Nutcracker")]
        dragons = [ids[n] for n in ("Longmu", "Fafnir", "Ragon", "River Dragon")]  # Longmu (Draconian) leads
        for tier in range(1, 6):
            spec = labels.random_spec(rng, self.catalog, cards=self.pool)
            spec["cards"][0], spec["arts"][0], spec["mutations"][0] = toys, [0] * 4, [0] * 4
            spec["blue"][0], spec["blue_tier"][0] = elf, tier
            self.same(spec, tier)
            spec["cards"][0] = dragons
            self.same(spec, tier)

    def test_hades_copies_neither_paradox_nor_a_fallen_hades(self):
        # user, 2026-10-07: Hades cannot copy Parallax's Paradox, and a fallen Hades offers only The Underworld; his
        # engine copied both, so this deck beat floor 105 Impossible every time
        from card_engine import tower
        from card_engine.teams import parse_side, spec
        enemy = tower.fixed_team(self.catalog, 105)
        enemy.update(borders=[1] * 4, mutations=[0] * 4, red=0, red_tier=0, blue=0, blue_tier=0)
        per_card = tower.engine_stats(self.catalog, enemy["cards"], tower.stats(105, "Impossible"))
        ally = parse_side(self.catalog, ["Parallax", "Hades", "Hades", "Robin Hood"], None, "Storm Spirit")
        battle = spec(ally, enemy)
        self.same(battle, 1, fixed=(1, per_card))
        probs, _ = kernel.evaluate(self.catalog, battle, 1, fixed=(1, per_card))
        self.assertLess(probs[0], 0.01)

    def test_hades_copies_again_back_at_the_front(self):
        # user, 2026-10-07: a second Hades with Piccolo behind it copies Piccolo once Piccolo swaps in and dies; both
        # engines play these lines alike
        from card_engine.teams import parse_side, spec
        rng = random.Random(11)
        for seed, ally in enumerate((["Hades", "Hades", "Piccolo", "Robin Hood"], ["Robin Hood", "Hades", "Piccolo", "Hades"],
                                     ["Hades", "Piccolo", "Hades", "Piccolo"])):
            enemy = labels.random_spec(rng, self.catalog, cards=self.pool)
            battle = spec(parse_side(self.catalog, ally), parse_side(self.catalog, ["Archer", "Archer", "Archer", "Archer"]))
            battle["cards"][1] = enemy["cards"][1]
            self.same(battle, seed + 1)
            self.same({**battle, "cards": battle["cards"][::-1], "borders": battle["borders"][::-1],
                       "mutations": battle["mutations"][::-1], "arts": battle["arts"][::-1]}, seed + 7)

    def test_divine_mist_strips_borders_not_stats(self):
        # user, 2026-10-07: Achyls resets borders, not stats; tower enemies carry their difficulty's border
        from card_engine import tower
        from card_engine.teams import parse_side, spec
        ally = parse_side(self.catalog, ["Achyls", "Achyls", "Parallax", "Robin Hood"], "The One Ring", "Guardian Angel")
        for level in ("Impossible", "Hell"):
            enemy = tower.fixed_team(self.catalog, 105)
            enemy.update(borders=[tower.enemy_border(self.catalog, level)] * 4, mutations=[0] * 4, red=0, red_tier=0,
                         blue=0, blue_tier=0)
            per_card = tower.engine_stats(self.catalog, enemy["cards"], tower.stats(105, level))
            self.same(spec(ally, enemy), 3, fixed=(1, per_card))
            probs, _ = kernel.evaluate(self.catalog, spec(ally, enemy), 3, fixed=(1, per_card))
            self.assertLess(probs[0], 0.01)  # his engine wiped the floor's stats: 0.48 at Impossible
        rng = random.Random(12)
        for seed in range(4):  # against bordered player cards both engines strip the same multiplier
            battle = labels.random_spec(rng, self.catalog, cards=self.pool)
            battle["cards"][0][0] = ally["cards"][0]
            battle["borders"][1] = [9, 3, 2, 16]
            self.same(battle, seed + 20)

    def test_a_runaway_battle_with_hundreds_of_created_cards(self):
        rng = random.Random(22883)  # label shard 22883, row 1219: Pandora's rolls create 836 cards
        spec = [labels.random_spec(rng, self.catalog) for _ in range(1220)][-1]
        self.same(spec, 22883 * 1_000_003 + 1219)

    def test_unknown_options_are_refused(self):
        spec = labels.random_spec(random.Random(1), self.catalog, cards=self.pool)
        with self.assertRaises(TypeError):
            kernel.evaluate(self.catalog, spec, 1, nodebudget=10)


if __name__ == "__main__":
    unittest.main()
