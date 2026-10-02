"""Team ownership likelihood: copies -> per-card ownership, and the 2-D independence parameter (user, 2026-10-02)."""

import math
import random
import unittest

from card_engine import ownership


class OwnershipTests(unittest.TestCase):
    def test_reproduces_the_64_ball_table(self):
        p = [50 / 64] * 3  # A, B, C each in 50 of 64 balls: >= 22 balls hold all three, at most 50
        self.assertAlmostEqual(ownership.bounds(p)[0] * 64, 22)
        expected = {0.0: (34.38, 78.13), 0.25: (37.70, 70.52), 0.5: (41.03, 62.90), 0.75: (44.36, 55.29),
                    0.9: (46.36, 50.73), 0.95: (47.02, 49.21), 1.0: (47.68, 47.68)}
        for independence, (low, high) in expected.items():
            a, b = ownership.team_range(p, independence)
            self.assertAlmostEqual(100 * a, low, delta=0.01)  # the table was computed from rounded percentages
            self.assertAlmostEqual(100 * b, high, delta=0.01)

    def test_copies_to_ownership(self):
        self.assertAlmostEqual(ownership.owned_probability(300, 300, 1e9), 1 - math.exp(-1), places=6)  # even spread
        self.assertAlmostEqual(ownership.owned_probability(300, 300, 1.0), 0.5)  # duplicates concentrated (d = 1)
        self.assertGreater(ownership.owned_probability(3000, 300, 1.0), 0.9)

    def test_fit_recovers_coupling(self):
        rng = random.Random(3)
        rates = {1: 0.5, 2: 0.4, 3: 0.3}
        data = {"total_players": 1000, "effective_players": None,
                "copies": {str(c): -1000 * math.log(1 - r) for c, r in rates.items()},
                "dispersion": 1e9, "independence": None, "coupling": None, "samples": {}}
        for player in range(600):  # progression luck: lucky players own everything (nested ownership)
            luck = rng.random()
            data["samples"][str(player)] = [c for c, r in rates.items() if luck < r]
        independence, coupling, _ = ownership.fit_independence(data)
        self.assertLessEqual(independence, 0.3)
        self.assertGreaterEqual(coupling, 0.8)
        for player in range(600):  # independent ownership
            data["samples"][str(player)] = [c for c, r in rates.items() if rng.random() < r]
        independence, _, _ = ownership.fit_independence(data)
        self.assertGreaterEqual(independence, 0.7)


    def test_fit_population_recovers_the_effective_player_count(self):
        # Copies held by 5000 all-time players, but the fit starts from 350 (the active count).
        rng = random.Random(4)
        true_n, d = 5000, 1.0
        copies = {c: n for c, n in zip(range(1, 9), (200, 600, 1500, 3000, 6000, 12000, 25000, 60000))}
        rates = {c: ownership.owned_probability(n, true_n, d) for c, n in copies.items()}
        samples = {str(p): [c for c, r in rates.items() if rng.random() < r] for p in range(500)}
        data = {"total_players": 350, "effective_players": None, "copies": {str(c): n for c, n in copies.items()},
                "dispersion": 1.0, "independence": None, "coupling": None, "samples": samples}
        n, fitted_d = ownership.fit_population(data)
        self.assertTrue(true_n / 2 <= n <= true_n * 2, n)



class ProgressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from card_engine.catalog import load_catalog
        cls.catalog = load_catalog()

    def test_parse_rarity(self):
        self.assertEqual(ownership.parse_rarity("30T"), 30e12)
        self.assertEqual(ownership.parse_rarity("50qd"), 50e15)
        self.assertEqual(ownership.parse_rarity("1/7,500,000"), 7.5e6)
        self.assertAlmostEqual(ownership.parse_rarity("1.5qn"), 1.5e18)

    def test_mixture_recovers_two_groups(self):
        rng = random.Random(6)
        values = [rng.gauss(13.5, 0.4) for _ in range(120)] + [rng.gauss(16.7, 0.5) for _ in range(60)]
        mixture = sorted(ownership.fit_mixture(values), key=lambda c: c[1])
        self.assertEqual(len(mixture), 2)
        self.assertAlmostEqual(mixture[0][1], 13.5, delta=0.3)
        self.assertAlmostEqual(mixture[1][1], 16.7, delta=0.3)
        self.assertAlmostEqual(mixture[1][0], 1 / 3, delta=0.1)

    def test_team_lands_inside_the_bounds_with_positive_coupling(self):
        data = {"progression": [(0.7, 13.5, 0.5), (0.3, 16.7, 0.6)], "kappa": 1.0, "copies": {}}
        team = [206, 112, 171, 70]  # Malik, River Dragon, Infected Maw, Sekhmet
        p = [ownership.progression_team(data, self.catalog, [c]) for c in team]
        low, independent, high = ownership.bounds(p)
        joint = ownership.progression_team(data, self.catalog, team)
        self.assertTrue(low - 1e-12 <= joint <= high + 1e-12)
        self.assertGreater(joint, independent)  # shared progression couples ownership

    def test_kappa_reproduces_a_known_owner_count(self):
        data = {"progression": [(0.75, 13.8, 0.5), (0.25, 16.5, 0.7)], "kappa": 1.0, "copies": {},
                "total_players": 300, "owner_counts": {"171@11": 6}}  # 6 own Infected Maw at GaCr (user)
        data["kappa"] = ownership.fit_kappa(data, self.catalog)
        owners = ownership.progression_team(data, self.catalog, [(171, 11)]) * 300
        self.assertAlmostEqual(owners, 6, delta=1)

    def test_universal_tiles_raise_kappa(self):
        data = {"progression": [(1.0, 12.0, 0.4)], "kappa": 1e-3, "copies": {}, "total_players": 300,
                "owner_counts": {"1@2": "all"}}  # Archer at Platinum: no index count -> everyone owns it
        data["kappa"] = ownership.fit_kappa(data, self.catalog)
        self.assertGreaterEqual(ownership.progression_team(data, self.catalog, [(1, 2)]), ownership.UNIVERSAL - 0.01)


if __name__ == "__main__":
    unittest.main()
