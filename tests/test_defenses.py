"""Damage arithmetic, maximum-HP state, support scope, and native parity."""

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from card_engine.catalog import load_catalog
from card_engine.simulator import Battle, Fighter, Options, simulate
from card_engine.simulator.catalog_rules import BLUE_SUPPORTS, compile_battle, compile_fighter, support_coverage, UnsupportedCardError
from card_engine.simulator import native


def damage_events(attacker, defender, steps=1):
    result = simulate(Battle(((attacker,), (defender,))), Options(mode="sample", max_steps=steps), trace=True)
    return [event for event in result.trace if event["phase"] == "DAMAGE_APPLY" and event["side"] == 1], result


class DefenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / native.DEFAULT_LIBRARY.name)
        cls.catalog = load_catalog()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_flat_reduction_uses_max_hp_and_precedes_halving(self):
        hits, _ = damage_events(Fighter(100, 50), Fighter(100, 0, damage_reduction_max_hp=.15, incoming_multiplier=.5), 5)
        self.assertEqual([hit["damage"] for hit in hits], [18, 18, 18])
        self.assertEqual([hit["hp"] for hit in hits], [82, 64, 46])
        hits, _ = damage_events(Fighter(100, 4), Fighter(100, 0, damage_reduction_max_hp=.1))
        self.assertEqual(hits[0]["damage"], 0)
        self.assertEqual(hits[0]["hp"], 100)

    def test_cap_stays_fixed_after_damage_and_rounds_at_hit_boundary(self):
        hits, _ = damage_events(Fighter(100, 100), Fighter(101, 0, damage_cap_max_hp=1/3), 3)
        self.assertEqual([hit["damage"] for hit in hits], [34, 34])
        self.assertEqual([hit["hp"] for hit in hits], [67, 33])

    def test_threshold_boundaries_are_strict_and_do_not_consume_block(self):
        target = Fighter(100, 0, dodge_below_max_hp=.3)
        hits, result = damage_events(Fighter(100, 29), target)
        self.assertFalse(hits)
        self.assertIn("threshold_dodge", [e.get("outcome") for e in result.trace])
        hits, _ = damage_events(Fighter(100, 30), target)
        self.assertEqual(hits[0]["damage"], 30)
        target = Fighter(100, 40, nullify_below_attack=.75)
        self.assertFalse(damage_events(Fighter(100, 29.9), target)[0])
        self.assertEqual(damage_events(Fighter(100, 30), target)[0][0]["damage"], 30)
        target = replace(target, block_mode=1)
        _, result = damage_events(Fighter(100, 29), target, 3)
        self.assertNotIn("block", [e.get("outcome") for e in result.trace])

    def test_separate_entry_scaling_updates_max_hp_before_defense(self):
        target = Fighter(100, 3, entry_hp_multiplier=3, entry_attack_multiplier=2, damage_cap_max_hp=.1)
        hits, result = damage_events(Fighter(100, 100), target)
        entries = [e for e in result.trace if e["phase"] == "ENTRY"]
        self.assertEqual((entries[1]["hp"], entries[1]["max_hp"], entries[1]["attack"]), (300, 300, 6))
        self.assertEqual(hits[0]["damage"], 30)

    def test_shielder_applies_to_each_ally_and_combines_once(self):
        battle = compile_battle(self.catalog, ((38, 165), (3,)), blue_supports=(1, 0))
        # The blue support is an aura on each ally (Marrowclaw can negate it), on top of the card's own multiplier.
        self.assertEqual((battle.teams[0][0].incoming_multiplier, battle.teams[0][0].aura_incoming_multiplier), (.25, .98))
        self.assertEqual((battle.teams[0][1].incoming_multiplier, battle.teams[0][1].aura_incoming_multiplier), (1.5, .98))
        self.assertEqual((battle.teams[1][0].incoming_multiplier, battle.teams[1][0].aura_incoming_multiplier), (1, 1))
        py = simulate(battle)
        c = native.simulate(battle, library_path=self.library)
        self.assertAlmostEqual(py.p_a, c["p_a"])
        for value in (True, 1.0, -1):
            with self.assertRaises(ValueError):
                compile_battle(self.catalog, ((3,), (3,)), blue_supports=(value, 0))
        with mock.patch.dict(BLUE_SUPPORTS, {12: [None] * 5}), self.assertRaises(UnsupportedCardError):
            compile_battle(self.catalog, ((3,), (3,)), blue_supports=(12, 0))  # an unread value is refused, not guessed
        support_rows = support_coverage(self.catalog)
        self.assertEqual(len(support_rows), 43)
        # Every support (all tiers read or supplied by the user, 2026-10-01).
        self.assertEqual([(r["color"], r["support_id"]) for r in support_rows if r["status"] == "unsupported"], [])

    def test_complete_source_descriptions_compile_new_abilities(self):
        for card in (2, 9, 36, 47, 59, 61, 71, 87, 93, 131, 165, 166, 182, 261, 264, 288):
            with self.subTest(card=card):
                compile_fighter(self.catalog, card).validate()

    def test_randomized_defense_and_max_hp_native_parity(self):
        rng = random.Random(19)
        battles = [Battle(tuple(tuple(Fighter(
                    hp=rng.randint(15, 50), attack=rng.randint(1, 20),
                    entry_hp_multiplier=rng.choice((.5, 1, 3)), entry_attack_multiplier=rng.choice((.5, 1, 2)),
                    entry_self_multiplier=rng.choice((1, 1.25)), enemy_entry_steal_fraction=rng.choice((0, .1)),
                    damage_reduction_max_hp=rng.choice((0, .1, .15)), damage_cap_max_hp=rng.choice((0, 1/3, .5)),
                    dodge_below_max_hp=rng.choice((0, .3)), nullify_below_attack=rng.choice((0, .75)),
                    dodge_probability=rng.choice((0, .1, .6)), block_mode=rng.randrange(3),
                    incoming_multiplier=rng.choice((.5, .98, 1, 1.5)),
                ) for _ in range(rng.randint(1, 3))) for _ in range(2))) for _ in range(24)]
        for mode in ("sample", "branch"):
            options = Options(mode=mode, max_steps=40, repeat_cycles=3, max_frontier=30, prune_probability=1e-7)
            actual = native.simulate_batch(battles, options, library_path=self.library)
            for index, (battle, c) in enumerate(zip(battles, actual, strict=True)):
                py = simulate(battle, replace(options, seed=options.seed + index))
                for field in ("p_a", "p_b", "tie", "unresolved", "expanded_states", "merged_states"):
                    self.assertAlmostEqual(getattr(py, field), c[field], places=11, msg=f"case {index}/{mode}/{field}")

    def test_defense_overflow_is_not_masked_by_cap_or_clamp(self):
        for target in (Fighter(1e300, 0, damage_reduction_max_hp=1e300),
                       Fighter(10, 0, incoming_multiplier=1e300, damage_cap_max_hp=.5)):
            battle = Battle(((Fighter(10, 1e300),), (target,)))
            with self.assertRaises(ValueError):
                simulate(battle)
            with self.assertRaises(native.NativeSimulationError):
                native.simulate(battle, library_path=self.library)


if __name__ == "__main__":
    unittest.main()
