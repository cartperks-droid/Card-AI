"""Differential checks exercise the actual compiled C/Python boundary."""

from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import Battle, Fighter, Options, simulate_batch
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator import native


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / native.DEFAULT_LIBRARY.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options):
        expected = simulate_batch(battles, options)
        actual = native.simulate_batch([asdict(b) for b in battles], asdict(options), library_path=self.library)
        for index, (py, c) in enumerate(zip(expected, actual, strict=True)):
            for field in ("p_a", "p_b", "tie", "unresolved"):
                self.assertAlmostEqual(getattr(py, field), c[field], places=11, msg=f"battle {index} {field}")
            for field in ("expanded_states", "merged_states"):
                self.assertEqual(getattr(py, field), c[field], f"battle {index} {field}")

    def test_user_observations_and_replacement_theft(self):
        catalog = load_catalog()
        cases = [compile_battle(catalog, (a, b), red_supports=(0, 0)) for a, b in (
            ((3,), (3,)), ((205,), (44,)), ((3, 205), (44,)), ((205, 38), (44, 3)), ((67,), (73,)),
        )]
        cases.append(compile_battle(catalog, ((67, 67), (73,)), mutations=(("Storm", "Snow"), ("None",))))
        self.compare(cases, Options())

    def test_default_stalemate_cutoff_is_100_ordinary_turns(self):
        self.assertEqual(Options().repeat_cycles, 50)
        # Synthetic zero-damage mirror isolates the timer from the unresolved
        # Witch stats/healing basis in the user's recording.
        for first in (0, 1):
            fighter = Fighter(10, 0)
            battle = Battle(((fighter,), (fighter,)), first)
            for steps in (99, 100):
                py = simulate_batch([battle], Options(max_steps=steps))[0]
                c = native.simulate(battle, {"max_steps": steps}, library_path=self.library)
                self.assertEqual(py.expanded_states, steps)
                self.assertEqual(py.unresolved, 1 if steps == 99 else 0)
                for key in ("p_a", "p_b", "tie", "unresolved", "expanded_states"):
                    self.assertEqual(getattr(py, key), c[key])
                if steps == 100:
                    self.assertEqual((py.p_a, py.p_b), (first, 1-first))
            self.assertEqual(native.simulate(battle, library_path=self.library)["expanded_states"], 100)

    def test_randomized_branch_and_sample_parity(self):
        rng = random.Random(713)
        cases = []
        for _ in range(48):
            teams = []
            for side in range(2):
                teams.append(tuple(Fighter(
                    hp=rng.randint(15, 30), attack=rng.randint(1, 10),
                    dodge_probability=rng.choice((0, .1, .5, 1)),
                    outgoing_multiplier=rng.choice((.5, 1, 2)), incoming_multiplier=rng.choice((0, .25, 1, 2)),
                    entry_self_multiplier=rng.choice((1, 1.25, 1.4)),
                    entry_enemy_attack_multiplier=rng.choice((.85, 1)),
                    enemy_entry_steal_fraction=rng.choice((0, .1)),
                    block_mode=rng.randrange(3),
                ) for _ in range(rng.randint(1, 3))))
            cases.append(Battle(tuple(teams), rng.randrange(2)))
        for mode in ("sample", "branch"):
            for stat_rounding in ("unrounded", "ceil"):
                with self.subTest(mode=mode, stat_rounding=stat_rounding):
                    self.compare(cases, Options(mode=mode, stat_rounding=stat_rounding, max_steps=60,
                                               max_frontier=40, repeat_cycles=3, prune_probability=1e-6, seed=53))

    def test_multi_hit_randomized_sample_and_branch_parity(self):
        rng = random.Random(211)
        cases = []
        for _ in range(24):
            # High HP avoids the intentionally unsupported mid-action KO path.
            teams = tuple(tuple(Fighter(
                hp=10000, attack=rng.randint(1, 10), attacks_per_action=rng.choice((1, 2, 3, 16)),
                dodge_probability=rng.choice((0, .1, .5, 1)),
                outgoing_multiplier=rng.choice((.5, 1)), incoming_multiplier=rng.choice((0, .25, 1)),
                entry_self_multiplier=rng.choice((1, 1.4)), entry_enemy_attack_multiplier=rng.choice((.85, 1)),
                block_mode=rng.randrange(3),
            ) for _ in range(rng.randint(1, 2))) for _ in range(2))
            cases.append(Battle(teams, rng.randrange(2)))
        for mode in ("sample", "branch"):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=90, max_frontier=50,
                                           repeat_cycles=1, prune_probability=1e-6, seed=31))
        probability_case = Battle(((Fighter(1, 1, attacks_per_action=3),), (Fighter(3, 1, dodge_probability=.5),)))
        self.compare([probability_case], Options())

    def test_multi_hit_knockout_cancels_pending_hits(self):
        for enemy in ((Fighter(1, 1),), (Fighter(1, 1), Fighter(10, 10))):
            battle = Battle(((Fighter(10, 10, attacks_per_action=3),), enemy))
            for mode in ("sample", "branch"):
                options = Options(mode=mode)
                with self.subTest(enemy_count=len(enemy), mode=mode):
                    self.compare([battle], options)
                    result = simulate_batch([battle], options)[0]
                    self.assertEqual(result.p_a, 1 if len(enemy) == 1 else 0)

    def test_multi_hit_count_rejected_before_ctypes_conversion(self):
        for count in (0, -1, 17, 2.5, True, 2**32 + 1):
            battle = Battle(((Fighter(10, 1, attacks_per_action=count),), (Fighter(10, 1),)))
            with self.subTest(count=count):
                with self.assertRaisesRegex(native.NativeSimulationError, "attacks_per_action"):
                    native.simulate_batch([battle], library_path=self.library)

    def test_merge_prune_and_all_rounding_modes(self):
        fighter = Fighter(5, 1.4, dodge_probability=.5, incoming_multiplier=0)
        battle = Battle(((fighter,), (fighter,)))
        for rounding in ("ceil", "floor", "unrounded", "nearest_half_up"):
            self.compare([battle], Options(rounding=rounding, repeat_cycles=2, max_frontier=1))
        self.compare([Battle(((Fighter(1, 1),), (Fighter(1, 1, dodge_probability=.1),)))], Options(prune_probability=.2))

    def test_underflowed_probability_branches_are_not_expanded(self):
        fighter = Fighter(5, 1, dodge_probability=1e-200)
        battle = Battle(((fighter,), (fighter,)))
        options = Options(max_steps=20, repeat_cycles=10)
        self.compare([battle], options)
        result = simulate_batch([battle], options)[0]
        self.assertGreater(result.p_b, 0)
        self.assertLess(result.expanded_states, 50)

    def test_entry_self_scaling_cannot_create_zero_hp_attacker(self):
        battle = Battle(((Fighter(.2, 20, entry_self_multiplier=.5),), (Fighter(10, 1),)))
        for rounding in ("floor", "nearest_half_up"):
            options = Options(stat_rounding=rounding)
            with self.subTest(rounding=rounding):
                with self.assertRaisesRegex(ValueError, "Lethal entry self scaling"):
                    simulate_batch([battle], options)
                with self.assertRaisesRegex(native.NativeSimulationError, "Unsupported lethal entry"):
                    native.simulate_batch([battle], options, library_path=self.library)

    def test_invalid_inputs_fail_before_ctypes_truncation(self):
        battle = {"teams": [[{"hp": 10, "attack": 3}], [{"hp": 10, "attack": 3}]], "first_side": 0}
        for options in ({"repeat_cycles": 0}, {"seed": -1}, {"max_frontier": 2**32}, {"max_steps": True},
                        {"prune_probability": False}, {"prune_probability": True}):
            with self.assertRaises(native.NativeSimulationError):
                native.simulate_batch([battle], options, library_path=self.library)
        battle["teams"][0][0]["unimplemented_revive"] = 1
        with self.assertRaises(native.NativeSimulationError):
            native.simulate_batch([battle], library_path=self.library)


if __name__ == "__main__":
    unittest.main()
