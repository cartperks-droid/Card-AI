import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from card_engine.simulator.catalog_rules import BLUE_SUPPORTS, SUPPORTED

from card_engine.catalog import load_catalog
from card_engine.selfplay.evaluator import MatchEvaluation, TeamSpec, evaluate_crossplay
from card_engine.simulator.catalog_rules import UnsupportedCardError
from card_engine.simulator.native import build_library
from card_engine.simulator.reference import Options


class CrossPlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = build_library(Path(cls.temp.name) / "test_kernel.so")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_batched_native_matches_python_with_independent_seeds(self):
        teams = (TeamSpec((3, 205), (1, 1)), TeamSpec((1,), (1,)), TeamSpec((44,), (1,)))
        for mode in ("branch", "sample"):
            args = dict(options=Options(mode=mode, seed=2**64-2))
            py = evaluate_crossplay(self.catalog, teams, teams, backend="python", **args)
            c = evaluate_crossplay(self.catalog, teams, teams, library_path=self.library, **args)
            self.assertEqual([(m.index_a, m.index_b) for m in c.matches],
                             [(i, j) for i in range(3) for j in range(3)])
            self.assertEqual(py.matches, c.matches)
            with self.assertRaises(ValueError):
                c.training_rows()
            if mode == "sample":
                with self.assertRaisesRegex(ValueError, "sample"):
                    c.mean_win_bounds(0)
            with self.assertRaises(ValueError):
                dataclasses.replace(c, matches=c.matches[:-1])

    def test_bounds_allow_only_float_tolerance_without_inverting_interval(self):
        result = MatchEvaluation(0, 0, 1 + 5e-10, 0, 0, 0, 0, 0, "synthetic", False)
        self.assertEqual(result.win_bounds(0), (1, 1))
        with self.assertRaises(ValueError):
            dataclasses.replace(result, index_a=-1)
        with self.assertRaises(ValueError):
            dataclasses.replace(result, training_labels_allowed=1)

    def test_unresolved_mass_is_visible_and_never_renormalized(self):
        team = TeamSpec((1,), (1,))
        result = evaluate_crossplay(self.catalog, [team], [team], backend="python",
                                    options=Options(max_steps=1))
        self.assertEqual(result.mean_win_bounds(0), ((0.0, 1.0),))
        self.assertEqual(result.mean_win_bounds(1), ((0.0, 1.0),))
        hypothetical_verified = dataclasses.replace(result, matches=tuple(
            dataclasses.replace(m, training_labels_allowed=True) for m in result.matches))
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            hypothetical_verified.training_rows()

    def test_unsupported_match_rejected_before_any_native_execution(self):
        # Every card is mapped now; a changed description is what makes a card unsupported.
        with mock.patch.dict(SUPPORTED, {37: ("changed text", {})}), self.assertRaises(UnsupportedCardError):
            evaluate_crossplay(self.catalog, [TeamSpec((3,), (1,))], [TeamSpec((37,), (1,))],
                               library_path="/no/kernel")
        with mock.patch.dict(BLUE_SUPPORTS, {12: [None] * 5}), self.assertRaises(UnsupportedCardError):
            evaluate_crossplay(self.catalog, [TeamSpec((3,), (1,), blue_support=12)],
                               [TeamSpec((3,), (1,))], library_path="/no/kernel")

    def test_explicit_ids_order_and_bounds(self):
        team = TeamSpec((3, 205), (1, 2))
        self.assertEqual(team.cards, (3, 205))
        for cards, borders in (((True,), (1,)), ((0,), (1,)), ((3,), (0,)), ((3,), (1, 2))):
            with self.assertRaises(ValueError):
                TeamSpec(cards, borders)
        with self.assertRaises(ValueError):
            evaluate_crossplay(self.catalog, [team], [team], max_matches=0)
        with self.assertRaises(ValueError):
            evaluate_crossplay(self.catalog, [], [team])


if __name__ == "__main__":
    unittest.main()
