import unittest
from unittest import mock

from card_engine.simulator.catalog_rules import SUPPORTED

import torch

from card_engine.catalog import load_catalog
from card_engine.model import StrategicConfig, StrategicModel, load_model_data
from card_engine.selfplay.pipeline import run_diagnostic_cycle
from card_engine.selfplay.search import CardVariant, LegalInventory, SearchConfig
from card_engine.simulator.catalog_rules import UnsupportedCardError


class PipelineTests(unittest.TestCase):
    def test_search_to_crossplay_preserves_legal_order_and_blocks_labels(self):
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(40)
            model = StrategicModel(StrategicConfig(width=16, layers=1, heads=2,
                                    attention_width=8, feedforward_width=32))
            embeddings = torch.randn(289, 16)
        team = tuple(CardVariant(c, 1) for c in (3, 38, 44, 205))
        inventory = LegalInventory((team,) * 4, "unique_card")
        cycle = run_diagnostic_cycle(model, embeddings, load_catalog(), seed_team_a=team,
                   seed_team_b=team[::-1], inventory_a=inventory, inventory_b=inventory,
                   search_config=SearchConfig(candidate_count=2, steps=1, seed=17),
                   pack_ids=load_model_data().pack_ids, backend="python")
        self.assertEqual(len(cycle.crossplay.matches), 4)
        self.assertEqual(cycle.search_a.target_side, 0)
        self.assertEqual(cycle.search_b.target_side, 1)
        self.assertNotEqual(cycle.search_a.candidates[0].seed, cycle.search_b.candidates[0].seed)
        for result, teams in ((cycle.search_a, cycle.crossplay.teams_a),
                              (cycle.search_b, cycle.crossplay.teams_b)):
            for candidate, compiled in zip(result.candidates, teams, strict=True):
                self.assertTrue(inventory.permits(candidate.team))
                self.assertEqual(compiled.cards, tuple(v.card_id for v in candidate.team))
                self.assertEqual(compiled.borders, tuple(v.border_id for v in candidate.team))
        for match in cycle.crossplay.matches:
            self.assertEqual(match.unresolved, 0)
            self.assertAlmostEqual(match.p_a + match.p_b + match.tie, 1)
        with self.assertRaisesRegex(ValueError, "Experimental"):
            cycle.crossplay.training_rows()

    def test_unsupported_inventory_fails_before_model_execution(self):
        team = tuple(CardVariant(c, 1) for c in (3, 38, 44, 205))
        slots = (team + (CardVariant(37, 1),),) * 4
        inventory = LegalInventory(slots, "unique_card")
        with mock.patch.dict(SUPPORTED, {37: ("changed text", {})}), self.assertRaises(UnsupportedCardError):
            run_diagnostic_cycle(None, None, load_catalog(), seed_team_a=team, seed_team_b=team,
                inventory_a=inventory, inventory_b=inventory, backend="python")


if __name__ == "__main__":
    unittest.main()
