"""Mutation identity stays consistent across masked models, search and battles."""

from pathlib import Path
import tempfile
import unittest

import torch

from card_engine.catalog import load_catalog
from card_engine.model import StrategicConfig, StrategicModel, save_checkpoint, load_checkpoint
from card_engine.selfplay.search import CardVariant, LegalInventory, SearchConfig, generate_candidates
from card_engine.selfplay.pipeline import run_diagnostic_cycle
from card_engine.simulator.catalog_rules import compile_battle


class MutationIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = StrategicConfig(width=8, layers=1, heads=2, attention_width=4, feedforward_width=16)
        self.model = StrategicModel(self.config)
        self.cards = torch.zeros(1, 2, 4, 8)
        self.borders = torch.ones(1, 2, 4, dtype=torch.long)
        self.supports = torch.zeros(1, 2, dtype=torch.long)

    def sequence(self, **metadata):
        return self.model.build_sequence(self.cards, self.borders, self.supports, self.supports, **metadata)

    def test_none_is_neutral_and_visible_mutation_changes_only_its_slot(self):
        baseline = self.sequence()
        ids = torch.zeros(1, 2, 4, dtype=torch.long)
        torch.testing.assert_close(self.sequence(mutation_ids=ids), baseline, rtol=0, atol=0)
        ids[0, 0, 1] = self.config.mutation_names.index("Storm")
        difference = self.sequence(mutation_ids=ids) - baseline
        torch.testing.assert_close(difference[0, 4], self.model.mutation_embedding.weight[ids[0, 0, 1]])
        difference[0, 4] = 0
        self.assertTrue(torch.equal(difference, torch.zeros_like(difference)))

    def test_hidden_mutations_cannot_leak_or_receive_gradients(self):
        visible = torch.ones(1, 2, 4, dtype=torch.bool)
        visible[:, 1] = False
        ids = torch.zeros(1, 2, 4, dtype=torch.long)
        ids[0, 0, 0] = 1
        baseline = self.sequence(mutation_ids=ids, card_visible=visible)
        ids[:, 1] = 999999
        hidden = self.sequence(mutation_ids=ids, card_visible=visible)
        torch.testing.assert_close(hidden, baseline, rtol=0, atol=0)
        hidden.sum().backward()
        grad = self.model.mutation_embedding.weight.grad
        self.assertGreater(float(grad[1].abs().sum()), 0)
        self.assertEqual(float(grad[2:].abs().sum()), 0)
        ids[0, 0, 0] = 999999
        with self.assertRaisesRegex(ValueError, "mutation_ids"):
            self.sequence(mutation_ids=ids)

    def test_search_combines_and_rescores_actual_mutation_variants(self):
        table = torch.zeros(4, 8)
        names = ("None", "Storm", "Snow", "Manga")
        variants = tuple(CardVariant(1, 1, name) for name in names)
        inventory = LegalInventory((variants,) * 4, "unique_variant")
        result = generate_candidates(self.model, table, seed_team=variants, opponent_team=variants,
                 inventory=inventory, target_side=0, intrinsic_weathers=("None",)*4,
                 config=SearchConfig(candidate_count=1, steps=0, initial_noise=0, nearest_k=1))
        self.assertEqual(result.candidates[0].team, variants)
        ids = torch.tensor([[self.config.mutation_names.index(v.mutation) for v in variants]] * 2).unsqueeze(0)
        with torch.no_grad():
            expected = self.model(self.cards, self.borders, self.supports, self.supports, mutation_ids=ids).softmax(-1)[0]
            fused = self.model.border_embedding.weight[0].expand(4, -1).clone()
            fused[1:] += self.model.mutation_embedding(ids[0, 0, 1:])
        torch.testing.assert_close(torch.tensor(result.candidates[0].model_probabilities), expected)
        torch.testing.assert_close(result.relaxed_embeddings[0], fused)
        with self.assertRaisesRegex(ValueError, "no complete team"):
            LegalInventory((variants,) * 4, "unique_card")

    def test_pipeline_keeps_mutations_through_compilation(self):
        variants = tuple(CardVariant(card, 1, name) for card, name in zip((3, 67, 73, 205), ("Storm", "Snow", "None", "Manga")))
        inventory = LegalInventory(tuple((v,) for v in variants), "unique_card")
        cycle = run_diagnostic_cycle(self.model, torch.zeros(289, 8), load_catalog(),
                  seed_team_a=variants, seed_team_b=variants, inventory_a=inventory, inventory_b=inventory,
                  search_config=SearchConfig(candidate_count=1, steps=0), backend="python")
        team = cycle.crossplay.teams_a[0]
        self.assertEqual(team.mutations, ("Storm", "Snow", "None", "Manga"))
        battle = compile_battle(load_catalog(), (team.cards, team.cards), mutations=(team.mutations, team.mutations))
        self.assertEqual((battle.teams[0][1].attack, battle.teams[0][1].hp), (78, 156))

    def test_checkpoint_pins_mutation_names_and_rejects_old_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(self.model, path)
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.mutation_names, self.config.mutation_names)
            torch.testing.assert_close(restored.mutation_embedding.weight, self.model.mutation_embedding.weight)
            legacy = StrategicModel(StrategicConfig(width=8, layers=1, heads=2, attention_width=4,
                                                  feedforward_width=16, mutation_names=()))
            save_checkpoint(legacy, path)
            restored, _ = load_checkpoint(path)
            self.assertIsNone(restored.mutation_embedding)
            torch.testing.assert_close(restored(self.cards, self.borders, self.supports, self.supports),
                                       legacy(self.cards, self.borders, self.supports, self.supports))
            payload = torch.load(path, weights_only=True)
            payload["schema_version"] = 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "schema"):
                load_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "mutation.*vocabulary"):
                variants = (CardVariant(1, 1, "Storm"),) * 4
                generate_candidates(restored, torch.zeros(4, 8), seed_team=variants, opponent_team=variants,
                                    inventory=LegalInventory(((variants[0],),) * 4, "allow"), target_side=0,
                                    intrinsic_weathers=("None",)*4)

    def test_weather_cards_are_ineligible_and_never_become_mutation_tokens(self):
        from card_engine.stats import base_stats
        stats = base_stats(load_catalog(), 33)
        self.assertEqual((stats.hp, stats.attack), (2560, 1280))
        self.assertEqual(stats.intrinsic_weather_multiplier, 3)
        self.assertEqual((stats.mutation, stats.mutation_multiplier, stats.mutation_eligible), ("None", 1, False))
        for name in ("Storm", "Rapture"):
            with self.assertRaises(ValueError):
                base_stats(load_catalog(), 33, mutation=name)
        team = (CardVariant(1, 1, "Storm"),)*4
        inventory = LegalInventory(((team[0],),)*4, "allow")
        with self.assertRaisesRegex(ValueError, "eligibility"):
            generate_candidates(self.model, torch.zeros(4, 8), seed_team=team, opponent_team=team,
                                inventory=inventory, target_side=0)
        with self.assertRaisesRegex(ValueError, "ineligible"):
            generate_candidates(self.model, torch.zeros(4, 8), seed_team=team, opponent_team=team,
                                inventory=inventory, target_side=0, intrinsic_weathers=("Storm", "None", "None", "None"))


if __name__ == "__main__":
    unittest.main()
