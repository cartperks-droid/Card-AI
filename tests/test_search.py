import random
import unittest

import torch

from card_engine.model import StrategicConfig, StrategicModel
from card_engine.selfplay.search import (
    CardVariant, LegalInventory, SearchConfig, generate_candidates,
)


class LinearStrategy(StrategicModel):
    """Use real metadata/slot assembly with an interpretable synthetic objective."""

    def __init__(self):
        super().__init__(StrategicConfig(width=8, layers=1, heads=2, attention_width=4,
                                         feedforward_width=16, class_names=("dragon",)))
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()

    def forward(self, cards, borders, red, blue, **metadata):
        sequence = self.build_sequence(cards, borders, red, blue, **metadata)
        a = sequence[:, 3:7, 0].mean(-1)
        b = sequence[:, 9:13, 0].mean(-1)
        return torch.stack((a, b), dim=-1)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.model = LinearStrategy()
        self.table = torch.zeros(5, 8)
        self.table[:, 0] = torch.arange(5)
        self.variants = tuple(CardVariant(i, 1) for i in range(1, 6))
        self.seed = (self.variants[0],) * 4
        self.opponent = (self.variants[1],) * 4
        self.inventory = LegalInventory((self.variants,) * 4, "allow")

    def search(self, **changes):
        arguments = dict(seed_team=self.seed, opponent_team=self.opponent,
                         inventory=self.inventory, target_side=0,
                         config=SearchConfig(candidate_count=3, steps=2))
        arguments.update(changes)
        return generate_candidates(self.model, self.table, **arguments)

    def test_gradient_improves_chosen_side_and_rescores_discrete_team(self):
        config = SearchConfig(candidate_count=2, steps=5, learning_rate=1,
                              nearest_k=1, initial_noise=0, anchor_penalty=0)
        for side in (0, 1):
            result = self.search(target_side=side, config=config)
            self.assertTrue(all(c.team == (self.variants[4],) * 4 for c in result.candidates))
            expected = torch.tensor([4., 1.] if side == 0 else [1., 4.]).softmax(0)
            for candidate in result.candidates:
                torch.testing.assert_close(torch.tensor(candidate.model_probabilities), expected)
                self.assertEqual(candidate.target_score, candidate.model_probabilities[side])
            self.assertGreater(float(result.relaxed_embeddings[0, :, 0].mean()), 0)
            self.assertFalse(result.relaxed_embeddings.requires_grad)
            self.assertFalse(result.training_labels_allowed)
            self.assertEqual(result.score_status, "uncalibrated_model_estimates")

    def test_metadata_is_in_relaxation_and_discrete_rescore(self):
        # Only the first card's platinum-like border is legal; pack/class/support
        # add separate, easily checked contributions to the ordered model input.
        special = CardVariant(1, 4)
        team = (special, self.variants[2], self.variants[1], self.variants[4])
        inventory = LegalInventory(tuple((v,) for v in team), "unique_card")
        packs = torch.tensor([1, 0, 2, 0, 0])
        classes = torch.tensor([[1.], [0.], [0.], [0.], [1.]])
        with torch.no_grad():
            self.model.border_embedding.weight[0, 0] = 2
            self.model.border_embedding.weight[3, 0] = 7
            self.model.pack_embedding.weight[0, 0] = 11
            self.model.pack_embedding.weight[1, 0] = 13
            self.model.class_embedding.weight[0, 0] = 17
        result = self.search(seed_team=team, inventory=inventory, pack_ids=packs,
                             class_weights=classes,
                             config=SearchConfig(candidate_count=1, steps=0, initial_noise=0))
        self.assertEqual(result.candidates[0].team, team)
        torch.testing.assert_close(result.relaxed_embeddings[0, :, 0], torch.tensor([35., 17., 3., 23.]))
        expected = torch.tensor([19.5, 3.]).softmax(0)
        torch.testing.assert_close(torch.tensor(result.candidates[0].model_probabilities), expected)
        # One gradient step must be identical whether the original border-1
        # embedding is zero or nonzero: it is subtracted before model assembly.
        before = result.relaxed_embeddings.clone()
        optimized = self.search(seed_team=team, inventory=inventory, pack_ids=packs,
                                class_weights=classes,
                                config=SearchConfig(candidate_count=1, steps=1,
                                                    initial_noise=0, anchor_penalty=0))
        self.assertTrue(bool((optimized.relaxed_embeddings[:, :, 0] >= before[:, :, 0]).all()))

    def test_slot_constraints_and_matching_avoid_greedy_dead_end(self):
        a, b, c, d, _ = self.variants
        slots = ((a, b), (a,), (c,), (d,))
        inventory = LegalInventory(slots, "unique_card")
        result = self.search(seed_team=(b, a, c, d), inventory=inventory,
                             config=SearchConfig(candidate_count=12, steps=0, nearest_k=1))
        self.assertTrue(all(candidate.team == (b, a, c, d) for candidate in result.candidates))
        for candidate in result.candidates:
            self.assertTrue(inventory.permits(candidate.team))

    def test_duplicate_policy_is_explicit_and_border_variants_are_distinct(self):
        variants = tuple(CardVariant(1, border) for border in range(1, 5))
        with self.assertRaisesRegex(ValueError, "no complete team"):
            LegalInventory((variants,) * 4, "unique_card")
        inventory = LegalInventory((variants,) * 4, "unique_variant")
        result = self.search(seed_team=variants, inventory=inventory,
                             config=SearchConfig(candidate_count=4, steps=0))
        self.assertTrue(all(len(set(c.team)) == 4 for c in result.candidates))
        with self.assertRaises(TypeError):
            LegalInventory((variants,) * 4)
        with self.assertRaisesRegex(ValueError, "bias sampling"):
            LegalInventory(((variants[0], variants[0]),) * 4, "allow")

    def test_reproducible_independent_candidates_and_global_rng_preserved(self):
        before_torch = torch.random.get_rng_state().clone()
        before_random = random.getstate()
        short = self.search(config=SearchConfig(candidate_count=3, steps=3, seed=41))
        long = self.search(config=SearchConfig(candidate_count=6, steps=3, seed=41))
        again = self.search(config=SearchConfig(candidate_count=3, steps=3, seed=41))
        self.assertEqual(short.candidates, long.candidates[:3])
        self.assertEqual(short.candidates, again.candidates)
        torch.testing.assert_close(short.relaxed_embeddings, long.relaxed_embeddings[:3], rtol=0, atol=0)
        torch.testing.assert_close(short.relaxed_embeddings, again.relaxed_embeddings, rtol=0, atol=0)
        self.assertTrue(torch.equal(before_torch, torch.random.get_rng_state()))
        self.assertEqual(before_random, random.getstate())
        self.assertEqual(len({c.seed for c in long.candidates}), 6)

    def test_parameters_gradients_mixed_modes_and_input_are_preserved(self):
        self.model.train()
        self.model.blocks[0].eval()
        self.table.requires_grad_(True)
        parameters = list(self.model.parameters())
        parameters[0].requires_grad_(False)
        for p in parameters:
            p.grad = torch.full_like(p, .125)
        values = [p.detach().clone() for p in parameters]
        gradients = [p.grad for p in parameters]
        modes = [(m, m.training) for m in self.model.modules()]
        flags = [p.requires_grad for p in parameters]
        with torch.no_grad():
            self.search()
        for p, old, grad, flag in zip(parameters, values, gradients, flags):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
            self.assertIs(p.grad, grad)
            self.assertTrue(bool(p.grad.eq(.125).all()))
            self.assertEqual(p.requires_grad, flag)
        self.assertTrue(all(m.training == old for m, old in modes))
        self.assertIsNone(self.table.grad)

    def test_modes_restore_on_failure_and_inference_mode_is_rejected(self):
        self.model.train()
        self.model.blocks[0].eval()
        modes = [(m, m.training) for m in self.model.modules()]
        def fail(*args, **kwargs):
            raise RuntimeError("intentional model failure")
        self.model.forward = fail
        with self.assertRaisesRegex(RuntimeError, "intentional model failure"):
            self.search()
        self.assertTrue(all(m.training == old for m, old in modes))
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "inference_mode"):
            self.search()

    def test_invalid_inputs_fail_before_returning_candidates(self):
        for changes in ({"target_side": True}, {"seed_team": self.seed[:3]},
                        {"red_support_ids": (29, 0)}, {"blue_support_ids": (0, True)},
                        {"pack_ids": torch.zeros(4, dtype=torch.long)},
                        {"class_weights": torch.ones(5, 2)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.search(**changes)
        with self.assertRaisesRegex(ValueError, "not permitted"):
            self.search(seed_team=(CardVariant(1, 2),) * 4)
        for arguments in ({"steps": True}, {"distance_temperature": 0},
                          {"candidate_count": 0}, {"seed": -1}, {"initial_noise": float("nan")}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SearchConfig(**arguments)

    def test_real_transformer_runs_and_returns_detached_ordered_candidates(self):
        model = StrategicModel(StrategicConfig(width=8, layers=1, heads=2,
                                               attention_width=4, feedforward_width=16))
        result = generate_candidates(model, self.table, seed_team=self.seed,
                                     opponent_team=self.opponent, inventory=self.inventory,
                                     target_side=0,
                                     config=SearchConfig(candidate_count=2, steps=2))
        self.assertEqual(tuple(result.relaxed_embeddings.shape), (2, 4, 8))
        for candidate in result.candidates:
            self.assertTrue(self.inventory.permits(candidate.team))
            self.assertAlmostEqual(sum(candidate.model_probabilities), 1, places=6)


if __name__ == "__main__":
    unittest.main()
