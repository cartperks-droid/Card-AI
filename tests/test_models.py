"""Model contracts only: no synthetic battles are treated as training labels."""

from pathlib import Path
import tempfile
import unittest

import torch

from card_engine.model import (
    BattleModel, DescriptionConfig, DescriptionModel, NarrowSelfAttention,
    SLOT_NAMES, StrategicConfig, StrategicModel, load_checkpoint, load_model_data,
    pad_descriptions, precompute_embeddings, save_checkpoint,
)


def description_config():
    return DescriptionConfig(width=16, layers=2, heads=4, feedforward_width=32,
                             projection_hidden_width=32, output_width=32)


def strategy_config():
    return StrategicConfig(width=32, layers=2, heads=4, attention_width=16,
                           feedforward_width=64, class_names=("dragon", "rng"))


def example_inputs(batch=2):
    return {
        "card_embeddings": torch.randn(batch, 2, 4, 32),
        "border_ids": torch.ones(batch, 2, 4, dtype=torch.long),
        "red_support_ids": torch.ones(batch, 2, dtype=torch.long),
        "blue_support_ids": torch.zeros(batch, 2, dtype=torch.long),
        "pack_ids": torch.ones(batch, 2, 4, dtype=torch.long),
        "class_weights": torch.zeros(batch, 2, 4, 2),
    }


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(19)

    def test_canonical_dataset_and_class_evidence_policy(self):
        data = load_model_data(class_names=("dragon", "rng"))
        self.assertEqual(len(data.names), 289)
        self.assertEqual(data.description_tokens.shape[0], 289)
        self.assertTrue(bool(data.description_tokens[:, 0].eq(439).all()))
        self.assertTrue(bool(data.description_tokens.eq(440).sum(-1).eq(1).all()))
        self.assertTrue(bool(data.pack_ids.ge(1).all() & data.pack_ids.le(14).all()))
        # User-verified memberships (2026-09-30): Archer is an RNG card.
        self.assertEqual(data.class_weights[0, 1].item(), 1.0)
        self.assertEqual(load_model_data(class_names=("rng",), class_policy="disabled").class_weights[0, 0].item(), 0.0)
        wiki = load_model_data(class_names=("dragon", "rng"), class_policy="wiki_supported")
        self.assertEqual(wiki.class_weights[0, 1].item(), 1.0)
        gathered = data.gather(torch.tensor([3, 1]))
        torch.testing.assert_close(gathered["description_tokens"][1], data.description_tokens[0])
        with self.assertRaises(ValueError):
            data.gather(torch.tensor([0]))

    def test_padding_and_batch_neighbors_do_not_change_card_representation(self):
        model = DescriptionModel(description_config()).eval()
        short = [439, 3, 4, 440]
        long = [439, 11, 12, 13, 14, 15, 440]
        with torch.no_grad():
            unpadded = model(pad_descriptions([short]))[0]
            padded = model(pad_descriptions([short], pad_to=15))[0]
            mixed = model(pad_descriptions([short, long], pad_to=15))[0]
        torch.testing.assert_close(unpadded, padded, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(unpadded, mixed, rtol=1e-5, atol=1e-6)

    def test_description_attention_is_causal(self):
        attention = NarrowSelfAttention(16, 16, 4).eval()
        before = torch.randn(1, 8, 16)
        after = before.clone()
        after[:, 4:] = torch.randn_like(after[:, 4:]) * 100
        torch.testing.assert_close(attention(before, causal=True)[:, :4],
                                   attention(after, causal=True)[:, :4])

    def test_description_boundaries_and_ids_are_checked(self):
        model = DescriptionModel(description_config())
        invalid = [
            [439, 1, 0, 440],  # padding before CARD
            [439, 440, 1, 0],  # content after CARD
            [439, 440, 440, 0],
            [439, 439, 440, 0],
            [439, 441, 440, 0],
            [1, 2, 440, 0],
            [0, 0, 0, 0],
        ]
        for ids in invalid:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                model(torch.tensor([ids]))
        with self.assertRaises(ValueError):
            model(torch.tensor([[439.0, 440.0]]))

    def test_requested_default_architectures_and_three_outcomes(self):
        description = DescriptionModel()
        self.assertEqual(description.encoder.token_embedding.weight.shape, (441, 128))
        self.assertEqual(len(description.encoder.blocks), 4)
        self.assertEqual(description.encoder.blocks[0].attention.head_width, 32)
        self.assertEqual(description.projection[-1].out_features, 768)
        source_data = load_model_data()
        encoded = precompute_embeddings(description, source_data.description_tokens)
        self.assertEqual(encoded.shape, (289, 768))
        self.assertTrue(bool(torch.isfinite(encoded).all()))
        strategy = StrategicModel().eval()
        self.assertEqual(strategy.border_embedding.num_embeddings, 16)
        self.assertEqual(strategy.red_support_embedding.num_embeddings, 28)
        self.assertEqual(strategy.blue_support_embedding.num_embeddings, 15)
        self.assertEqual(strategy.pack_embedding.num_embeddings, 14)
        self.assertEqual(len(strategy.blocks), 4)
        self.assertEqual(strategy.blocks[0].attention.head_width, 24)
        self.assertEqual(strategy.blocks[0].feedforward[0].out_features, 3072)
        self.assertEqual(len(SLOT_NAMES), 15)
        with torch.no_grad():
            probabilities = strategy.probabilities(
                torch.randn(1, 2, 4, 768), torch.ones(1, 2, 4, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long))
        self.assertEqual(probabilities.shape, (1, 2))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))

    def test_hidden_enemy_identity_and_all_metadata_cannot_leak(self):
        model = StrategicModel(strategy_config()).eval()
        inputs = example_inputs()
        inputs["card_visible"] = torch.ones(2, 2, 4, dtype=torch.bool)
        inputs["card_visible"][:, 1] = False
        inputs["support_visible"] = torch.ones(2, 2, 2, dtype=torch.bool)
        inputs["support_visible"][:, 1] = False
        baseline = model(**inputs)
        changed = {name: value.clone() for name, value in inputs.items()}
        # Hidden content is ignored even if it would be invalid as visible content.
        changed["card_embeddings"][:, 1] = float("nan")
        changed["border_ids"][:, 1] = 999
        changed["pack_ids"][:, 1] = -50
        changed["class_weights"][:, 1] = float("nan")
        changed["red_support_ids"][:, 1] = 999
        changed["blue_support_ids"][:, 1] = -7
        torch.testing.assert_close(baseline, model(**changed), rtol=0, atol=0)
        # Visible metadata still matters; this prevents an always-ignore-metadata false pass.
        changed["border_ids"][:, 0, 0] = 2
        self.assertFalse(torch.allclose(baseline, model(**changed)))

    def test_gradients_reach_visible_search_slots_but_not_hidden_slots(self):
        model = StrategicModel(strategy_config()).eval()
        inputs = example_inputs()
        inputs["card_embeddings"].requires_grad_()
        inputs["card_visible"] = torch.ones(2, 2, 4, dtype=torch.bool)
        inputs["card_visible"][:, 1, 2:] = False
        # This differentiability probe is not a battle target or training example.
        model(**inputs)[:, 0].sum().backward()
        gradient = inputs["card_embeddings"].grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(gradient[:, 0].abs().sum().item(), 0.0)
        self.assertEqual(gradient[:, 1, 2:].abs().sum().item(), 0.0)

    def test_fixed_sequence_retains_card_positions(self):
        model = StrategicModel(strategy_config()).eval()
        inputs = example_inputs(batch=1)
        baseline = model.build_sequence(**inputs)
        shifted = {name: value.clone() for name, value in inputs.items()}
        shifted["card_embeddings"][:, 0, 1, :] += 2
        delta = model.build_sequence(**shifted) - baseline
        # Ally card 2 is exactly sequence index 4; no implicit sorting/permutation.
        torch.testing.assert_close(delta[:, 4], torch.full((1, 32), 2.0))
        delta[:, 4] = 0
        torch.testing.assert_close(delta, torch.zeros_like(delta))

    def test_visible_shape_range_and_metadata_checks(self):
        model = StrategicModel(strategy_config())
        for key, value in [("border_ids", 0), ("red_support_ids", 29),
                           ("blue_support_ids", 16), ("pack_ids", 15)]:
            inputs = example_inputs()
            inputs[key].fill_(value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                model(**inputs)
        inputs = example_inputs()
        inputs["card_visible"] = torch.ones(2, 2, 4)  # wrong dtype
        with self.assertRaises(ValueError):
            model(**inputs)

    def test_composed_model_ignores_hidden_description_content(self):
        model = BattleModel(description_config(), strategy_config()).eval()
        inputs = example_inputs(batch=1)
        del inputs["card_embeddings"]
        inputs["description_tokens"] = torch.tensor([439, 1, 440, 0]).expand(1, 2, 4, 4).clone()
        inputs["card_visible"] = torch.ones(1, 2, 4, dtype=torch.bool)
        inputs["card_visible"][:, 1] = False
        before = model(**inputs)
        inputs["description_tokens"][:, 1] = -123
        torch.testing.assert_close(before, model(**inputs), rtol=0, atol=0)

    def test_checkpoint_round_trip_and_embedding_snapshot(self):
        model = BattleModel(description_config(), strategy_config()).eval()
        inputs = example_inputs(batch=1)
        before = model(**inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path, metadata={"purpose": "contract test", "labels": None})
            restored, metadata = load_checkpoint(path)
            self.assertEqual(metadata["purpose"], "contract test")
            self.assertFalse(restored.training)
            self.assertEqual(restored.strategic_config.class_names, ("dragon", "rng"))
            torch.testing.assert_close(before, restored(**inputs), rtol=0, atol=0)
            description = DescriptionModel(description_config()).double().eval()
            tokens = pad_descriptions([[439, 1, 440]])
            save_checkpoint(description, path)
            restored_description, _ = load_checkpoint(path)
            self.assertEqual(next(restored_description.parameters()).dtype, torch.float64)
            torch.testing.assert_close(description(tokens), restored_description(tokens), rtol=0, atol=0)
        descriptions = pad_descriptions([[439, 1, 440], [439, 2, 3, 440]])
        model.description.train()
        cached = precompute_embeddings(model.description, descriptions, batch_size=1)
        self.assertEqual(cached.shape, (2, 32))
        self.assertFalse(cached.requires_grad)
        self.assertTrue(model.description.training)
        with torch.no_grad():
            torch.testing.assert_close(cached, model.description(descriptions))


if __name__ == "__main__":
    unittest.main()
