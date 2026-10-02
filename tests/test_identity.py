"""Permanent card identity keys and the learned identity embedding."""

import json
import tempfile
from pathlib import Path
import unittest

import torch

from card_engine.card_keys import assign_identity_keys
from card_engine.model.config import StrategicConfig
from card_engine.model.data import load_model_data
from card_engine.model.networks import StrategicModel


class IdentityTests(unittest.TestCase):
    def test_inserted_card_does_not_shift_existing_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "keys.json"
            self.assertEqual(assign_identity_keys(["Archer", "Good Boy", "Raze"], path), [1, 2, 3])
            # A new card inserted mid-list (shifting list positions) takes the next key.
            self.assertEqual(assign_identity_keys(["Archer", "New Card", "Good Boy", "Raze"], path), [1, 4, 2, 3])
            # Removed cards retire their key; it is never reused.
            self.assertEqual(assign_identity_keys(["Archer", "Raze", "Newer"], path), [1, 3, 5])
            table = json.loads(path.read_text())
            table["aliases"]["archer the second"] = "archer"
            path.write_text(json.dumps(table))
            self.assertEqual(assign_identity_keys(["Archer The Second"], path), [1])  # Recorded rename.

    def test_dataset_carries_keys_and_model_uses_them(self):
        data = load_model_data()
        self.assertEqual(len(set(data.identity_keys.tolist())), len(data.names))
        model = StrategicModel(StrategicConfig(width=32, layers=1, heads=2, attention_width=16, feedforward_width=64)).eval()
        cards = torch.randn(1, 2, 4, 32)
        borders = torch.ones(1, 2, 4, dtype=torch.long)
        supports = torch.zeros(1, 2, dtype=torch.long)
        keys = data.identity_keys[torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]]) - 1]
        base = model(cards, borders, supports, supports)
        # Zero-initialised: before training, identities change nothing.
        self.assertTrue(torch.allclose(base, model(cards, borders, supports, supports, identity_keys=keys)))
        with torch.no_grad():
            model.identity_embedding.weight[keys[0, 1, 0]] += torch.randn(32)  # A constant shift would vanish in LayerNorm.
        changed = model(cards, borders, supports, supports, identity_keys=keys)
        self.assertFalse(torch.allclose(base, changed))
        hidden = torch.ones(1, 2, 4, dtype=torch.bool)
        hidden[0, 1, 0] = False  # A hidden enemy card reveals no identity.
        self.assertGreater(float(model.identity_penalty()), 0)  # Memorisation costs loss.
        self.assertTrue(torch.allclose(model(cards, borders, supports, supports, card_visible=hidden),
                                       model(cards, borders, supports, supports, card_visible=hidden, identity_keys=keys)))


if __name__ == "__main__":
    unittest.main()
