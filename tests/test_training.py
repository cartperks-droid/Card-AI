"""Label generation and the training loop (tiny runs)."""

import random
import tempfile
import unittest
from pathlib import Path

import torch

from card_engine.catalog import load_catalog
from card_engine.model import BattleModel, load_model_data
from card_engine.simulator import drago, kernel
from card_engine.training import labels
from card_engine.training.flags import snapshot
from card_engine.training.train import Inputs, card_table, evaluate, load_split, train


DECK_PASSIVES = ("General Moon Zoo", "Julius Leader")


class TrainingTests(unittest.TestCase):
    def test_dropout_only_at_the_gelu_upscale(self):
        from card_engine.training.train import set_dropout
        from card_engine.model.networks import TransformerBlock
        model = BattleModel()
        set_dropout(model, 0.1)
        for block in model.modules():
            if isinstance(block, TransformerBlock):
                self.assertEqual(block.feedforward[2].p, 0.1)
                self.assertEqual((block.residual_dropout.p, block.attention.dropout), (0.0, 0.0))

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.temp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_random_specs_compile_and_label(self):
        rng = random.Random(4)
        specs = [labels.random_spec(rng, self.catalog) for _ in range(20)]
        for spec in specs:
            self.assertEqual([len(team) for team in spec["cards"]], [4, 4])
            for side in (0, 1):
                for card, mutation, art in zip(spec["cards"][side], spec["mutations"][side], spec["arts"][side]):
                    self.assertTrue(mutation == 0 or self.catalog.card(card).weather_id == 1)  # weather cards never mutate
                    self.assertEqual(art > 0, card == 56)  # each Astraeus art is its own card
                self.assertIn(spec["red_tier"][side], (0, 1, 2, 3, 4, 5))
        probs, exact = labels.label_specs(self.catalog, specs[:6], seed=1, rolloutError=0.2, nodeBudget=500)
        self.assertTrue(((probs.sum(1) - 1) ** 2 < 1e-9).all())
        self.assertTrue(exact.any())  # many battles are deterministic

    def test_an_engine_failure_leaves_the_row_unfinished(self):
        from unittest import mock
        spec = labels.random_spec(random.Random(1), self.catalog)
        log = Path(self.temp.name) / "errors.jsonl"
        with mock.patch.object(kernel, "evaluate", side_effect=RuntimeError("Maximum call stack size exceeded")), \
                mock.patch.object(labels, "ENGINE_ERRORS", log):
            self.assertEqual(labels.evaluate(self.catalog, spec, 5), ((0.0, 0.0, 0.0, 1.0), False))
        self.assertIn("Maximum call stack", log.read_text())
        with mock.patch.object(kernel, "evaluate", side_effect=BrokenPipeError(32, "Broken pipe")), \
                mock.patch.object(labels, "ENGINE_ERRORS", log), mock.patch.object(drago, "_WORKER", object()):
            self.assertEqual(labels.evaluate(self.catalog, spec, 5), ((0.0, 0.0, 0.0, 1.0), False))
            self.assertIsNone(drago._WORKER)  # the next battle starts a fresh engine process

    def test_saved_turns_give_the_same_answers_as_replaying(self):
        rng = random.Random(21)
        for index in range(12):
            spec = labels.random_spec(rng, self.catalog)
            resumed = drago.evaluate(self.catalog, spec, index, nodeBudget=300)
            replayed = drago.evaluate(self.catalog, spec, index, nodeBudget=300, snapshots=False)
            self.assertEqual(resumed, replayed)

    def test_astraeus_plays_the_art_it_is_given(self):
        spec = {"cards": [[56, 56, 1, 2], [3, 4, 5, 6]], "borders": [[1] * 4] * 2, "mutations": [[0] * 4] * 2,
                "arts": [[3, 4, 0, 0], [0] * 4], "red": [0, 0], "red_tier": [0, 0], "blue": [0, 0], "blue_tier": [0, 0]}
        abilities = [card[2] for card in drago.initial_cards(self.catalog, spec)[0][:2]]
        self.assertEqual(abilities, ["ConstellarVirgo", "ConstellarGemini"])
        with self.assertRaises(ValueError):
            drago.loadouts(self.catalog, {**spec, "arts": [[0] * 4, [0] * 4]})

    def test_engine_supports_every_card_and_aura(self):
        cards, blue = drago.supported(self.catalog)
        self.assertEqual(len(cards), len(self.catalog.cards))
        self.assertEqual(len(blue), len(self.catalog.blue_supports))

    def test_tablebase_stores_exact_outcomes(self):
        from card_engine.training.tablebase import Tablebase
        rng = random.Random(8)
        specs = [labels.random_spec(rng, self.catalog) for _ in range(6)]
        base = Tablebase("test", root=Path(self.temp.name) / "tb")
        probs, exact = labels.label_specs(self.catalog, specs, seed=1, tablebase=base, rolloutError=0.2, nodeBudget=500)
        self.assertEqual(len(base), int(exact.sum()))
        for spec, p, e in zip(specs, probs, exact):
            if e:
                self.assertEqual(tuple(round(x, 6) for x in base.get(spec)), tuple(round(float(x), 6) for x in p))
        again, again_exact = labels.label_specs(self.catalog, specs, seed=2, tablebase=base, rolloutError=0.2, nodeBudget=500)
        self.assertTrue((again[exact] == probs[exact]).all() and (again_exact == exact).all())  # exact rows come from the base

    def test_support_tiers_and_astraeus_arts_reach_the_model(self):
        data = load_model_data()
        self.assertEqual(len(set(data.art_identity_keys[1:].tolist())), 7)
        self.assertTrue(set(data.art_identity_keys[1:].tolist()).isdisjoint(data.identity_keys.tolist()))
        model = BattleModel().eval()
        rows = {"cards": torch.tensor([[[56, 3, 4, 5], [6, 7, 8, 9]]]), "borders": torch.ones(1, 2, 4, dtype=torch.long),
                "mutations": torch.zeros(1, 2, 4, dtype=torch.long), "arts": torch.tensor([[[1, 0, 0, 0], [0] * 4]]),
                "red": torch.tensor([[3, 3]]), "red_tier": torch.tensor([[1, 1]]), "blue": torch.tensor([[2, 2]]),
                "blue_tier": torch.tensor([[1, 1]])}
        inputs = Inputs("cpu")
        with torch.no_grad():
            table = card_table(model, inputs.data.description_tokens)
            base = model(**inputs(rows, table))
            rows["red_tier"] = torch.tensor([[5, 1]])
            self.assertFalse(torch.allclose(base, model(**inputs(rows, table))))
        self.assertEqual(int(inputs(rows, table)["identity_keys"][0, 0, 0]), int(data.art_identity_keys[1]))

    def test_card_stats_match_the_engine_at_battle_start(self):
        catalog, inputs, rng = load_catalog(), Inputs("cpu"), random.Random(7)
        passives = {card.id for card in catalog.cards if card.name in DECK_PASSIVES}  # left to the model as abilities
        specs = [spec for spec in (labels.random_spec(rng, catalog) for _ in range(400))
                 if passives.isdisjoint(spec["cards"][0] + spec["cards"][1])]
        rows = {name: torch.tensor([spec[name] for spec in specs]) for name in labels.FIELDS}
        stats = inputs(rows, torch.zeros(289, 768))["card_stats"].double()
        expected = torch.tensor([drago.initial_stats(catalog, spec) for spec in specs], dtype=torch.float64)
        torch.testing.assert_close(stats, expected, rtol=1e-6, atol=0)

    def test_stat_mlp_sees_only_ratios_and_starts_silent(self):
        model = BattleModel().eval()
        torch.manual_seed(0)
        stats = torch.rand(3, 2, 4, 2) * 1e4 + 1
        visible = torch.ones(3, 2, 4, dtype=torch.bool)
        normalized = model.strategy.normalized_stats(stats, visible)
        torch.testing.assert_close(normalized, model.strategy.normalized_stats(stats * 1e6, visible))
        for layer in (model.strategy.stat_projection, model.strategy.stat_mlp[-1]):  # trained: scale still cannot matter
            torch.nn.init.normal_(layer.weight, std=0.02)
        common = dict(card_embeddings=torch.zeros(3, 2, 4, 768), border_ids=torch.ones(3, 2, 4, dtype=torch.long),
                      red_support_ids=torch.zeros(3, 2, dtype=torch.long), blue_support_ids=torch.zeros(3, 2, dtype=torch.long))
        with torch.no_grad():
            torch.testing.assert_close(model.strategy(**common, card_stats=stats), model.strategy(**common, card_stats=stats * 1e6))
            self.assertFalse(torch.allclose(model.strategy(**common, card_stats=stats), model.strategy(**common)))
            for layer in (model.strategy.stat_projection, model.strategy.stat_mlp[-1]):
                torch.nn.init.zeros_(layer.weight)
            torch.testing.assert_close(model.strategy(**common, card_stats=stats), model.strategy(**common))

    def test_training_runs_and_resumes(self):
        root = Path(self.temp.name) / "labels"
        directory = root / "store"
        directory.mkdir(parents=True)
        for seed in (1, 25):  # 25 is a validation shard
            labels._worker((seed, 6, str(directory), snapshot(), Path(self.temp.name) / "tb2"))
        train_rows, val_rows = load_split(directory, "cpu")
        import json as _json
        self.assertEqual((train_rows["target"].shape[1], val_rows["cards"].shape[1:]), (2, (2, 4)))  # A win, B win (no ties)
        model, inputs = BattleModel().eval(), Inputs("cpu")
        with torch.no_grad():
            table = card_table(model, inputs.data.description_tokens)
        metrics = evaluate(model, inputs, table, val_rows)  # one shared card table, as in training
        self.assertTrue(0 <= metrics["accuracy"] <= 1 and metrics["kl"] >= 0)
        run = Path(self.temp.name) / "run"
        train(steps=2, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=run, label_root=root)
        state = torch.load(run / "trainer.pt", weights_only=True)
        self.assertEqual(state["step"], 2)
        train(steps=4, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=run, label_root=root)
        self.assertEqual(torch.load(run / "trainer.pt", weights_only=True)["step"], 4)
        from card_engine.model.checkpoint import load_checkpoint
        before = {k: v.clone() for k, v in load_checkpoint(run / "model.checkpoint")[0].description.state_dict().items()}
        train(steps=6, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=run, label_root=root,
              freeze_language_at=4)
        after = load_checkpoint(run / "model.checkpoint")[0].description.state_dict()
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))  # frozen language transformer
        # a new machine: continue from the downloaded model with a fresh optimizer, keeping the step
        moved = Path(self.temp.name) / "moved"
        train(steps=8, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=moved, label_root=root,
              init_from=run / "model.checkpoint")
        state = torch.load(moved / "trainer.pt", weights_only=True)
        self.assertEqual((state["step"], state["warm_from"]), (8, 6))


if __name__ == "__main__":
    unittest.main()
