"""Label generation and the training loop (tiny runs)."""

import random
import tempfile
import math
import unittest
from pathlib import Path

import numpy as np
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

    def test_unfreezing_language_starts_its_optimizer_fresh(self):
        root = Path(self.temp.name) / "unfreeze"
        directory = root / "store"
        directory.mkdir(parents=True)
        for seed in (1, 25):
            labels._worker((seed, 6, str(directory), snapshot(), False, Path(self.temp.name) / "tb3"))
        run = Path(self.temp.name) / "unfreeze_run"
        train(steps=2, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=run,
              label_root=root, freeze_language_at=0)
        state = torch.load(run / "trainer.pt", weights_only=True)
        from card_engine.model.checkpoint import load_checkpoint
        model = load_checkpoint(run / "model.checkpoint")[0]
        single = torch.optim.AdamW(model.parameters())  # an optimizer saved before the language group existed
        for parameter in model.parameters():
            single.state[parameter] = {"step": torch.tensor(2.0), "exp_avg": torch.ones_like(parameter),
                                       "exp_avg_sq": torch.ones_like(parameter)}
        torch.save({**state, "optimizer": single.state_dict()}, run / "trainer.pt")
        before = {k: v.clone() for k, v in model.description.state_dict().items()}
        train(steps=4, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=run,
              label_root=root, freeze_language_at=100, language_lr=1e-3)
        after = load_checkpoint(run / "model.checkpoint")[0].description.state_dict()
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))  # unfrozen: it trains again
        saved = torch.load(run / "trainer.pt", weights_only=True)["optimizer"]
        self.assertEqual([g["base_lr"] for g in saved["param_groups"]], [3e-4, 1e-3])

    def test_hard_examples_keep_distinct_teams(self):
        from card_engine.training.hard import distinct
        def team(cards, red=1, blue=1):
            return {"cards": cards, "red": red, "blue": blue}
        teams = [team([1, 2, 3, 4]), team([1, 2, 3, 5]), team([2, 1, 3, 4]), team([1, 2, 3, 4], red=2, blue=2),
                 team([6, 7, 8, 9])]
        # a one-card variant is skipped; a swap of two slots, two support changes or a new lineup are kept
        self.assertEqual(distinct(teams, [0, 1, 2, 3, 4], keep=4), [0, 2, 3, 4])
        self.assertEqual(distinct(teams, [0, 1, 2, 3, 4], keep=2), [0, 2])

    def test_the_miner_can_hold_one_tower_floor(self):
        import random
        from card_engine import tower
        from card_engine.catalog import load_catalog
        from card_engine.training.hard import draw_enemy
        catalog, rng = load_catalog(), random.Random(1)
        fixed = tower.fixed_team(catalog, 105)["cards"]
        for _ in range(3):  # every enemy is floor 105 Impossible: its fixed team at its stats
            team, stats = draw_enemy(rng, catalog, (105, "impossible"))
            self.assertEqual((team["cards"], stats), (fixed, tower.stats(105, "Impossible")))
        team, stats = draw_enemy(rng, catalog, (104, "Hell"))  # no fixed team there: random enemies at its stats
        self.assertEqual((len(team["cards"]), stats), (4, tower.stats(104, "Hell")))

    def test_mined_teams_are_also_labelled_defending(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from card_engine.catalog import load_catalog
        from card_engine.training import hard
        with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn"), initializer=hard._init) as pool:
            arrays, probs, exact, extra, *_ = hard.hard_shard(None, load_catalog(), 3, 1, pool, candidates=80, keep=2,
                                                              keep_random=2)
        kept = len(probs) // 2  # each kept team again with the fixed-stat enemy attacking first, the engine's own label
        self.assertEqual(extra["fixed_side"].tolist(), [1] * kept + [0] * kept)
        np.testing.assert_array_equal(arrays["cards"][kept:, 0], arrays["cards"][:kept, 1])
        np.testing.assert_array_equal(arrays["cards"][kept:, 1], arrays["cards"][:kept, 0])
        np.testing.assert_array_equal(extra["fixed_stats"][kept:], extra["fixed_stats"][:kept])

    def test_hard_examples_take_a_share_of_each_batch(self):
        import numpy as np
        root = Path(self.temp.name) / "hardmix"
        directory = root / "store"
        directory.mkdir(parents=True)
        labels._worker((1, 6, str(directory), snapshot(), False, Path(self.temp.name) / "tb4"))
        labels._worker((2, 6, str(directory), snapshot(), True))
        with np.load(directory / "fixed_00000002.npz") as shard:  # stands in for a hard shard
            np.savez(directory / "hard_00000003.npz", **{k: shard[k] for k in shard.files})
        train_rows, _ = load_split(directory, "cpu")
        self.assertEqual(int(train_rows["hard"].sum()), 6)
        self.assertEqual(len(train_rows["hard"]), int(train_rows["target"].shape[0]))  # no copies
        run = Path(self.temp.name) / "hardmix_run"
        train(steps=2, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=run,
              label_root=root, mix=(0.5, 0.0, 0.0))

    def test_fresh_deeper_run_borrows_the_language_side_and_starts_on_its_curriculum(self):
        from card_engine.model.checkpoint import load_checkpoint, save_checkpoint
        from card_engine.training.train import mix_rows, mix_shares
        root = Path(self.temp.name) / "fresh"
        directory = root / "store"
        directory.mkdir(parents=True)
        labels._worker((1, 6, str(directory), snapshot(), False, Path(self.temp.name) / "tb5"))
        labels._worker((2, 6, str(directory), snapshot(), True))
        train_rows, _ = load_split(directory, "cpu")
        mixed = mix_rows(train_rows)
        self.assertEqual(set(torch.nonzero(train_rows["fixed_side"] >= 0)[:, 0].tolist()), set(mixed["fixed"].tolist()))
        self.assertEqual([round(x, 6) for x in mix_shares(5, (0.1, 0.2, 0.3), (0.3, 0.4, 0.5), 10)], [0.2, 0.3, 0.4, 0.0, 0.0])
        self.assertEqual(mix_shares(12, (0.1, 0.2, 0.3), (0.3, 0.4, 0.5), 10), (0.1, 0.2, 0.3, 0.0, 0.0))
        source = BattleModel()
        save_checkpoint(source, Path(self.temp.name) / "source.checkpoint", metadata={"step": 7})
        run = Path(self.temp.name) / "fresh_run"
        train(steps=2, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=run,
              label_root=root, layers=2, language_from=Path(self.temp.name) / "source.checkpoint",
              mix=(0.1, 0.2, 0.3), mix_start=(0.2, 0.4, 0.4), mix_until=1)
        model, metadata = load_checkpoint(run / "model.checkpoint")
        self.assertEqual((len(model.strategy.blocks), metadata["step"]), (2, 2))
        ema, ema_meta = load_checkpoint(run / "ema.checkpoint")  # the weight average, beside the live weights
        self.assertEqual((len(ema.strategy.blocks), ema_meta["step"]), (2, 2))
        live, average = dict(model.named_parameters()), dict(ema.named_parameters())
        self.assertTrue(any(not torch.equal(live[n], average[n]) for n in live))  # it trails the live weights
        for name, value in source.description.state_dict().items():  # copied, then frozen
            self.assertTrue(torch.equal(value, model.description.state_dict()[name]), name)

    def test_reload_appends_new_shards_in_place(self):
        import numpy as np
        from card_engine.training import train as trainer
        directory = Path(self.temp.name) / "append" / "store"
        directory.mkdir(parents=True)
        for seed in (1, 2, 3, 4, 25):  # 24 training rows: a quarter spare holds one more shard
            labels._worker((seed, 6, str(directory), snapshot(), True))
        first, _ = load_split(directory, "cpu")
        storage = first["target"].untyped_storage().data_ptr()
        labels._worker((5, 6, str(directory), snapshot(), True))
        second, val = load_split(directory, "cpu")
        self.assertEqual(second["target"].untyped_storage().data_ptr(), storage)  # appended, not rebuilt
        self.assertEqual(len(second["target"]), len(first["target"]) + 6)
        trainer._TENSORS.clear()
        trainer._SHARD_CACHE.clear()
        rebuilt, rebuilt_val = load_split(directory, "cpu")  # the same rows as a full load, in another order
        for key in rebuilt:
            self.assertTrue(torch.equal(second[key].float().sum(0), rebuilt[key].float().sum(0)), key)
        self.assertTrue(torch.equal(val["target"], rebuilt_val["target"]))

    def test_impossible_duplicates_are_dropped(self):
        cards = np.ones((3, 2, 4), dtype=np.int16)
        cards[1, 1, :2] = 250  # two Time Lord Stryx on one side
        cards[2, 0, 0] = cards[2, 1, 0] = 240  # one Fate Seamstress on each side: fine
        self.assertEqual(labels.possible_rows(cards).tolist(), [True, False, True])

    def test_fixed_stats_match_the_engine_at_battle_start(self):
        from card_engine import tower
        catalog, inputs, rng = load_catalog(), Inputs("cpu"), random.Random(11)
        passives = {card.id for card in catalog.cards if card.name in DECK_PASSIVES}
        specs, battles = [], []
        while len(specs) < 40:
            spec = labels.random_spec(rng, catalog)
            if passives.isdisjoint(spec["cards"][0] + spec["cards"][1]):
                battles.append(labels.fixed_battle(rng, catalog, spec))
                specs.append(spec)
        self.assertEqual({side for side, _ in battles}, {0, 1})
        borders = {tower.enemy_border(catalog, level) for level in tower.DIFFICULTIES}  # none, or a difficulty's
        self.assertTrue(all(len(set(spec["borders"][side])) == 1 and spec["borders"][side][0] in borders
                            for spec, (side, _) in zip(specs, battles)))
        shu = labels.random_spec(rng, catalog)  # floor 95 Impossible: Shu and Sekhmet carry 1.7x HP
        shu["cards"][1], shu["arts"][1] = tower.fixed_team(catalog, 95)["cards"], [0] * 4
        shu["borders"][1], shu["mutations"][1] = [1] * 4, [0] * 4
        specs.append(shu)
        battles.append((1, tower.stats(95, "Impossible")))
        rows = {name: torch.tensor([spec[name] for spec in specs]) for name in labels.FIELDS}
        rows["fixed_side"] = torch.tensor([side for side, _ in battles], dtype=torch.int8)
        rows["fixed_stats"] = torch.tensor([b[:2] for _, b in battles], dtype=torch.float32)
        rows["fixed_hp_mult"] = torch.tensor([b[2] for _, b in battles], dtype=torch.int8)
        stats = inputs(rows, torch.zeros(289, 768))["card_stats"].double()
        expected = torch.tensor([drago.initial_stats(catalog, spec, fixed=(side, tower.engine_stats(catalog, spec["cards"][side], b)))
                                 for spec, (side, b) in zip(specs, battles)], dtype=torch.float64)
        torch.testing.assert_close(stats, expected, rtol=1e-6, atol=1.0)  # the engine rounds HP up
        self.assertAlmostEqual(float(stats[-1, 1, 0, 0] / stats[-1, 1, 2, 0]), 1.7, places=5)

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

    def test_stats_beside_the_card_stay_bounded_and_the_run_keeps_its_layout(self):
        from card_engine.model.checkpoint import load_checkpoint
        from card_engine.model.config import StrategicConfig
        strategy = BattleModel(strategic_config=StrategicConfig(layers=1, stat_width=64, pack_embedding=False,
                                                                mutation_embedding=False)).strategy.eval()
        cards = torch.randn(2, 2, 4, 768)
        small, huge = torch.tensor([[0.1, 0.1]]), torch.tensor([[9.0, 9.0]])  # log-stat gaps of e^0.1 and e^9 (8,100x)
        with torch.no_grad():
            parts = [strategy.with_stats(cards, gap.expand(2, 2, 4, 2))[0] for gap in (small, huge)]
        for part in parts:  # each part normalised on its own: the stats cannot outgrow the card
            self.assertEqual(part.shape[-1], 768)
            self.assertLess(float(part[..., 704:].norm(dim=-1).max()), 9.0)
        torch.testing.assert_close(parts[0][..., :704], parts[1][..., :704])  # the card's channels ignore the stats
        root = Path(self.temp.name) / "layout"
        directory = root / "store"
        directory.mkdir(parents=True)
        labels._worker((1, 6, str(directory), snapshot(), False, Path(self.temp.name) / "tb7"))
        labels._worker((2, 6, str(directory), snapshot(), True))
        run = Path(self.temp.name) / "layout_run"
        train(steps=2, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=run,
              label_root=root, layers=1, architecture={"stat_width": 64, "pack_embedding": False, "mutation_embedding": False})
        model, _ = load_checkpoint(run / "model.checkpoint")
        config = model.strategy.config
        self.assertEqual((config.stat_width, config.pack_embedding, config.mutation_embedding), (64, False, False))
        self.assertIsNone(model.strategy.pack_embedding)

    def test_stat_tokens_sit_beside_the_cards_and_hidden_cards_hide_their_stats(self):
        from card_engine.model.config import StrategicConfig
        torch.manual_seed(2)
        strategy = BattleModel(strategic_config=StrategicConfig(layers=1, stat_tokens=True)).strategy.eval()
        common = dict(card_embeddings=torch.randn(2, 2, 4, 768), border_ids=torch.ones(2, 2, 4, dtype=torch.long),
                      red_support_ids=torch.zeros(2, 2, dtype=torch.long), blue_support_ids=torch.zeros(2, 2, dtype=torch.long))
        stats = torch.rand(2, 2, 4, 2) * 1e4 + 1
        with torch.no_grad():
            sequence = strategy.build_sequence(**common, card_stats=stats)
            self.assertEqual(sequence.shape[1], 23)  # 15 slots and 8 stat tokens, MODE and PREDICT last
            torch.testing.assert_close(sequence[:, 3:7], strategy.build_sequence(**common, card_stats=stats * 3)[:, 3:7])
            self.assertFalse(torch.allclose(strategy(**common, card_stats=stats),
                                            strategy(**common, card_stats=stats.flip(2))))  # whose stats is which matters
            hidden = torch.ones(2, 2, 4, dtype=torch.bool)
            hidden[:, 1, 2] = False
            changed = stats.clone()
            changed[:, 1, 2] *= 50
            torch.testing.assert_close(strategy.build_sequence(**common, card_stats=stats, card_visible=hidden)[:, 15 + 6],
                                       strategy.build_sequence(**common, card_stats=changed, card_visible=hidden)[:, 15 + 6])
        with self.assertRaises(ValueError):
            StrategicConfig(stat_tokens=True, stat_width=64)

    def test_stat_pairs_compare_each_card_with_the_other_seven_from_its_own_side(self):
        from card_engine.model.config import StrategicConfig
        strategy = BattleModel(strategic_config=StrategicConfig(layers=1, stat_tokens=True, stat_pairs=True)).strategy
        torch.manual_seed(3)
        stats = torch.rand(2, 2, 4, 2) * 1e4 + 1
        visible = torch.ones(2, 2, 4, dtype=torch.bool)
        inputs = strategy.stat_inputs(stats, visible)
        self.assertEqual(inputs.shape, (2, 2, 4, 37))
        torch.testing.assert_close(inputs, strategy.stat_inputs(stats * 1e5, visible))  # ratios only
        swapped = strategy.stat_inputs(stats.flip(1), visible)  # the sides trade places: each card sees the same
        torch.testing.assert_close(swapped, inputs.flip(1))
        hits = inputs[0, 0, 0, 2 + 3 * 5 + 2]  # ally card 1 against the enemy's first card: hits to kill it
        self.assertAlmostEqual(float(hits), float(stats[0, 1, 0, 0].log() - stats[0, 0, 0, 1].log()), places=4)
        visible[:, 1, 3] = False
        hidden = strategy.stat_inputs(stats, visible)
        self.assertTrue(bool((hidden[:, 0, 0, 2 + 6 * 5:] == 0).all()))  # nothing about an unseen card

    def test_incomplete_mode_rows_hide_their_side_and_only_the_newest_generations_count(self):
        from card_engine.training.labels import GENERATION_SEEDS, HIDDEN_TEAM
        from card_engine.training.train import Inputs, newest_generations
        names = [Path(f"hidden_{g * GENERATION_SEEDS + n:08d}.npz") for g in (0, 1, 2) for n in (1, 2)]
        kept = newest_generations([Path("shard_00000001.npz"), Path("pvp_00000001.npz"), *names], 2)
        self.assertEqual({p.name for p in kept}, {"shard_00000001.npz", "pvp_00000001.npz", *(p.name for p in names[2:])})
        team = {key: value[0] for key, value in labels.random_spec(random.Random(4), load_catalog()).items()}
        rows = {key: torch.tensor([[team[key], HIDDEN_TEAM[key]], [HIDDEN_TEAM[key], team[key]]], dtype=torch.int16)
                for key in labels.FIELDS}
        rows["hidden_side"] = torch.tensor([1, 0], dtype=torch.int8)
        metadata = Inputs("cpu")(rows, torch.zeros(289, 768))
        self.assertEqual(metadata["card_visible"][0].tolist(), [[True] * 4, [False] * 4])
        self.assertEqual(metadata["card_visible"][1].tolist(), [[False] * 4, [True] * 4])
        self.assertEqual(metadata["support_visible"][1, 0].tolist(), [False, False])
        self.assertEqual(metadata["mode_ids"].tolist(), [1, 1])

    def test_pvp_players_own_more_with_rolls_and_luck_and_attackers_stay_near_the_defender(self):
        import numpy as np
        from card_engine.training import incomplete
        progression = incomplete.Progression(load_catalog())
        rng = np.random.default_rng(1)
        sizes = [len(progression.pool(rng, rolls, luck).entries) for rolls, luck in ((1e5, 1), (1e7, 1), (1e7, 100), (1e9, 100))]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])
        pool = progression.pool(rng, 1e9, 100)
        self.assertTrue((pool.copies >= 1).all() and (pool.copies <= incomplete.MAX_COPIES).all())
        self.assertEqual(len(pool.reds), 5 * len(load_catalog().red_supports))  # every support at every tier
        self.assertAlmostEqual(incomplete.mutation_chance(1), 1 / 2000)
        self.assertAlmostEqual(incomplete.mutation_chance(100), 1 / 150)
        draws = np.array([incomplete.draw_attacker(rng, 1e7, 3) for _ in range(2000)])
        self.assertTrue(((draws[:, 0] >= 1e5) & (draws[:, 0] <= 1e9) & (draws[:, 1] >= 1) & (draws[:, 1] <= 100)).all())
        self.assertLess(np.log10(draws[:, 0]).std(), np.log10(draws[:, 1]).std())  # luck varies more than rolls

    def test_found_shards_are_their_own_kind(self):
        import numpy as np
        root = Path(self.temp.name) / "found"
        directory = root / "store"
        directory.mkdir(parents=True)
        labels._worker((1, 6, str(directory), snapshot(), False, Path(self.temp.name) / "tb8"))
        labels._worker((2, 6, str(directory), snapshot(), True))
        with np.load(directory / "fixed_00000002.npz") as shard:  # stands in for a found shard
            np.savez(directory / "found_00000003.npz", **{k: shard[k] for k in shard.files})
        from card_engine.training.train import mix_rows
        train_rows, _ = load_split(directory, "cpu")
        mixed = mix_rows(train_rows)
        self.assertEqual((len(mixed["found"]), len(mixed["hard"])), (6, 0))
        self.assertFalse(set(mixed["found"].tolist()) & set(mixed["fixed"].tolist()))
        train(steps=2, batch_size=8, warmup=1, eval_every=100, checkpoint_every=2, device="cpu",
              run_dir=Path(self.temp.name) / "found_run", label_root=root, mix=(0.0, 0.0, 0.0, 0.0, 0.5))

    def test_the_stat_rule_is_a_log_odds_prior_with_a_residual(self):
        from card_engine.model.config import StrategicConfig
        from card_engine.model.networks import STAT_PRIOR_LOGIT
        torch.manual_seed(4)
        strategy = BattleModel(strategic_config=StrategicConfig(layers=1, stat_prior=True)).strategy.eval()
        common = dict(card_embeddings=torch.randn(3, 2, 4, 768), border_ids=torch.ones(3, 2, 4, dtype=torch.long),
                      red_support_ids=torch.zeros(3, 2, dtype=torch.long), blue_support_ids=torch.zeros(3, 2, dtype=torch.long))
        stats = torch.ones(3, 2, 4, 2)
        stats[0, 0] *= 100  # A far stronger
        stats[1, 1] *= 100  # B far stronger
        stats[2, 1] *= 100
        with torch.no_grad():
            probs = strategy(**common, card_stats=stats).softmax(-1)[:, 0]
        favourite = torch.sigmoid(torch.tensor(STAT_PRIOR_LOGIT))
        torch.testing.assert_close(probs[0], favourite)  # the residual starts at 0: the rule's 91.4%
        torch.testing.assert_close(probs[1], 1 - favourite)
        with torch.no_grad():
            strategy.outcome_head[-1].bias.copy_(torch.tensor([1.0, -1.0]))  # a residual of +2 log-odds for A
            shifted = strategy(**common, card_stats=stats).softmax(-1)[1, 0]
            residual = strategy.prior_residual(strategy.build_sequence(**common, card_stats=stats))
        torch.testing.assert_close(shifted, torch.sigmoid(torch.tensor(-STAT_PRIOR_LOGIT + 2.0)))
        torch.testing.assert_close(residual, torch.full((3,), 2.0))
        hidden = torch.ones(3, 2, 4, dtype=torch.bool)
        hidden[2, 1] = False  # a side unseen: no rule, the residual alone
        with torch.no_grad():
            seen = strategy(**common, card_stats=stats, card_visible=hidden).softmax(-1)[2, 0]
        torch.testing.assert_close(seen, torch.sigmoid(torch.tensor(2.0)))
        smooth = BattleModel(strategic_config=StrategicConfig(layers=1, stat_prior=True, stat_prior_scale=3.0,
                                                              stat_prior_slope=1.0)).strategy.eval()
        stats[2, 1] /= 100  # equal stats
        with torch.no_grad():
            probs = smooth(**common, card_stats=stats).softmax(-1)[:, 0]
        expected = torch.sigmoid(3 * torch.tanh(torch.tensor(math.log(100) / 3)))  # A has 100x the strength
        torch.testing.assert_close(probs, torch.stack([expected, 1 - expected, torch.tensor(0.5)]))
        grown = stats.clone().requires_grad_(True)  # the smooth prior passes a gradient to the stats
        smooth(**common, card_stats=grown)[:, 0].sum().backward()
        self.assertGreater(float(grown.grad[2, 0].sum()), 0.0)

    def test_the_smooth_prior_fit_recovers_its_numbers(self):
        from card_engine.training.train import fit_prior
        generator = torch.Generator().manual_seed(0)
        delta = torch.randn(200_000, generator=generator, dtype=torch.float64) * 3
        target = torch.sigmoid(0.3 + 4.0 * torch.tanh(1.5 * delta / 4.0))  # scale 4, slope 1.5, A's edge 0.3
        scale, slope, bias, report = fit_prior(delta, target)
        self.assertAlmostEqual(bias, 0.3, places=2)
        self.assertAlmostEqual(scale, 4.0, places=2)
        self.assertAlmostEqual(slope, 1.5, places=2)
        self.assertLess(report["loss"], report["step_loss"])

    def test_training_runs_and_resumes(self):
        root = Path(self.temp.name) / "labels"
        directory = root / "store"
        directory.mkdir(parents=True)
        for seed in (1, 25):  # 25 is a validation shard
            labels._worker((seed, 6, str(directory), snapshot(), False, Path(self.temp.name) / "tb2"))
        labels._worker((25, 6, str(directory), snapshot(), True))  # tower battles, read alongside
        released = []
        train_rows, val_rows = load_split(directory, "cpu", release=lambda: released.append(True))
        self.assertEqual(released, [True])  # a reload drops the old tensors before building new ones
        self.assertTrue((val_rows["fixed_side"] >= 0).any())
        self.assertEqual(set(train_rows["fixed_side"].tolist()), {-1})
        import json as _json
        self.assertEqual((train_rows["target"].shape[1], val_rows["cards"].shape[1:]), (2, (2, 4)))  # A win, B win (no ties)
        model, inputs = BattleModel().eval(), Inputs("cpu")
        with torch.no_grad():
            table = card_table(model, inputs.data.description_tokens)
        metrics = evaluate(model, inputs, table, val_rows)  # one shared card table, as in training
        self.assertTrue(0 <= metrics["accuracy"] <= 1 and metrics["kl"] >= 0)
        upsets = metrics["upset_rows"] / val_rows["target"].shape[0]  # the KL splits into upsets and the rest
        self.assertAlmostEqual(metrics["kl"], upsets * metrics["upset_kl"] + (1 - upsets) * metrics["favourite_kl"], places=5)
        run = Path(self.temp.name) / "run"
        smooth_run = Path(self.temp.name) / "smooth"  # a new run fits its smooth prior, kept in its checkpoints
        train(steps=2, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=smooth_run,
              label_root=root, smooth_prior=True, layers=1)
        from card_engine.model.checkpoint import load_checkpoint
        config = load_checkpoint(smooth_run / "model.checkpoint")[0].strategic_config
        self.assertTrue(config.stat_prior and config.stat_prior_scale > 0 and config.stat_prior_slope > 0)
        self.assertEqual(load_checkpoint(smooth_run / "ema.checkpoint")[0].strategic_config, config)
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
        # a new run's language side waits for the stat MLP's plateau, its card vectors 0 until then; then it thaws and
        # freezes by the stat MLP (a plateau_spread of 1e9 reads any two readings as a plateau, -1 never; a
        # learning_rise of 0 reads any reading as learning, 1e9 never)
        small = {"architecture": {"width": 64, "attention_width": 64, "feedforward_width": 128, "stat_hidden_width": 128},
                 "language": {"width": 64, "layers": 1, "heads": 1, "feedforward_width": 128, "output_width": 64,
                              "projection_hidden_width": 64}, "layers": 1}
        plateau = Path(self.temp.name) / "plateau"
        spells = dict(eval_every=100, checkpoint_every=100, batch_size=4, warmup=1, device="cpu", run_dir=plateau,
                      label_root=root, watch=None, plateau_evals=2, frozen_min=100, thaw_min=100)
        vectors = lambda: float(card_table(load_checkpoint(plateau / "model.checkpoint")[0],
                                           inputs.data.description_tokens).detach().abs().max())
        state = lambda: torch.load(plateau / "trainer.pt", weights_only=True)["language_spells"]
        phase = lambda: {k: v for k, v in state().items() if k != "baseline"}
        train(steps=100, language_after_plateau=True, plateau_spread=-1, **spells, **small)
        self.assertEqual((state()["frozen"], vectors()), (True, 0.0))  # waiting, and no plateau yet
        first = _json.loads((plateau / "log.jsonl").read_text().splitlines()[0])
        self.assertEqual(first["grok"]["language_gain"]["kl"], 0.0)  # no card text yet: zeroing it changes nothing
        train(steps=300, plateau_spread=1e9, learning_rise=1e9, language_budget=1000, **spells)  # 2 readings: thaws at 300
        self.assertEqual(phase(), {"frozen": False, "since": 300, "thawed": 0, "done": False})
        records = [_json.loads(line) for line in (plateau / "log.jsonl").read_text().splitlines()]
        self.assertAlmostEqual(state()["baseline"], (records[-2]["stat_drift"] + records[-1]["stat_drift"]) / 2, places=2)
        train(steps=500, plateau_spread=-1, learning_rise=0, language_budget=1000, **spells)  # learning: refreezes at 400
        self.assertEqual(phase(), {"frozen": True, "since": 400, "thawed": 100, "done": False})
        self.assertGreater(vectors(), 0.0)  # it learned from zero
        train(steps=800, plateau_spread=1e9, learning_rise=1e9, language_budget=150, **spells)  # thaws at 700, done at 750
        self.assertEqual(phase(), {"frozen": True, "since": 750, "thawed": 150, "done": True})
        records = [_json.loads(line) for line in (plateau / "log.jsonl").read_text().splitlines()]
        events = [(r["step"], k) for r in records for k in ("language_thawed", "language_refrozen") if r.get(k)]
        self.assertEqual(events, [(300, "language_thawed"), (400, "language_refrozen"), (700, "language_thawed")])
        self.assertTrue(records[-1]["language_done"])
        done = {k: v.clone() for k, v in load_checkpoint(plateau / "model.checkpoint")[0].description.state_dict().items()}
        train(steps=900, plateau_spread=-1, learning_rise=1e9, language_budget=300, **spells)  # a larger budget reopens it
        self.assertEqual(phase(), {"frozen": False, "since": 800, "thawed": 150, "done": False})
        reopened = load_checkpoint(plateau / "model.checkpoint")[0].description.state_dict()
        self.assertFalse(all(torch.equal(done[k], reopened[k]) for k in done))  # learning again
        # a new machine: continue from the downloaded model with a fresh optimizer, keeping the step
        moved = Path(self.temp.name) / "moved"
        train(steps=8, batch_size=4, warmup=1, eval_every=100, checkpoint_every=2, device="cpu", run_dir=moved, label_root=root,
              init_from=run / "model.checkpoint")
        state = torch.load(moved / "trainer.pt", weights_only=True)
        self.assertEqual((state["step"], state["warm_from"]), (8, 6))


if __name__ == "__main__":
    unittest.main()
