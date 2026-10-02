"""User-observed counter scheduling plus stochastic damage and kill healing."""

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import Battle, Fighter, Options, simulate, simulate_batch
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator import native


def duel(a, b, first_side=0):
    return Battle(((a,), (b,)), first_side)


class ReactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / native.DEFAULT_LIBRARY.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        py = simulate_batch(battles, options)
        c = native.simulate_batch(battles, options, library_path=self.library)
        for i, (expected, actual) in enumerate(zip(py, c, strict=True)):
            for key in ("p_a", "p_b", "tie", "unresolved", "expanded_states", "merged_states"):
                self.assertAlmostEqual(getattr(expected, key), actual[key], places=10, msg=f"{i}/{key}")

    def test_raze_mirror_defender_wins_without_counter_chain(self):
        for initiator in (0, 1):
            battle = compile_battle(load_catalog(), ((210,), (210,)), first_side=initiator)
            result = simulate(battle, Options(mode="sample"), trace=True)
            self.assertEqual((result.p_a, result.p_b), (initiator, 1-initiator))
            self.assertEqual([e["side"] for e in result.trace if e["phase"] == "ATTACK_DECLARE"], [initiator, 1-initiator])
            self.assertEqual([e["side"] for e in result.trace if e["phase"] == "COUNTER_DECLARE"], [1-initiator])
            self.compare([battle])

    def test_raze_counter_interrupts_dancer_before_second_hit(self):
        battle = compile_battle(load_catalog(), ((67,), (210,)))
        result = simulate(battle, Options(mode="sample"), trace=True)
        hits = [e for e in result.trace if e["phase"] == "HIT_DECLARE"]
        self.assertEqual([(e["side"], e["hit_index"], e["kind"]) for e in hits], [(0, 1, "normal"), (1, 1, "counter")])
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_surviving_multi_hit_attacker_resumes_after_each_counter(self):
        battle = duel(Fighter(100, 1, attacks_per_action=3), Fighter(100, 1, counter_on_damage=1))
        result = simulate(battle, Options(mode="sample", max_steps=7), trace=True)
        events = [e for e in result.trace if e["phase"] == "HIT_DECLARE"]
        self.assertEqual([e["kind"] for e in events], ["normal", "counter", "normal", "counter", "normal", "counter", "normal"])
        self.assertEqual([e["side"] for e in events], [0, 1, 0, 1, 0, 1, 1])
        self.assertEqual([e["hit_index"] for e in events if e["side"] == 0], [1, 2, 3])
        self.compare([battle], Options(max_steps=7))

    def test_lethal_blocked_dodged_and_zero_damage_never_queue_counter(self):
        targets = (Fighter(10, 1, counter_on_damage=1),
                   Fighter(100, 1, counter_on_damage=1, block_mode=1),
                   Fighter(100, 1, counter_on_damage=1, dodge_probability=1),
                   Fighter(100, 1, counter_on_damage=1, incoming_multiplier=0))
        for target in targets:
            result = simulate(duel(Fighter(100, 20), target), Options(mode="sample", max_steps=1), trace=True)
            self.assertFalse(any(e["phase"] == "COUNTER_QUEUED" for e in result.trace))
            self.compare([duel(Fighter(100, 20), target)], Options(max_steps=1))

    def test_kill_in_counter_cancels_pending_hits_and_enters_next_ally(self):
        battle = Battle(((Fighter(1, 1, attacks_per_action=3), Fighter(20, 20)),
                         (Fighter(10, 1, counter_on_damage=1),)))
        result = simulate(battle, Options(mode="sample", max_steps=3), trace=True)
        transitions = [e for e in result.trace if e["phase"] == "TRANSITION"]
        self.assertEqual([(e["side"], e["next_slot"]) for e in transitions], [(0, 1)])
        self.assertEqual([e["side"] for e in result.trace if e["phase"] == "ATTACK_DECLARE"], [0, 1])
        self.compare([battle])

    def test_critical_probability_changes_exact_win_mass(self):
        battle = duel(Fighter(1, 1, critical_probability=.5), Fighter(2, 1, counter_on_damage=1))
        result = simulate(battle)
        self.assertEqual((result.p_a, result.p_b, result.tie, result.unresolved), (.5, .5, 0, 0))
        self.compare([battle])
        # Dodge and two damage outcomes create three distinct successors.
        battle = duel(Fighter(10, 2, critical_probability=.5), Fighter(10, 2, dodge_probability=.25))
        self.compare([battle], Options(max_steps=10, max_frontier=1, prune_probability=.01))

    def test_kill_healing_restores_modified_max_hp_before_replacement(self):
        battle = Battle(((Fighter(10, 5, entry_hp_multiplier=2, heal_on_kill=1),),
                         (Fighter(10, 3), Fighter(20, 3))))
        result = simulate(battle, Options(mode="sample", max_steps=3), trace=True)
        healing = [e for e in result.trace if e["phase"] == "ON_KILL"]
        self.assertEqual([(e["hp"], e["healed"]) for e in healing], [(20, 3)])
        self.compare([battle])

    def test_arthur_survives_one_poseidon_hit_then_dies(self):
        battle = compile_battle(load_catalog(), ((10,), (44,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode="sample"), trace=True)
        survival = [e for e in result.trace if e["phase"] == "LETHAL_REPLACEMENT"]
        poseidon_hits = [e for e in result.trace if e["phase"] == "DAMAGE_APPLY" and e["side"] == 0]
        self.assertEqual(len(survival), 1)
        self.assertEqual(survival[0]["uses"], 1)
        self.assertEqual(len(poseidon_hits), 2)
        self.assertEqual(poseidon_hits[0]["hp"], 1)
        self.assertLessEqual(poseidon_hits[1]["hp"], 0)
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_lethal_charge_survives_between_hits_but_does_not_reset(self):
        battle = duel(Fighter(10, 10, attacks_per_action=3), Fighter(2, 0, lethal_survivals=1))
        result = simulate(battle, Options(mode="sample"), trace=True)
        hits = [e for e in result.trace if e["phase"] == "HIT_DECLARE"]
        self.assertEqual([e["hit_index"] for e in hits], [1, 2])
        self.assertEqual(result.p_a, 1)
        self.compare([battle])
        # A blocked lethal hit leaves the survival charge for the next hit.
        battle = duel(Fighter(10, 10, attacks_per_action=3), Fighter(2, 0, lethal_survivals=1, block_mode=1))
        result = simulate(battle, Options(mode="sample"), trace=True)
        self.assertEqual([e["hp"] for e in result.trace if e["phase"] == "DAMAGE_APPLY"], [1, -9])
        self.compare([battle])

    def test_randomized_combined_reactions_and_defenses_match_native(self):
        rng = random.Random(73)
        battles = []
        for _ in range(32):
            teams = tuple(tuple(Fighter(hp=rng.randint(8, 40), attack=rng.randint(1, 12),
                        attacks_per_action=rng.choice((1, 3, 5)), counter_on_damage=rng.randrange(2),
                        heal_on_kill=rng.randrange(2), critical_probability=rng.choice((0, .5, 1)),
                        critical_multiplier=rng.choice((1, 1.5, 2)), dodge_probability=rng.choice((0, .1, .5)),
                        damage_cap_max_hp=rng.choice((0, .5)), damage_reduction_max_hp=rng.choice((0, .1)),
                        block_mode=rng.randrange(3), entry_hp_multiplier=rng.choice((1, 1.5)),
                        lethal_survivals=rng.randrange(3),
                    ) for _ in range(rng.randint(1, 3))) for _ in range(2))
            battles.append(Battle(teams, rng.randrange(2)))
        for mode in ("sample", "branch"):
            self.compare(battles, Options(mode=mode, max_steps=70, repeat_cycles=3, max_frontier=30, prune_probability=1e-5))

    def test_invalid_reaction_parameters_rejected_in_both_engines(self):
        for fields in ({"critical_probability": 1.01}, {"critical_multiplier": .9},
                       {"counter_on_damage": True}, {"heal_on_kill": -1}):
            battle = duel(replace(Fighter(10, 1), **fields), Fighter(10, 1))
            with self.assertRaises(ValueError):
                simulate(battle)
            with self.assertRaises(native.NativeSimulationError):
                native.simulate(battle, library_path=self.library)


if __name__ == "__main__":
    unittest.main()
