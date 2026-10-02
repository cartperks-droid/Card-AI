"""Entry-phase ordering, reaction interruption, replacement queues, and mass."""
from pathlib import Path
import random
import tempfile
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import Battle, Fighter, Options, simulate, simulate_batch
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator import native


class EntryAttackTests(unittest.TestCase):
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
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states'):
                self.assertAlmostEqual(getattr(expected, key), actual[key], places=10, msg=f'{i}/{key}')

    def hits(self, battle, steps=1000, **kwargs):
        result = simulate(battle, Options(mode='sample', max_steps=steps, **kwargs), trace=True)
        return result, [(e['side'], e['slot'], e['kind']) for e in result.trace if e['phase'] == 'HIT_DECLARE']

    def test_knightmare_mirror_entries_precede_initiator_normal_attack(self):
        for first in (0, 1):
            battle = compile_battle(load_catalog(), ((14,), (14,)), first_side=first, red_supports=(0, 0))
            result, hits = self.hits(battle, steps=3)
            self.assertEqual(hits, [(first, 0, 'entry'), (1-first, 0, 'entry'), (first, 0, 'normal')])
            damage = [e['damage'] for e in result.trace if e['phase'] == 'DAMAGE_APPLY']
            self.assertEqual(damage, [33, 33, 65])
            self.compare([battle])

    def test_raze_counters_knightmare_entry_before_any_normal_attack(self):
        battle = compile_battle(load_catalog(), ((14,), (210,)), red_supports=(0, 0))
        result, hits = self.hits(battle)
        self.assertEqual(hits, [(0, 0, 'entry'), (1, 0, 'counter')])
        self.assertEqual(result.p_b, 1)
        self.compare([battle])

    def test_good_boy_ko_replacement_entry_then_first_normal_attack(self):
        battle = compile_battle(load_catalog(), ((3, 14), (14,)), red_supports=(0, 0))
        result, hits = self.hits(battle, steps=3)
        self.assertEqual(hits, [(1, 0, 'entry'), (0, 1, 'entry'), (0, 1, 'normal')])
        phases = [(e['phase'], e.get('side')) for e in result.trace]
        self.assertLess(phases.index(('ENTRY', 0)), phases.index(('ENTRY_ATTACK_DECLARE', 1)))
        self.assertLess(phases.index(('TRANSITION', 0)), phases.index(('ENTRY_ATTACK_DECLARE', 0)))
        self.compare([battle])

    def test_entry_counters_resume_queue_without_consuming_normal_turn(self):
        fighter = Fighter(100, 2, entry_hit_multiplier=.5, counter_on_damage=1)
        battle = Battle(((fighter,), (fighter,)))
        _, hits = self.hits(battle, steps=7)
        self.assertEqual([(side, kind) for side, _, kind in hits],
                         [(0, 'entry'), (1, 'counter'), (1, 'entry'), (0, 'counter'),
                          (0, 'normal'), (1, 'counter'), (1, 'normal')])
        self.compare([battle], Options(max_steps=7))

    def test_entry_ko_cancels_dead_cards_hit_and_queues_replacement(self):
        battle = Battle(((Fighter(100, 20, entry_hit_multiplier=1),),
                         (Fighter(1, 1, entry_hit_multiplier=10), Fighter(30, 2, entry_hit_multiplier=1))))
        _, hits = self.hits(battle, steps=3)
        self.assertEqual(hits, [(0, 0, 'entry'), (1, 1, 'entry'), (0, 0, 'normal')])
        self.compare([battle])

    def test_entry_counter_ko_preserves_remaining_initial_entry(self):
        battle = Battle(((Fighter(1, 1, entry_hit_multiplier=1), Fighter(100, 1, entry_hit_multiplier=1)),
                         (Fighter(100, 2, entry_hit_multiplier=1, counter_on_damage=1),)))
        _, hits = self.hits(battle, steps=7)
        self.assertEqual(hits, [(0, 0, 'entry'), (1, 0, 'counter'), (1, 0, 'entry'),
                               (0, 1, 'entry'), (1, 0, 'counter'), (0, 1, 'normal'), (1, 0, 'counter')])
        self.compare([battle])

    def test_entry_damage_uses_defenses_and_does_not_consume_normal_multihits(self):
        a = Fighter(100, 10, entry_hit_multiplier=.5, attacks_per_action=3)
        b = Fighter(100, 1, block_mode=1, incoming_multiplier=.5)
        result, hits = self.hits(Battle(((a,), (b,))), steps=4)
        self.assertEqual([kind for _, _, kind in hits], ['entry', 'normal', 'normal', 'normal'])
        self.assertEqual([e['damage'] for e in result.trace if e['phase'] == 'DAMAGE_APPLY'], [5, 5, 5])
        self.compare([Battle(((a,), (b,)))])

    def test_entry_hits_and_reactions_do_not_advance_repeat_action_count(self):
        fighter = Fighter(100, 1, entry_hit_multiplier=1, counter_on_damage=1)
        battle = Battle(((fighter,), (fighter,)))
        result, hits = self.hits(battle, repeat_cycles=1)
        self.assertEqual(len(hits), 8)  # Two entries + two normal actions, each with a counter.
        self.assertEqual([e['reason'] for e in result.trace if e['phase'] == 'DEATH'], ['repeat_limit']*2)
        self.compare([battle], Options(repeat_cycles=1))

    def test_entry_branches_keep_unfinished_probability_visible(self):
        a = Fighter(10, 8, entry_hit_multiplier=.5, critical_probability=.5)
        b = Fighter(8, 0)
        battle = Battle(((a,), (b,)))
        result = simulate(battle, Options(max_steps=1))
        self.assertEqual((result.p_a, result.p_b, result.unresolved), (.5, 0, .5))
        self.compare([battle], Options(max_steps=1))

    def test_all_entry_mappings_use_full_description_fingerprints(self):
        catalog = load_catalog()
        for identifier, factor in ((14, .5), (35, 2.5), (72, 1)):
            battle = compile_battle(catalog, ((identifier,), (162,)))
            self.assertEqual(battle.teams[0][0].entry_hit_multiplier, factor)
            self.compare([battle])

    def test_randomized_entry_replacement_and_reaction_native_parity(self):
        rng = random.Random(191)
        battles = []
        for _ in range(32):
            teams = tuple(tuple(Fighter(hp=rng.randint(2, 50), attack=rng.randint(1, 12),
                               entry_hit_multiplier=rng.choice((0, .5, 1, 2.5)),
                               attacks_per_action=rng.choice((1, 3)), counter_on_damage=rng.randrange(2),
                               block_mode=rng.randrange(3), lethal_survivals=rng.randrange(2),
                               dodge_probability=rng.choice((0, .2)), critical_probability=rng.choice((0, .5)),
                               entry_hp_multiplier=rng.choice((1, 2)), heal_on_kill=rng.randrange(2))
                          for _ in range(rng.randint(1, 3))) for _ in range(2))
            battles.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            self.compare(battles, Options(mode=mode, max_steps=35, max_frontier=100, repeat_cycles=3,
                                         prune_probability=.0001, seed=111))

    def test_invalid_or_overflowing_entry_multiplier_fails_explicitly(self):
        for factor in (-1, float('inf'), float('nan'), True):
            battle = Battle(((Fighter(10, 1, entry_hit_multiplier=factor),), (Fighter(10, 1),)))
            with self.assertRaises(ValueError):
                simulate(battle)
            with self.assertRaises(ValueError):
                native.simulate(battle, library_path=self.library)
        battle = Battle(((Fighter(10, 1e308, entry_hit_multiplier=2),), (Fighter(10, 1, damage_cap_max_hp=.1),)))
        with self.assertRaisesRegex(ValueError, 'overflow'):
            simulate(battle)
        with self.assertRaises(ValueError):
            native.simulate(battle, library_path=self.library)


if __name__ == '__main__':
    unittest.main()
