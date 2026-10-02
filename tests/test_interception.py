"""Piccolo interception and True Prophet dodges; parity is not game evidence.
User observations are cited per test. Displaced-actor turn handling remains provisional."""

import math
import random
import tempfile
from pathlib import Path
import unittest

from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch
from card_engine.simulator import native
from card_engine.catalog import load_catalog
from card_engine.simulator.catalog_rules import compile_battle


class InterceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'interception.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for py, c in zip(simulate_batch(battles, options),
                         native.simulate_batch(battles, options, library_path=self.library), strict=True):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=10, msg=key)

    def test_piccolo_intercepts_lethal_raze_counter_then_kills_raze(self):
        # User-observed: Piccolo blocks the counter aimed at Forest Spirit, takes
        # Raze's normal attack, then kills Raze without a counter.
        battle = compile_battle(load_catalog(), ((7, 104), (210,)), red_supports=(0, 0))
        result = simulate(battle, Options(mode='sample'), trace=True)
        phases = [e['phase'] for e in result.trace]
        intercept = next(e for e in result.trace if e['phase'] == 'INTERCEPT')
        self.assertEqual((intercept['hp'], intercept['attack']), (12288, 6144))  # User: Piccolo is 3,072 ATK base.
        self.assertLess(phases.index('COUNTER_DECLARE'), phases.index('INTERCEPT'))
        after = [e for e in result.trace[phases.index('INTERCEPT') + 1:] if e['phase'] == 'DAMAGE_APPLY']
        self.assertEqual([(e['side'], e['damage']) for e in after], [(0, 2078), (1, 6144)])
        self.assertNotIn('DEATH', phases[:phases.index('INTERCEPT')])
        self.assertEqual(phases.count('COUNTER_DECLARE'), 1)
        self.assertEqual(result.p_a, 1)
        self.compare([battle])

    def test_interception_needs_a_lethal_hit_and_adjacent_piccolo(self):
        piccolo = Fighter(50, 10, intercept_lethal_multiplier=2)
        nonlethal = simulate(Battle(((Fighter(10, 1), piccolo), (Fighter(100, 5),)), 1),
                             Options(mode='sample', max_steps=1), trace=True)
        self.assertNotIn('INTERCEPT', [e['phase'] for e in nonlethal.trace])
        far = Battle(((Fighter(10, 1), Fighter(10, 1), piccolo), (Fighter(100, 20),)), 1)
        phases = [e['phase'] for e in simulate(far, Options(mode='sample', max_steps=1), trace=True).trace]
        self.assertNotIn('INTERCEPT', phases)
        self.assertIn('DEATH', phases)
        self.compare([far], Options(max_steps=1))

    def test_own_lethal_survival_precedes_interception(self):
        arthur = Fighter(10, 1, lethal_survivals=1)
        battle = Battle(((arthur, Fighter(50, 10, intercept_lethal_multiplier=2)), (Fighter(100, 20),)), 1)
        events = simulate(battle, Options(mode='sample', max_steps=3), trace=True).trace
        phases = [e['phase'] for e in events]
        self.assertLess(phases.index('LETHAL_REPLACEMENT'), phases.index('INTERCEPT'))
        self.compare([battle], Options(max_steps=3))

    def test_displaced_card_returns_after_piccolo_dies(self):
        # Provisional swap: the displaced card waits behind Piccolo.
        battle = Battle(((Fighter(10, 1), Fighter(5, 1, intercept_lethal_multiplier=2)),
                         (Fighter(1000, 20),)), 1)
        events = simulate(battle, Options(mode='sample', max_steps=4), trace=True).trace
        phases = [e['phase'] for e in events]
        self.assertLess(phases.index('INTERCEPT'), phases.index('REENTRY'))
        reentry = next(e for e in events if e['phase'] == 'REENTRY')
        self.assertEqual((reentry['hp'], reentry['attack']), (10, 1))
        self.compare([battle], Options(max_steps=4))

    def test_arthur_two_piccolos_cycle(self):
        # User-observed against a GaRuCrPl (border 16) Poseidon: Arthur survives at
        # 1 HP, Piccolo swaps in and plays until it dies, Arthur returns, and the
        # third-slot Piccolo swaps in again.
        for first_side in (0, 1):
            battle = compile_battle(load_catalog(), ((10, 104, 104), (44,)), borders=((1, 1, 1), (16,)),
                                    red_supports=(0, 0), first_side=first_side)
            events = simulate(battle, Options(mode='sample'), trace=True).trace
            poseidon = next(e for e in events if e['phase'] == 'ENTRY' and e['side'] == 1)
            self.assertEqual(math.ceil(poseidon['attack']), 81_736_447)  # User-read displayed ATK.
            sequence = [(e['phase'], e.get('slot')) for e in events if e['phase'] in (
                'LETHAL_REPLACEMENT', 'INTERCEPT', 'REENTRY', 'DEATH')]
            self.assertEqual(sequence, [
                ('LETHAL_REPLACEMENT', None), ('INTERCEPT', 0), ('DEATH', 0), ('REENTRY', 1),
                ('INTERCEPT', 1), ('DEATH', 1), ('REENTRY', 2), ('DEATH', 2)])
            self.assertTrue(all(e['hp'] == 1 for e in events if e['phase'] == 'REENTRY'))
            self.compare([battle])

    def test_returning_poseidon_keeps_boost_without_reboosting(self):
        # User-observed: Poseidon ahead of Piccolo does not re-boost on return.
        battle = compile_battle(load_catalog(), ((44, 104), (44,)), borders=((1, 1), (16,)),
                                red_supports=(0, 0), first_side=0)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        entry = next(e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0)
        reentry = next(e for e in events if e['phase'] == 'REENTRY')
        self.assertEqual(sum(e['phase'] == 'ENTRY' and e['side'] == 0 for e in events), 1)
        self.assertEqual(reentry['attack'], entry['attack'])
        self.assertAlmostEqual(entry['attack'], 446 * 1.4)
        self.compare([battle])

    def test_returning_knightmare_and_good_boy_repeat_no_entry_ability(self):
        # User-observed: neither repeats its on-entry ability after Piccolo dies.
        catalog = load_catalog()
        for card in (14, 3):
            battle = compile_battle(catalog, ((card, 104), (44,)), borders=((1, 1), (16,)),
                                    red_supports=(0, 0), first_side=0)
            events = simulate(battle, Options(mode='sample'), trace=True).trace
            self.assertIn('REENTRY', [e['phase'] for e in events])
            self.assertEqual(sum(e['phase'] == 'ENTRY' and e['side'] == 0 for e in events), 1)
            self.assertEqual(sum(e['phase'] == 'ENTRY_ATTACK_DECLARE' for e in events), card == 14)
            self.compare([battle])

    def test_chronus_steals_from_swapping_piccolo(self):
        # User-observed: Chronus steals from Piccolo when it swaps in.
        battle = compile_battle(load_catalog(), ((7, 104), (205,)), red_supports=(0, 0))
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        phases = [e['phase'] for e in events]
        theft = events[phases.index('INTERCEPT') + 1]
        self.assertEqual((theft['phase'], theft['entering_side']), ('ENEMY_ENTRY', 0))
        self.assertAlmostEqual(theft['stolen_hp'], 1228.8)  # 10% after 2x of the user-read 3,072 ATK base.
        self.assertAlmostEqual(theft['stolen_attack'], 614.4)
        self.compare([battle])

    def test_thief_never_steals_from_the_same_card_twice(self):
        # User-observed: Chronus does not steal from the same card twice.
        thief = Fighter(1000, 30, enemy_entry_steal_fraction=.1)
        battle = Battle(((Fighter(10, 1), Fighter(5, 1, intercept_lethal_multiplier=2)), (thief,)), 1)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        phases = [e['phase'] for e in events]
        self.assertIn('REENTRY', phases)
        self.assertEqual(phases.count('ENEMY_ENTRY'), 2)  # First card on entry, Piccolo on swap.
        self.assertLess(max(i for i, p in enumerate(phases) if p == 'ENEMY_ENTRY'), phases.index('REENTRY'))
        self.compare([battle])

    def test_true_prophet_platinum_witch_versus_raze(self):
        # User-observed: Platinum (border 2) Witch, 3471 HP, dodges Raze's first hit,
        # survives the counter to her attack, then dies to Raze's next normal attack.
        battle = compile_battle(load_catalog(), ((164, 94), (210,)), borders=((1, 2), (1,)), red_supports=(0, 0))
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        witch = next(e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0 and e['hp'] > 170)
        self.assertEqual(witch['hp'], 3471)
        hits = [(e.get('outcome') or e['phase'], e.get('hp')) for e in events
                if e.get('side') == 0 and (e['phase'] == 'DAMAGE_APPLY' or e.get('outcome') == 'prophet_dodge')]
        self.assertEqual(hits, [('DAMAGE_APPLY', -1908), ('prophet_dodge', None),
                                ('DAMAGE_APPLY', 1393), ('DAMAGE_APPLY', -685)])
        self.compare([battle])

    def test_protection_order_prophet_then_survival_then_piccolo(self):
        # User-stated: Prophet dodge > survive at 1 HP > Piccolo swap.
        catalog = load_catalog()
        battle = compile_battle(catalog, ((164, 10, 104), (44,)), borders=((1, 1, 1), (16,)),
                                red_supports=(0, 0), first_side=1)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        order = [e.get('outcome') or e['phase'] for e in events
                 if e.get('outcome') == 'prophet_dodge' or e['phase'] in ('LETHAL_REPLACEMENT', 'INTERCEPT')]
        self.assertEqual(order, ['prophet_dodge', 'LETHAL_REPLACEMENT', 'INTERCEPT'])
        self.compare([battle])

    def test_true_prophet_next_card_dodges_lethal_raze_hit_then_recovers(self):
        # User-observed: after the Prophet dies, Shu dodges Raze's lethal hit, then recovers.
        battle = compile_battle(load_catalog(), ((164, 74), (210,)), red_supports=(0, 0))
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        phases = [e['phase'] for e in events]
        dodge = next(i for i, e in enumerate(events) if e.get('outcome') == 'prophet_dodge')
        self.assertEqual(events[dodge]['side'], 0)
        self.assertEqual(phases[dodge + 1], 'DAMAGE_RECOVERY')
        self.assertEqual(sum(e.get('outcome') == 'prophet_dodge' for e in events), 1)
        self.assertEqual(phases.count('DEATH'), 2)  # Prophet, then Shu on the next lethal hit.
        self.compare([battle])

    def test_true_prophet_grants_dodge_on_death_to_the_next_card(self):
        # User-stated on-death ability; the dodge applies to the next card's next
        # hit. A card dying ahead of the Prophet grants nothing.
        prophet = Fighter(5, 1, next_card_dodges=1)
        battle = Battle(((Fighter(5, 1), prophet, Fighter(5, 1), Fighter(5, 1)), (Fighter(1000, 10),)), 1)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        grants = [e for e in events if e['phase'] == 'ON_DEATH']
        self.assertEqual([(e['slot'], e['charges']) for e in grants], [(2, 1)])
        phases = [e['phase'] for e in events]
        self.assertLess(phases.index('ON_DEATH'), phases.index('PRE_HIT'))
        self.assertEqual(sum(e.get('outcome') == 'prophet_dodge' for e in events), 1)
        self.assertEqual(phases.count('DEATH'), 4)
        self.compare([battle])

    def test_true_prophet_mirror_dodges_nonlethal_first_hit(self):
        # User-observed: both Shus dodged the first hit (362 of 1230 HP, not lethal)
        # and showed recovery even at full HP, then recovered 70% of later damage.
        battle = compile_battle(load_catalog(), ((164, 74), (164, 74)), red_supports=(0, 0))
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        # Hits on a Shu: Shu's 362 killing a Prophet leaves negative HP.
        shu_hits = [e for e in events if (e['phase'] == 'DAMAGE_APPLY' and e['damage'] == 362 and e['hp'] > 0)
                    or e.get('outcome') == 'prophet_dodge']
        for side in (0, 1):
            self.assertEqual(next(e for e in shu_hits if e['side'] == side).get('outcome'), 'prophet_dodge')
        recoveries = [e for e in events if e['phase'] == 'DAMAGE_RECOVERY']
        self.assertEqual([e['healed'] for e in recoveries[:2]], [0, 0])  # Capped at full HP.
        self.assertEqual({e['amount'] for e in recoveries[2:]}, {254})  # ceil(362 x 0.7)
        self.assertEqual(sum(e.get('outcome') == 'prophet_dodge' for e in events), 2)
        self.compare([battle])

    def test_displaced_actor_ends_its_multi_hit_action_and_turn(self):
        dancer = Fighter(10, 1, attacks_per_action=3, actions_per_turn=2)
        battle = Battle(((dancer, Fighter(50, 10, intercept_lethal_multiplier=2)),
                         (Fighter(1000, 20, counter_on_damage=1),)), 0)
        events = simulate(battle, Options(mode='sample', max_steps=4), trace=True).trace
        declares = [e for e in events if e['phase'] in ('HIT_DECLARE', 'INTERCEPT')]
        # Hit, counter intercepted, then the enemy's normal turn.
        self.assertEqual([(e['phase'], e.get('side'), e.get('kind')) for e in declares[:4]],
                         [('HIT_DECLARE', 0, 'normal'), ('HIT_DECLARE', 1, 'counter'),
                          ('INTERCEPT', 0, None), ('HIT_DECLARE', 1, 'normal')])
        self.compare([battle], Options(max_steps=12))

    def test_randomized_interception_parity(self):
        rng = random.Random(104)
        cases = []
        for _ in range(64):
            teams = []
            for side in range(2):
                teams.append(tuple(Fighter(
                    hp=rng.randint(5, 30), attack=rng.randint(1, 12),
                    dodge_probability=rng.choice((0, 0, .25)),
                    block_mode=rng.randrange(3), counter_on_damage=rng.choice((0, 1)),
                    lethal_survivals=rng.choice((0, 0, 1)), attacks_per_action=rng.choice((1, 1, 3)),
                    critical_probability=rng.choice((0, .5)), heal_on_kill=rng.choice((0, 1)),
                    action_end_attack_multiplier=rng.choice((1, 1.3)), lifetime_actions=rng.choice((0, 0, 3)),
                    intercept_lethal_multiplier=rng.choice((0, 2, 2)),
                    next_card_dodges=rng.choice((0, 1)), heal_damage_taken_fraction=rng.choice((0, .7)),
                    entry_hit_multiplier=rng.choice((0, 0, .5)), entry_self_multiplier=rng.choice((1, 1.4)),
                ) for _ in range(rng.randint(1, 4))))
            cases.append(Battle(tuple(teams), rng.randrange(2)))
        self.assertTrue(any('INTERCEPT' in [e['phase'] for e in simulate(
            b, Options(mode='sample', max_steps=80), trace=True).trace] for b in cases))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=80, max_frontier=64,
                                            repeat_cycles=5, prune_probability=1e-6, seed=11))


if __name__ == '__main__':
    unittest.main()
