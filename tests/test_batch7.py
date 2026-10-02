"""Batch 7 primitives (extrapolated from card text, provisional until user-tested):
death blasts, periodic specials, follow-up hits, encounter statuses, low-HP extra
actions and the friendship bonus."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch


class BatchSevenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'batch7.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states', 'estimated'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def events(self, battle, seed=1):
        return simulate(battle, Options(mode='sample', seed=seed), trace=True).trace

    def test_death_blast_hits_the_killer_and_boosts_allies(self):
        savior = Fighter(50, 1, death_damage_max_hp_fraction=1, death_ally_stat_multiplier=1.2)
        battle = Battle(((savior, Fighter(10, 1)), (Fighter(200, 60),)), 1)
        events = self.events(battle)
        blast = next(e for e in events if e['phase'] == 'DEATH_BLAST')
        self.assertEqual((blast['damage'], blast['hp']), (50, 150))
        entry = [e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0][-1]
        self.assertEqual((entry['hp'], entry['attack']), (12, 1.2))
        self.compare([battle])

    def test_frankenstein_blast_is_a_coin_flip(self):
        frank = Fighter(100, 1, death_damage_max_hp_fraction=1, death_damage_chance=.5)
        battle = Battle(((frank,), (Fighter(150, 200),)), 1)
        self.assertAlmostEqual(simulate(battle).p_b, 1)  # The blast alone cannot win.
        tough = Battle(((frank, Fighter(10, 60)), (Fighter(150, 200),)), 1)
        self.assertAlmostEqual(simulate(tough).p_a, .5)
        self.compare([battle, tough])

    def test_periodic_special_every_third_turn(self):
        hollow = Fighter(10**6, 10, periodic_attack_period=3, periodic_attack_multiplier=2, periodic_attack_heal_max_hp_fraction=.25)
        battle = Battle(((hollow,), (Fighter(10**6, 1),)), 0)
        hits = [e['damage'] for e in self.events(battle) if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1][:6]
        self.assertEqual(hits, [10, 10, 20, 10, 10, 20])
        melanin = Fighter(10**6, 10, periodic_attack_period=3, periodic_attack_hits=3, periodic_attack_multiplier=.5)
        battle = Battle(((melanin,), (Fighter(10**6, 1),)), 0)
        hits = [e['damage'] for e in self.events(battle) if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1][:5]
        self.assertEqual(hits, [10, 10, 5, 5, 5])
        susanoo = Fighter(10**6, 10, periodic_attack_period=3, periodic_attack_multiplier=2, periodic_enemy_slow_turns=1)
        battle = Battle(((susanoo,), (Fighter(10**6, 1),)), 0)
        self.assertIn('SKIP', [e['phase'] for e in self.events(battle)][:40])
        self.compare([Battle(((hollow,), (Fighter(300, 30),)), 0), Battle(((melanin,), (Fighter(300, 30),)), 1),
                      Battle(((susanoo,), (Fighter(300, 30),)), 0)])

    def test_followups(self):
        sorceror = Fighter(1000, 10, followup_multiplier=.5, followup_bypass=1)
        dodger = Fighter(10**6, 1, dodge_probability=.5)
        events = self.events(Battle(((sorceror,), (dodger,)), 0), seed=3)
        declares = [e for e in events if e['phase'] == 'HIT_DECLARE' and e['side'] == 0]
        self.assertEqual([e['hit_index'] for e in declares[:4]], [1, 2, 1, 2])
        followups = [e for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1 and e['hit_index'] == 2]
        self.assertTrue(followups and all(e['damage'] == 5 for e in followups))
        durante = Fighter(1000, 10, critical_probability=.3, critical_multiplier=1.5, followup_multiplier=.5, followup_on_critical=1)
        for seed in range(1, 6):
            events = self.events(Battle(((durante,), (Fighter(10**6, 1),)), 0), seed=seed)
            applied = [e for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1]
            for first, second in zip(applied, applied[1:]):
                if second['hit_index'] == 2:
                    self.assertTrue(first['critical'])  # Follow-ups only after critical strikes.
        self.compare([Battle(((sorceror,), (Fighter(200, 30, dodge_probability=.5),)), 0),
                      Battle(((durante,), (Fighter(200, 30),)), 1)])

    def test_encounter_freeze_and_frozen_bonus(self):
        ice_king = Fighter(1000, 10, encounter_freeze_turns=1, frozen_target_multiplier=2)
        battle = Battle(((ice_king,), (Fighter(15, 5), Fighter(1000, 5))), 0)
        events = self.events(battle)
        self.assertEqual([e['damage'] for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1][0], 20)
        skips = [(e['side'], e['slot']) for e in events if e['phase'] == 'SKIP']
        self.assertIn((1, 1), skips)  # The next enemy card is frozen on arrival too.
        self.compare([battle])

    def test_low_hp_extra_actions(self):
        tartarus = Fighter(100, 10, low_hp_extra_actions=2, low_hp_actions_threshold=.5)
        battle = Battle(((tartarus,), (Fighter(1000, 30),)), 1)
        declares = [e['side'] for e in self.events(battle) if e['phase'] == 'ATTACK_DECLARE']
        self.assertIn([0, 0, 0], [declares[i:i + 3] for i in range(len(declares))])
        self.compare([battle])

    def test_catalog_cards_compile_and_match(self):
        catalog = load_catalog()
        friends = compile_battle(catalog, ((109, 110), (10,)), red_supports=(0, 0))
        alone = compile_battle(catalog, ((109,), (10,)), red_supports=(0, 0))
        self.assertAlmostEqual(friends.teams[0][0].entry_attack_multiplier, 1.4)  # +40% for the other friendship card
        self.assertAlmostEqual(alone.teams[0][0].entry_attack_multiplier, 1.0)  # user: a lone one gets nothing
        rng = random.Random(7)
        cards = (4, 16, 26, 27, 80, 95, 109, 110, 111, 116, 184, 189, 192, 194, 218, 269)
        others = (10, 22, 31, 93, 104, 114, 164, 172, 210)
        battles = [compile_battle(catalog, (tuple(rng.sample(cards, 3)), tuple(rng.sample(others, 3))),
                                  red_supports=(0, 0), first_side=rng.randrange(2)) for _ in range(12)]
        for mode in ('sample', 'branch'):
            self.compare(battles, Options(mode=mode, max_steps=400, prune_probability=1e-7, seed=11))

    def test_randomized_parity(self):
        rng = random.Random(77)
        cases = []
        for _ in range(40):
            teams = tuple(tuple(Fighter(
                hp=rng.randint(10, 80), attack=rng.randint(1, 15), dodge_probability=rng.choice((0, 0, .5)),
                critical_probability=rng.choice((0, .3)), critical_multiplier=1.5,
                death_damage_max_hp_fraction=rng.choice((0, 0, .5, 1)), death_damage_chance=rng.choice((1, .5)),
                death_ally_stat_multiplier=rng.choice((1, 1, 1.2)), periodic_attack_period=rng.choice((0, 0, 2, 3)),
                periodic_attack_multiplier=rng.choice((1, 2, .5)), periodic_attack_hits=rng.choice((0, 0, 3)),
                periodic_attack_heal_max_hp_fraction=rng.choice((0, .25)), periodic_enemy_slow_turns=rng.choice((0, 1)),
                followup_multiplier=rng.choice((0, 0, .5, 2)), followup_bypass=rng.choice((0, 1)), followup_on_critical=rng.choice((0, 1)),
                encounter_freeze_turns=rng.choice((0, 0, 1)), encounter_confusion_turns=rng.choice((0, 0, 3)),
                frozen_target_multiplier=rng.choice((1, 2)), low_hp_extra_actions=rng.choice((0, 0, 2)), low_hp_actions_threshold=.5,
                entry_hit_multiplier=rng.choice((0, .25)), entry_hit_all_enemies=rng.choice((0, 1)), ratio_kill=rng.choice((0, 0, 1)),
            ) for _ in range(rng.randint(1, 4))) for _ in range(2))
            cases.append(Battle(teams, rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=120, max_frontier=64, repeat_cycles=6, prune_probability=1e-6, seed=5))


if __name__ == '__main__':
    unittest.main()
