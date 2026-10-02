"""Waiting-card mechanics and the user's Academy Student / Seraphim tests.
User: Academy Student's other enemies lose 30% of damage dealt directly;
Seraphim's entry attack hits the whole team for 25%. Waiting cards reduced to 0 HP
die in the deck. The game's display then drops one card from the front per death, so
the shown card is usually an earlier, surviving one (user's theory, which reproduces
every recorded run). Unused True Prophet grants pass on when the holder dies."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'bench.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states', 'estimated'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def test_user_test_1_academy_student_splashes_damage_dealt(self):  # PG Academy behaved normally.
        battle = compile_battle(load_catalog(), ((10, 164, 93, 104), (88,)), borders=((1, 1, 1, 1), (16,)),
                                red_supports=(0, 0), first_side=0)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        first = [e['damage'] for e in events if e['phase'] == 'BENCH_HIT'][:3]
        self.assertEqual(first, [20, 20, 20])  # 30% of the 64 HP Arthur lost, rounded up.
        self.assertAlmostEqual(simulate(battle).p_b, 1, places=6)  # User: the ally team loses.
        self.compare([battle])

    def test_user_inari_test_forest_spirit_dies_in_the_deck(self):
        # User: Arthur survives one lethal hit and dies; the "Forest Spirit" that attacks once is
        # really Arthur at 1 HP (the display dropped Arthur for Forest Spirit's deck death).
        battle = compile_battle(load_catalog(), ((10, 7, 114), (172,)), borders=((1, 1, 1), (8,)),
                                mutations=(("Eclipse", "None", "None"), ("None",)), red_supports=(0, 0), first_side=0)
        self.ids[id(battle)] = (10, 7, 114)
        self.assertEqual(self.story(battle), ['survive', 'dead 7', 'attack 10', 'dead 10', 'attack 114', 'attack 114', 'attack 114', 'dead 114'])
        self.compare([battle])

    def test_inari_invincibility_does_not_carry_over(self):
        # User: [Inari, Forest Spirit] vs Raze - Inari dies on its 3rd turn after Raze's
        # counter, and Forest Spirit dies on Raze's first attack.
        battle = compile_battle(load_catalog(), ((114, 7), (210,)), red_supports=(0, 0), first_side=0)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        deaths = [(e['slot'], e['reason']) for e in events if e['phase'] == 'DEATH' and e['side'] == 0]
        self.assertEqual(deaths[0], (0, 'lifetime'))
        fs_hits = [e for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 0]
        self.assertLessEqual(fs_hits[0]['hp'], 0)  # Raze's first hit on Forest Spirit kills it.
        self.compare([battle])

    def story(self, battle, seed=1):
        """Ally-side events: survivals, deaths (by card ID) and attacks, in order."""
        team = battle.teams[0]
        ids = self.ids.get(id(battle))
        out = []
        for e in simulate(battle, Options(mode='sample', seed=seed), trace=True).trace:
            card = ids[e['slot']] if ids and 'slot' in e else None
            if e['phase'] == 'LETHAL_REPLACEMENT' and e['side'] == 0 and e.get('effect') == 'survive_at_one_hp':
                out.append('survive')
            elif e['phase'] == 'DEATH' and e['side'] == 0:
                out.append(f'dead {card}')
            elif e['phase'] == 'ATTACK_DECLARE' and e['side'] == 0:
                out.append(f'attack {card}')
            elif e['phase'] == 'BENCH_DODGE' and e['side'] == 0:
                out.append(f'deck dodge {card}')
        return out

    def seraphim(self, team, border=16, team_borders=None, front=(), mutations=None):
        battle = compile_battle(load_catalog(), (team, tuple(front) + (172,)),
                                borders=(team_borders or (1,) * len(team), (1,) * len(front) + (border,)),
                                mutations=mutations, red_supports=(0, 0))
        self.ids[id(battle)] = team
        self.compare([battle])
        self.compare([battle], Options(mode='sample', seed=1))
        return battle

    def setUp(self):
        self.ids = {}

    def test_seraphim_user_runs_with_display_shift(self):
        # Each run's shown sequence is the real one with one card dropped from the front per
        # deck death (comments give what the user saw).
        A, FS, IN, MAW = 10, 7, 114, 171
        for border in (12, 15, 16):
            # IMG_0293 / border tests: "Forest Spirit dies on arrival, Inari dies after an attack,
            # Maw lasts 3 turns" = Arthur at 1 HP, then Inari.
            self.assertEqual(self.story(self.seraphim((A, FS, IN, MAW), border)),
                             ['survive', f'dead {FS}', f'dead {MAW}', f'attack {A}', f'dead {A}'] + [f'attack {IN}'] * 3 + [f'dead {IN}'])
        # IMG_0296: "Maw has Inari's ability" = Inari.
        self.assertEqual(self.story(self.seraphim((IN, FS, MAW))), [f'dead {FS}', f'dead {MAW}'] + [f'attack {IN}'] * 3 + [f'dead {IN}'])
        # Test 1: Forest Spirit dies at entry; "Inari loses invincibility" is the display drop.
        self.assertEqual(self.story(self.seraphim((FS, IN, MAW))), [f'dead {MAW}', f'dead {FS}'] + [f'attack {IN}'] * 3 + [f'dead {IN}'])
        # Test 2: "Maw dies to the first attack, Inari normal" = Arthur at 1 HP, then Inari.
        self.assertEqual(self.story(self.seraphim((A, FS, MAW, IN))),
                         ['survive', f'dead {FS}', f'dead {MAW}', f'attack {A}', f'dead {A}'] + [f'attack {IN}'] * 3 + [f'dead {IN}'])
        # Mid-battle controls behind Stone Scientist (Crystal Arthur) and Mrs. Claus (RuCr Arthur).
        for front, arthur in ((84, 3), (259, 7)):
            events = self.story(self.seraphim((A, FS, IN, MAW), 12, (arthur, 1, 1, 1), (front,)))
            self.assertEqual(events[events.index('survive'):events.index('survive') + 3], ['survive', f'dead {FS}', f'dead {MAW}'])
            self.assertEqual(events[-4:], [f'attack {IN}'] * 3 + [f'dead {IN}'])

    def test_seraphim_prophet_grant_goes_to_the_front_card(self):
        # User: "True Prophet enters empty, dodges a lethal hit and dies" (RuCrPl) and Result 0's
        # "Nao dodges once / Piccolo gains Nao's dodge": Prophet dies in the deck, Arthur (front)
        # uses the grant, then Nao plays with her own 60% dodge.
        A, TP, NAO, PIC, MAL = 10, 164, 93, 104, 206
        rucrpl = self.seraphim((A, TP, NAO, MAL), 8, (1, 1, 1, 8), mutations=(("Eclipse", "None", "None", "None"), ("None",)))
        events = self.story(rucrpl)
        self.assertEqual(events[:6], ['survive', f'dead {TP}', f'attack {A}', f'attack {A}', f'dead {A}', f'attack {NAO}'])
        self.assertAlmostEqual(simulate(rucrpl).p_a, 1, places=9)
        result_0 = self.seraphim((A, TP, NAO, PIC))
        self.assertEqual(self.story(result_0)[:5], ['survive', f'dead {TP}', f'dead {PIC}', f'attack {A}', f'attack {A}'])

    def test_front_death_effects_can_be_wasted_on_a_dead_card(self):
        # User: effects go to the next card, dead or alive. Prophet dies at the front after its
        # neighbour died in the deck, so the grant is wasted and the last card cannot dodge.
        prophet, doomed, last = Fighter(10, 1, next_card_dodges=1), Fighter(5, 1), Fighter(30, 1)
        seraph = Fighter(1000, 100, entry_hit_multiplier=.1, entry_hit_all_enemies=1)
        battle = Battle(((prophet, doomed, last), (seraph,)), 1)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        self.assertEqual([(e['slot'], e['reason']) for e in events if e['phase'] == 'DEATH' and e['side'] == 0][:3],
                         [(1, 'bench'), (0, 'damage'), (2, 'damage')])
        self.assertEqual([e.get('slot') for e in events if e.get('effect') == 'grant_dodge'], [1])
        self.compare([battle])

    def test_seraphim_mrs_claus_deck_death_buffs_the_front_card(self):
        # User: [Arthur, Mrs. Claus, Maw, Noveau] - the shown "Mrs. Claus" had 66 ATK, double
        # Arthur's 33: her deck death doubled the card still playing. Noveau then revives twice.
        A, CL, MAW, NOV = 10, 259, 171, 169
        battle = self.seraphim((A, CL, MAW, NOV))
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        deaths = [e for e in events if e['phase'] == 'DEATH']
        self.assertEqual([(e['side'], e['slot'], e['reason']) for e in deaths][:2], [(0, 1, 'bench'), (0, 2, 'bench')])
        story = self.story(battle)
        self.assertEqual(story.count('survive'), 4)  # Arthur once, Noveau at entry and twice more.
        self.assertEqual(story[-1], f'dead {NOV}')
        gift = next(e for e in events if e.get('effect') == 'scale_stats')
        self.assertEqual((gift['slot'], gift['attack']), (0, 66))  # Arthur, the card still playing.

    def test_seraphim_deus_ex_and_noveau(self):
        # User Deus Ex runs: Deus Ex always dodges the entry while waiting (the 2nd and 3rd shown
        # cards, really Arthur and Deus Ex, always survive); Maw's deck death causes the display
        # shift. Prophet's grant (Prophet died in the deck) goes to Arthur.
        A, TP, DE, MAW, PIC = 10, 164, 54, 171, 104
        for seed in (1, 2, 3):
            self.assertEqual(self.story(self.seraphim((A, TP, DE, MAW)), seed=seed),
                             ['survive', f'deck dodge {DE}', f'dead {TP}', f'dead {MAW}', f'attack {A}', f'attack {A}',
                              f'dead {A}', f'attack {DE}', f'dead {DE}'])
        self.assertEqual(self.story(self.seraphim((A, DE, MAW, PIC)))[:4], ['survive', f'deck dodge {DE}', f'dead {MAW}', f'dead {PIC}'])
        # Without Maw nothing dies in the deck, so Arthur is shown surviving at 1 HP.
        self.assertEqual(self.story(self.seraphim((A, DE)))[:3], ['survive', f'deck dodge {DE}', f'attack {A}'])
        # [Arthur, Raze, Noveau, RuCrPl Malik]: the entry uses one of Noveau's three revives.
        RAZE, NOV, JD = 210, 169, 50
        gacrpl = self.story(self.seraphim((A, RAZE, NOV, MAL := 206), 12, (1, 1, 1, 8)))
        self.assertEqual(gacrpl[:5], ['survive', 'survive', f'dead {RAZE}', f'attack {A}', f'dead {A}'])
        self.assertEqual(gacrpl.count('survive'), 4)  # Arthur once, Noveau three times.
        self.assertIn(f'attack {MAL}', gacrpl)  # "Exchanges with Malik normally."
        pg = self.story(self.seraphim((A, RAZE, NOV, MAL), 16, (1, 1, 1, 8)))
        self.assertEqual(pg[:4], ['survive', 'survive', f'dead {RAZE}', f'dead {MAL}'])
        # Judgement Day dies in the deck, so its infinite damage never procs.
        jd = self.story(self.seraphim((A, RAZE, NOV, JD)))
        self.assertEqual(jd[:4], ['survive', 'survive', f'dead {RAZE}', f'dead {JD}'])
        self.assertNotIn(f'attack {JD}', jd)

    def test_parallax_reflects_one_lethal_hit_onto_the_attacker(self):
        parallax = Fighter(50, 5, death_reflects=1)
        battle = Battle(((parallax, Fighter(10, 1)), (Fighter(1000, 60), Fighter(1000, 60))), 1)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        deaths = [(e['side'], e['slot']) for e in events if e['phase'] == 'DEATH']
        self.assertEqual(deaths[0], (1, 0))  # The first attacker dies instead of Parallax.
        self.assertIn((0, 0), deaths[1:])  # Only once: the next lethal hit kills Parallax.
        self.compare([battle])
        # User: #248's mutual destruction still kills both, even against Parallax.
        mutual = Battle(((Fighter(100, 40, death_reflects=1),), (Fighter(100, 50, mutual_destruction_below_max_hp=.3),)), 0)
        self.assertEqual(simulate(mutual, Options(mode='sample', max_steps=4)).p_b + simulate(mutual, Options(mode='sample', max_steps=4)).p_a, 1)
        self.compare([mutual])

    def test_pg_parallax_runs(self):
        # User (all won): Seraphim dies to Parallax's deck reflect and Arthur survives at 1 HP.
        # Where Malik also dies in the deck, the display drops Arthur ("Arthur dies").
        A, IN, PAR, MAL = 10, 114, 65, 206
        catalog = load_catalog()
        inari_parallax = compile_battle(catalog, ((IN, PAR), (172,)), borders=((1, 1), (16,)), red_supports=(0, 0))
        self.assertEqual(simulate(inari_parallax).p_a, 1)  # User: Inari and Parallax defeat PG Seraphim.
        self.compare([inari_parallax])
        for team, arthur_survives in (((A, IN, PAR), True), ((A, IN, PAR, MAL), True), ((A, IN, MAL, PAR), True)):
            borders = ((1, 1, 1, 8) if MAL in team and team.index(MAL) == 3 else
                       (1, 1, 8, 1) if MAL in team else (1, 1, 1), (16,))
            battle = compile_battle(catalog, (team, (172,)), borders=borders, red_supports=(0, 0))
            events = simulate(battle, Options(mode='sample'), trace=True).trace
            self.assertIn('death_reflect', [e.get('effect') for e in events])
            self.assertEqual(simulate(battle).p_a, 1)
            arthur_died = any(e['phase'] == 'DEATH' and e['side'] == 0 and e['slot'] == 0 for e in events)
            self.assertEqual(arthur_died, not arthur_survives)
            self.compare([battle])

    def test_cards_reduced_to_zero_in_the_deck_die_there(self):
        attacker = Fighter(1000, 90, entry_hit_multiplier=.5, entry_hit_all_enemies=1)
        battle = Battle(((Fighter(100, 1), Fighter(20, 5), Fighter(200, 1, invincible=1, lifetime_actions=2)), (attacker,)), 1)
        events = simulate(battle, Options(mode='sample'), trace=True).trace
        self.assertEqual([(e['slot'], e['hp']) for e in events if e['phase'] == 'BENCH_HIT'], [(1, -25)])
        self.assertEqual([(e['slot'], e['reason']) for e in events if e['phase'] == 'DEATH'], [(1, 'bench'), (0, 'damage'), (2, 'lifetime')])
        self.assertFalse(any(e['phase'] == 'ATTACK_DECLARE' and e['side'] == 0 and e['slot'] == 1 for e in events))
        self.compare([battle])

    def test_monte_carlo_playouts_resolve_pruned_mass(self):
        # User suggestion: average Monte Carlo playouts for low-probability paths,
        # weighted by each pruned path's probability.
        battle = compile_battle(load_catalog(), ((10, 164, 93, 104), (88,)), borders=((1, 1, 1, 1), (16,)),
                                red_supports=(0, 0), first_side=0)
        exact = simulate(battle)
        pruned = simulate(battle, Options(prune_probability=1e-4))
        playouts = simulate(battle, Options(prune_probability=1e-4, rollouts=20))
        self.assertGreater(pruned.unresolved, 1e-4)
        self.assertEqual(playouts.unresolved, 0)
        self.assertAlmostEqual(playouts.estimated, pruned.unresolved, places=12)
        self.assertAlmostEqual(playouts.p_b, exact.p_b, places=4)
        self.assertLess(playouts.expanded_states, exact.expanded_states)
        self.compare([battle], Options(prune_probability=1e-4, rollouts=20))
        self.compare([battle], Options(max_frontier=8, rollouts=5, seed=7))

    def test_randomized_bench_parity(self):
        rng = random.Random(88)
        cases = []
        for _ in range(48):
            teams = []
            for side in range(2):
                teams.append(tuple(Fighter(
                    hp=rng.randint(10, 60), attack=rng.randint(1, 15), dodge_probability=rng.choice((0, 0, .5)),
                    lethal_survivals=rng.choice((0, 0, 1)), next_card_dodges=rng.choice((0, 0, 1)),
                    intercept_lethal_multiplier=rng.choice((0, 0, 2)), invincible=rng.choice((0, 0, 0, 1)),
                    lifetime_actions=rng.choice((0, 3)), entry_hit_multiplier=rng.choice((0, .25, .5)),
                    entry_hit_all_enemies=rng.choice((0, 1)), splash_damage_fraction=rng.choice((0, 0, .4)),
                    next_card_stat_add_fraction=rng.choice((0, 0, .5)), alternate_dodge=rng.choice((0, 0, 1)),
                    lethal_survival_hp_fraction=rng.choice((0, 0, 1)),
                ) for _ in range(rng.randint(1, 4))))
            cases.append(Battle(tuple(teams), rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=100, max_frontier=64,
                                            repeat_cycles=6, prune_probability=1e-6, seed=29))


if __name__ == '__main__':
    unittest.main()
