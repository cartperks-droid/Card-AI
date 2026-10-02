"""Batch-1 extrapolated primitives. These encode card-text readings, not game
observations; the tests pin semantics and Python/C parity only."""

import random
import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_battle
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch, _BATCH1_FIELDS

BATCH1_CARDS = (11, 13, 17, 34, 39, 46, 58, 60, 66, 70, 78, 86, 91, 112, 115, 127, 160, 163, 171, 176, 180,
                185, 207, 209, 213, 219, 222, 226, 229, 238, 241, 255, 256, 262, 266, 271, 282)


class BatchOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'batch1.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battles, options=Options()):
        for index, (py, c) in enumerate(zip(simulate_batch(battles, options),
                                            native.simulate_batch(battles, options, library_path=self.library), strict=True)):
            for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states'):
                self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=f'battle {index} {key}')

    def events(self, a, b, first_side=0, steps=20):
        return simulate(Battle((tuple(a), tuple(b)), first_side), Options(mode='sample', max_steps=steps), trace=True).trace

    def test_every_batch_card_runs_in_both_engines(self):
        catalog = load_catalog()
        battles = [compile_battle(catalog, ((card, 7), (opponent, 3)), red_supports=(0, 0), first_side=side)
                   for card in BATCH1_CARDS for opponent in (73, 210, 94) for side in (0, 1)]
        self.compare(battles, Options(max_frontier=200, prune_probability=1e-9))

    def test_retaliation_can_kill_the_attacker_and_ends_its_action(self):
        events = self.events([Fighter(5, 3, attacks_per_action=3)], [Fighter(100, 10, thorns_attack_fraction=1)])
        phases = [e['phase'] for e in events]
        self.assertEqual(phases.count('HIT_DECLARE'), 1)
        self.assertEqual(next(e for e in events if e['phase'] == 'DEATH')['reason'], 'retaliation')

    def test_turn_start_drain_can_kill_and_skips_the_turn(self):
        events = self.events([Fighter(10, 1), Fighter(10, 1)], [Fighter(100, 1, enemy_turn_start_max_hp_loss_fraction=1)],
                             first_side=0, steps=3)
        death = next(e for e in events if e['phase'] == 'DEATH')
        self.assertEqual(death['reason'], 'turn_start')
        self.assertEqual(events[events.index(death) + 3]['side'], 1)  # TRANSITION, ENTRY, then enemy acts.

    def test_kill_rewards_steal_then_grow_then_heal(self):
        killer = Fighter(100, 50, kill_steal_fraction=.5, kill_attack_multiplier=2, kill_heal_max_hp_fraction=1)
        events = self.events([killer], [Fighter(40, 10), Fighter(1000, 1)], steps=3)
        heal = next(e for e in events if e['phase'] == 'ON_KILL_HEAL')
        self.assertEqual((heal['max_hp'], heal['hp']), (120, 120))  # +50% of 40 max HP, then healed.
        second_hit = [e for e in events if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 1][1]
        self.assertEqual(second_hit['damage'], 110)  # (50 + 50% of 10) x 2.

    def test_user_answers_for_171_and_91(self):
        catalog = load_catalog()
        battle = compile_battle(catalog, ((91,), (171,)), red_supports=(0, 0), first_side=1)
        events = simulate(battle, Options(mode='sample', max_steps=4), trace=True).trace
        # User: Infected Maw (#171)'s text is right - the enemy loses 25% of its max HP each turn
        # (Maw seeming to lose its own HP was a display bug).
        drain = next(e for e in events if e['phase'] == 'TURN_START_DRAIN')
        entry = next(e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0)
        self.assertEqual((drain['side'], drain['damage']), (0, -(-entry['max_hp'] * .25 // 1)))
        # #91 at full HP still gains HP from damage dealt, above its starting max.
        hits = self.events([Fighter(10, 5, hit_hp_gain_fraction=1)], [Fighter(100, 1)], steps=2)
        self.assertEqual(max(e['hp'] for e in hits if e['phase'] == 'DAMAGE_APPLY' and e['side'] == 0), 14)
        self.compare([battle])

    def test_vampire_lord_drains_attack_up_to_the_targets(self):
        # IMG_0336: 5,410 ATK Vampire Lord hit a 2,705 ATK card; the card fell to 0 ATK and Vampire Lord gained 2,705.
        from card_engine.simulator.reference import _advance, _initial
        battle = Battle(((Fighter(10**6, 5410, hit_attack_drain_fraction=1),), (Fighter(10**6, 2705),)), 0)
        state, _ = _advance(battle, _initial(battle, Options(), None), False, Options(), None)
        self.assertEqual((state.attack[0][0], state.attack[1][0]), (8115, 0))
        self.compare([battle])

    def test_limitless_rivals_gains_a_fixed_amount_each_turn(self):
        # User: a fixed gain per turn, sized by buffs/debuffs taken before (IMG_0336: 2,705 -> 2,976 -> 3,246 -> 3,517 -> 3,787).
        from card_engine.simulator.reference import _advance, _initial
        for start, step in ((2705, 270.5), (1623, 162.3)):  # 1,623: a Gunslinger-disarmed start.
            battle = Battle(((Fighter(1000, start, turn_start_stat_growth=.1, turn_start_incoming_reduction=.1, incoming_reduction_cap=.5),),
                             (Fighter(10**9, 1, never_attacks=1),)), 0)
            state = _initial(battle, Options(), None)
            seen = []
            for _ in range(16):
                state, _ = _advance(battle, state, False, Options(), None)
                seen.append((state.attack[0][0], state.incoming_scale[0][0]))
            gains = sorted(set(seen))
            self.assertEqual([round(a, 6) for a, _ in gains[:4]], [round(start + step * n, 6) for n in range(1, 5)])
            self.assertEqual(sorted({round(i, 9) for _, i in seen}, reverse=True), [.9, .8, .7, .6, .5])  # Capped at 50%.
            self.assertEqual(state.max_hp[0][0] - state.hp[0][0], 0)  # The max HP gain also adds to current HP.
            self.compare([battle])

    def test_witch_steals_half_the_attack_half_the_time(self):
        # User: 50% chance; IMG_0336 round 8: Pterodactylus 2,705 -> 1,353, Witch 2,705 -> 4,058.
        from card_engine.simulator.catalog_rules import compile_fighter
        witch = compile_fighter(load_catalog(), 268, 1)
        self.assertEqual((witch.entry_steal_attack_fraction, witch.entry_steal_attack_chance), (.5, .5))
        battle = Battle(((Fighter(10**6, 2705), Fighter(10**6, 2705, entry_steal_attack_fraction=.5, entry_steal_attack_chance=.5)),
                         (Fighter(10, 1), Fighter(10**6, 2705))), 0)
        self.compare([battle])

    def test_next_card_gifts_add_dead_cards_max_stats(self):
        events = self.events([Fighter(10, 4, next_card_stat_add_fraction=.5), Fighter(10, 2)], [Fighter(100, 20)], first_side=1)
        second = [e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0][-1]
        self.assertEqual((second['hp'], second['attack']), (15, 4))

    def test_randomized_batch_one_parity(self):
        rng = random.Random(2026)
        choices = {name: ((1, 1, .8, 1.25) if 'multiplier' in name else (0, 0, .2, .5)) for name in _BATCH1_FIELDS}
        cases = []
        for _ in range(64):
            teams = []
            for side in range(2):
                teams.append(tuple(Fighter(
                    hp=rng.randint(10, 40), attack=rng.randint(1, 12),
                    dodge_probability=rng.choice((0, 0, .25)), counter_on_damage=rng.choice((0, 1)),
                    attacks_per_action=rng.choice((1, 1, 2)), heal_damage_dealt_fraction=rng.choice((0, .5)),
                    intercept_lethal_multiplier=rng.choice((0, 0, 2)), next_card_dodges=rng.choice((0, 1)),
                    entry_hit_multiplier=rng.choice((0, 0, .5)), heal_on_kill=rng.choice((0, 1)),
                    **{name: rng.choice(values) for name, values in choices.items() if rng.random() < .3},
                ) for _ in range(rng.randint(1, 4))))
            cases.append(Battle(tuple(teams), rng.randrange(2)))
        for mode in ('sample', 'branch'):
            with self.subTest(mode=mode):
                self.compare(cases, Options(mode=mode, max_steps=120, max_frontier=64,
                                            repeat_cycles=6, prune_probability=1e-6, seed=5))


if __name__ == '__main__':
    unittest.main()
