"""Batch 12: the last 56 cards (user answers 2026-09-30/10-01; IMG_0304, 0310, 0311, 0327-0341).

Behaviour checks for the mechanics with clear user or video evidence, then Python/C parity for every card."""

import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import SUPPORTED, ability_pool, compile_battle
from card_engine.simulator.reference import (POOL_CODE, POOL_GLAMOUR, POOL_LIMITED, Battle, Fighter, Options, _advance,
                                             _initial, simulate)

BATCH12 = (30, 37, 43, 48, 53, 56, 76, 77, 90, 98, 99, 101, 117, 119, 120, 122, 125, 126, 129, 130, 133, 138, 139, 143, 144,
           152, 175, 177, 186, 195, 199, 203, 204, 214, 215, 224, 225, 228, 230, 233, 234, 236, 239, 240, 250, 254, 265, 270,
           272, 273, 277, 278, 281, 283, 286, 289)
OPPONENTS = ((1, 18), (210,), (104, 21), (27, 146, 52), (247, 79))
RANDOM_POOL = {37, 43, 130, 289}  # Pool draws: branch mode would enumerate ~290 outcomes per draw.


class Batch12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'batch12.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, battle, options):
        keys = ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states')
        py, c = simulate(battle, options), native.simulate(battle, options, library_path=self.library)
        for key in keys:
            self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=key)

    def battle(self, mine, theirs, **kw):
        return compile_battle(self.catalog, (mine, theirs), red_supports=(0, 0), **kw)

    def test_every_card_is_mapped(self):
        self.assertEqual(len(SUPPORTED), 289)
        self.assertTrue(set(BATCH12) <= set(SUPPORTED))

    def test_parity_for_every_new_card(self):
        for index, card in enumerate(BATCH12):
            for opponents in OPPONENTS[index % 2::2][:2]:
                battle = self.battle((card, 18), opponents, first_side=index % 2)
                with self.subTest(card=card, opponents=opponents):
                    self.compare(battle, Options(mode='sample', seed=index + 1, max_steps=300))
                    if card not in RANDOM_POOL:
                        self.compare(battle, Options(mode='branch', max_steps=60, max_frontier=200, prune_probability=1e-3, rollouts=1))

    def test_nuwa_summons_a_random_card_with_its_stats(self):
        # User: any pack except Limited and Glamour; IMG_0336: summons carry Nuwa's stats.
        battle = self.battle((130,), (210,))
        pool = ability_pool(self.catalog)
        for seed in range(1, 6):
            state = _initial(battle, Options(mode='sample', seed=seed), None, _sample_chance(seed))
            self.assertEqual(len(state.order[0]), 2)
            summoned = state.order[0][1]
            code = state.ability[0][summoned]
            self.assertGreaterEqual(code, POOL_CODE)
            self.assertFalse(pool[code - POOL_CODE].pool_flags & (POOL_LIMITED | POOL_GLAMOUR))
            self.assertEqual((state.hp[0][summoned], state.attack[0][summoned]), (state.max_hp[0][0], state.attack[0][0]))

    def test_fresno_copies_have_35_percent_and_no_ability(self):
        battle = self.battle((195,), (210,))
        state = _initial(battle, Options(), None)
        self.assertEqual(len(state.order[0]), 3)
        for copy in state.order[0][1:]:
            self.assertAlmostEqual(state.max_hp[0][copy], state.max_hp[0][0] * .35, delta=1)
            self.assertEqual(state.ability[0][copy], 0)  # a blank spare: no ability (user: they do not summon)

    def test_broken_one_splits_into_joy_and_sorrow(self):
        battle = self.battle((230,), (210,), borders=[[1], [5]], first_side=1)
        events = simulate(battle, Options(mode='sample', seed=3, max_steps=20), trace=True).trace
        entries = [e for e in events if e['phase'] == 'ENTRY' and e['side'] == 0]
        self.assertGreaterEqual(len(entries), 2)
        first = entries[0]
        self.assertAlmostEqual(entries[1]['max_hp'], first['max_hp'] * .5, delta=1)  # Joy: half its max HP

    def test_sleep_paralysis_stuns_then_both_cards_die(self):
        # IMG_0345 (twice, entering first and mid-battle): it hits, the stunned enemy skips, then they trade hits
        # until both active cards die at the start of its 4th turn.
        sp = Fighter(10**6, 1, perish_turns=3, entry_slow_turns=1)
        enemy = Fighter(10**6, 1)
        battle = Battle(((sp,), (enemy,)), 0)
        state = _initial(battle, Options(), None)
        hits, outcome = [], None
        for _ in range(10):
            before = (state.hp[0][0], state.hp[1][0])
            state, outcome = _advance(battle, state, False, Options(), None)
            if outcome is not None:
                break
            hits.append('S' if state.hp[1][0] < before[1] else 'E' if state.hp[0][0] < before[0] else '-')
        self.assertEqual(hits, ['S', '-', 'S', 'E', 'S', 'E'])
        self.assertEqual(outcome, 1)  # both die: the initiator (side 0) needs a survivor
        self.compare(battle, Options(mode='branch', max_steps=40))

    def test_hanged_man_curses_its_killer(self):
        battle = Battle(((Fighter(10**4, 10**4),), (Fighter(10, 1, curse_killer=1), Fighter(10**6, 1))), 0)
        state = _initial(battle, Options(), None)
        state, _ = _advance(battle, state, False, Options(), None)
        self.assertTrue(state.ext[8][0][0])  # curse array (_EXT_CARD order)
        before = state.hp[0][0]
        state, _ = _advance(battle, state, False, Options(), None)  # enemy turn
        state, _ = _advance(battle, state, False, Options(), None)  # killer's turn: 25% max HP lost
        self.assertEqual(before - state.hp[0][0], 2500 + 1)

    def test_mist_spirit_dodges_cards_with_a_status(self):
        attacker = Fighter(100, 10, hit_poison_turns=3, poison_fraction=1)
        mist = Fighter(100, 1, status_dodge_max=5)
        battle = Battle(((mist,), (attacker,)), 1)
        state = _initial(battle, Options(), None)
        state, _ = _advance(battle, state, False, Options(), None)  # attacker hits (no status yet)
        self.assertEqual(state.hp[0][0], 90)
        state = _poisoned(state)
        state, _ = _advance(battle, state, False, Options(), None)  # mist's turn
        state, _ = _advance(battle, state, False, Options(), None)  # poisoned attacker cannot hit
        self.assertEqual(state.hp[0][0], 90)

    def test_tsukuyomi_reflects_even_enemy_turns(self):
        battle = Battle(((Fighter(10**6, 1, reflect_even_turns=1),), (Fighter(10**6, 100),)), 1)
        state = _initial(battle, Options(), None)
        hits = []
        for _ in range(4):
            before = (state.hp[0][0], state.hp[1][0])
            state, _ = _advance(battle, state, False, Options(), None)
            hits.append((before[0] - state.hp[0][0], before[1] - state.hp[1][0]))
        self.assertEqual(hits[0], (100, 0))  # 1st enemy turn: normal
        self.assertEqual(hits[2], (0, 100))  # 2nd enemy turn: the attack lands on itself

    def test_anubis_revives_at_the_back_when_an_ally_dies(self):
        battle = self.battle((76, 18, 21), (210,), borders=[[1, 1, 1], [7]], first_side=1)
        events = simulate(battle, Options(mode='sample', seed=2, max_steps=40), trace=True).trace
        self.assertTrue(any(e['phase'] == 'REVIVE' for e in events))

    def test_longmu_ability_depends_on_its_position(self):
        first = self.battle((234, 18), (210,))
        later = self.battle((18, 234), (210,))
        self.assertEqual(first.teams[0][0].block_mode, 1)  # Safeguarding
        self.assertEqual(later.teams[0][1].team_class_shields, 2)  # Mother of Dragons

    def test_glamour_applies_each_forms_entry_effects(self):
        # IMG_0330: Toy Bear doubled Glamour's 7,680 ATK on the change; Slum Dweller's entry attack came first.
        glamour = Fighter(1000, 100, turn_start_random_ability=1)
        doubles = Fighter(1, 0, entry_attack_multiplier=2)
        battle = Battle(((glamour,), (Fighter(10**6, 1),)), 0, pool=(doubles,))
        state = _initial(battle, Options(), None)
        state, _ = _advance(battle, state, False, Options(), None)
        self.assertEqual(state.attack[0][0], 200)
        hitter = Fighter(1, 0, entry_hit_multiplier=1)
        battle = Battle(((glamour,), (Fighter(10**6, 1),)), 0, pool=(hitter,))
        state = _initial(battle, Options(), None)
        state, _ = _advance(battle, state, False, Options(), None)
        self.assertEqual((state.entry_queue, state.hp[1][0]), ((0,), 10**6))  # the entry attack is queued first
        state, _ = _advance(battle, state, False, Options(), None)
        self.assertEqual(state.hp[1][0], 10**6 - 100)
        state, _ = _advance(battle, state, False, Options(), None)  # then its normal attack, without a second change
        self.assertEqual(state.hp[1][0], 10**6 - 200)
        self.compare(battle, Options(mode='sample', seed=4, max_steps=50))

    def test_awakened_one_weapons(self):
        # IMG_0342: the six Six Realms weapons; Great Nirvana Sword - Zero gives 2 of the other 5.
        pool = ability_pool(self.catalog)
        names = [pool[c - POOL_CODE] for c in range(POOL_CODE + SUPPORTED_COUNT, POOL_CODE + len(pool))]
        vajra = Fighter(10**6, 1, parry_chance=1.0, parry_reflect_fraction=.75)
        battle = Battle(((Fighter(1000, 400),), (vajra,)), 0)
        state = _initial(battle, Options(), None)
        state, _ = _advance(battle, state, False, Options(), None)
        self.assertEqual((state.hp[1][0], state.hp[0][0]), (10**6, 1000 - 300))  # parried; 75% of 400 reflected
        scythe = Fighter(100, 50, entry_hit_multiplier=1, entry_bypass_defenses=1, entry_hit_also_next=1)
        battle = Battle(((scythe,), (Fighter(1000, 1), Fighter(1000, 1))), 0)
        state = _initial(battle, Options(), None)
        state, _ = _advance(battle, state, False, Options(), None)
        self.assertEqual((state.hp[1][0], state.hp[1][1]), (950, 950))  # the first 2 enemies
        self.assertTrue(names)
        for seed in range(1, 4):
            self.compare(self.battle((129, 18), (210, 21)), Options(mode='sample', seed=seed, max_steps=200))

    def test_buddha_revives_at_the_back_with_full_hp(self):
        battle = Battle(((Fighter(100, 1, buddha=1), Fighter(500, 1)), (Fighter(10**6, 1),)), 0)
        state = _initial(battle, Options(), None)
        from dataclasses import replace
        state = replace(state, hp=((100.0, 0.0), (10.0**6,)))
        state, _ = _advance(battle, state, False, Options(), None)
        self.assertEqual(state.hp[0][1], 500)

    def test_astraeus_arts_are_separate_cards(self):
        # User: each art is its own card named Astraeus; the nullifying sign's threshold drops 15% per nullification.
        battle = self.battle((56,), (210,), arts={(0, 0): 'cancer'})
        self.assertEqual((battle.teams[0][0].nullify_max_hp_threshold, battle.teams[0][0].form_count), (1.0, 0))
        nullifier = Fighter(1000, 1, nullify_max_hp_threshold=1.0, nullify_threshold_step=.15)
        battle = Battle(((nullifier,), (Fighter(10**6, 800),)), 1)
        state = _initial(battle, Options(), None)
        losses = []
        for _ in range(6):
            before = state.hp[0][0]
            state, _ = _advance(battle, state, False, Options(), None)
            losses.append(before - state.hp[0][0])
        self.assertEqual(losses[0::2], [0, 0, 800])  # 800 < 100% and < 85% of 1,000 max HP, not < 70%
        for art in ('scorpio', 'aquarius', 'virgo', 'gemini', 'sagittarius', 'taurus', 'cancer'):
            self.compare(self.battle((56, 56), (210, 21), arts={(0, 0): art}), Options(mode='sample', seed=3, max_steps=200))


SUPPORTED_COUNT = len(SUPPORTED)


def _sample_chance(seed):
    from card_engine.simulator.reference import _Chance, _RNG
    return _Chance(rng=_RNG(seed))


def _poisoned(state):
    from dataclasses import replace
    poison = tuple(tuple(row) for row in ((0,), (3,)))
    return replace(state, poison=poison, poison_damage=((0.0,), (1.0,)))


if __name__ == '__main__':
    unittest.main()
