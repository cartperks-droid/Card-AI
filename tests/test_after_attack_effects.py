"""After-attack effects against an opponent that never attacks (a Buddha stand-in).

Buddha itself is not mapped yet; the stand-in does what it did in IMG_0306: it never attacked,
kept healing, and outlasted 30 s of Gambler hits (so its HP is set very high). Effects that
happen "after attacking" must follow real attacks only, never skipped turns."""

import tempfile
from pathlib import Path
import unittest

from card_engine.catalog import load_catalog
from card_engine.simulator import native
from card_engine.simulator.catalog_rules import compile_fighter
from card_engine.simulator.reference import Battle, Fighter, Options, simulate, simulate_batch

BUDDHA_STAND_IN = Fighter(10**12, 5241, never_attacks=1, action_start_heal_max_hp_fraction=0.5)


class AfterAttackEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = native.build_library(Path(cls.temp.name) / 'after_attack.dylib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def sequence(self, card_id, border=1, steps=12, seed=3):
        """The card's own turns: A = attack, s = skipped, xF = Gambler draw, B = block pool."""
        battle = Battle(((compile_fighter(self.catalog, card_id, border),), (BUDDHA_STAND_IN,)), 0)
        events = simulate(battle, Options(mode='sample', seed=seed, max_steps=steps), trace=True).trace
        out = []
        for e in events:
            if e.get('side') == 0 and e['phase'] == 'ATTACK_DECLARE':
                out.append('A')
            elif e.get('side') == 0 and e['phase'] == 'SKIP':
                out.append('s')
            elif e['phase'] == 'STAT_GAMBLE':
                out.append(f"x{e['factor']}")
        return battle, out

    def compare(self, battle):
        options = Options(max_steps=40)
        py, c = simulate_batch([battle], options)[0], native.simulate_batch([battle], options, library_path=self.library)[0]
        for key in ('p_a', 'p_b', 'tie', 'unresolved', 'expanded_states', 'merged_states'):
            self.assertAlmostEqual(getattr(py, key), c[key], places=9, msg=key)

    def test_failed_mage_attacks_then_skips(self):  # User: it does skip its next turn.
        battle, seq = self.sequence(96)
        self.assertEqual(seq, ['A', 's'] * 3)
        self.compare(battle)

    def test_cyberdon_charges_before_every_attack(self):
        battle, seq = self.sequence(155)
        self.assertEqual(seq, ['s', 'A'] * 3)
        self.compare(battle)

    def test_gambler_draws_once_per_attack(self):  # IMG_0306: a draw after every attack on Buddha.
        battle, seq = self.sequence(83, border=4)
        self.assertEqual([s for s in seq if s == 'A'], ['A'] * 6)
        self.assertEqual(len([s for s in seq if s.startswith('x')]), 6)
        self.assertTrue(all(s in ('A', 'x0.9', 'x1.15', 'x1.3') for s in seq))
        self.compare(battle)

    def test_steve_fills_three_blocks_without_being_hit(self):
        from card_engine.simulator.reference import _advance, _initial
        battle = Battle(((compile_fighter(self.catalog, 183, 5),), (BUDDHA_STAND_IN,)), 0)
        state = _initial(battle, Options(), None)
        pools = []
        for _ in range(8):
            state, _ = _advance(battle, state, False, Options(), None)
            pools.append(state.block_pool[0][0])
        max_hp = state.max_hp[0][0]
        self.assertAlmostEqual(max(pools), 3 * 0.4 * max_hp)  # Capped at 3 blocks of 40% max HP.
        self.assertEqual(self.sequence(183, border=5)[1], ['A'] * 6)  # Never skips.
        self.compare(battle)


if __name__ == '__main__':
    unittest.main()
