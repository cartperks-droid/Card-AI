"""ctypes interface to the limited C battle kernel.

Compilation is explicit: call build_library() before loading a new checkout.
Every failure raises; unsupported mechanics are never treated as no-ops.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "sim"
DEFAULT_LIBRARY = SOURCE_DIR / ("libcard_sim.dylib" if platform.system() == "Darwin" else "libcard_sim.so")


class NativeSimulationError(ValueError):
    """Invalid input, unsupported field, or explicit native execution failure."""


_BATCH1_FIELDS = ("action_end_hp_multiplier", "action_start_heal_attack_fraction", "kill_heal_max_hp_fraction", "kill_attack_multiplier", "kill_hp_multiplier", "kill_steal_fraction", "hit_attack_gain_fraction", "hit_target_attack_reduction_fraction", "hit_target_max_hp_reduction_fraction", "hit_target_attack_multiplier", "hit_steal_fraction", "damaged_attack_multiplier", "damaged_stat_multiplier", "damaged_heal_max_hp_fraction", "attacked_steal_fraction", "thorns_attack_fraction", "reflect_damage_fraction", "counter_multiplier", "entry_enemy_hp_multiplier", "entry_enemy_stat_multiplier", "enemy_turn_start_attack_multiplier", "enemy_turn_start_stat_multiplier", "enemy_turn_start_hp_loss_fraction", "enemy_turn_start_max_hp_loss_fraction", "next_card_stat_add_fraction", "next_card_max_hp_add_fraction", "hit_hp_gain_fraction")
_BATCH2_FLOATS = ("charged_attack_multiplier", "alternate_dodge_heal_fraction", "lethal_survival_hp_fraction", "lethal_dodge_heal_max_hp_fraction", "lethal_dodge_attack_multiplier", "first_attack_multiplier", "hit_growth_multiplier", "weak_target_multiplier", "weak_target_threshold", "execute_below_max_hp", "execute_after_below_max_hp", "mutual_destruction_below_max_hp", "lost_hp_damage_fraction", "max_hp_damage_fraction", "target_attack_damage_fraction", "advantage_damage_multiplier", "disadvantage_incoming_multiplier", "lost_hp_damage_bonus", "lost_hp_lifesteal", "self_turn_start_hp_loss_fraction", "self_turn_start_max_hp_loss_fraction", "entry_fallen_attack_fraction", "entry_fallen_stat_bonus", "entry_party_attack_bonus", "low_hp_incoming_multiplier", "low_hp_incoming_threshold", "action_end_stat_multiplier", "periodic_stat_multiplier", "action_end_attack_add_base")
_BATCH2_INTS = ("max_hp_damage_costs", "recharge_turns", "first_turn_actions", "actions_growth_per_turn", "alternate_dodge", "restore_lowered_attack", "own_lethal_dodges", "bypass_defenses", "entry_bypass_defenses", "entry_hit_requires_fallen", "action_end_stat_turns", "periodic_stat_period")
_BATCH2_POSITIVE = {'lethal_dodge_attack_multiplier', 'action_end_stat_multiplier', 'periodic_stat_multiplier'}
_BATCH3_INTS = ("burn_on_attacked_turns", "bleed_on_attacked_turns", "hit_burn_turns", "hit_bleed_turns", "entry_burn_turns", "entry_freeze_turns", "entry_slow_turns")
_BATCH4_FLOATS = ("frostbite_incoming_multiplier", "hit_freeze_chance", "skip_enemy_turn_chance", "chance_survival_probability", "chance_survival_hp_fraction", "gamble_chance", "gamble_multiplier", "gamble_fail_heal_fraction", "dodge_attack_multiplier", "dodge_bonus_multiplier", "dodge_growth", "dodge_cap", "entry_borderless_reset_chance", "borderless_hp", "borderless_attack", "guardian_chance", "confusion_fail_attack_ratio")
_BATCH4_INTS = ("hit_frostbite_turns", "death_frostbite_turns", "entry_random_debuff_turns", "hit_freeze_turns", "entry_confusion_turns", "random_multiplier_max", "ratio_kill", "hit_chance_squared", "chance_survival_freeze_turns")
_BATCH4_ONES = {"frostbite_incoming_multiplier", "hit_freeze_chance", "gamble_multiplier",
                 "dodge_attack_multiplier", "dodge_bonus_multiplier", "dodge_cap"}
_BATCH5_FLOATS = ("fade_threshold", "fade_outgoing_multiplier", "fade_incoming_multiplier", "fade_stat_multiplier", "survival_damage_max_hp_fraction", "entry_next_card_stat_multiplier", "next_card_stat_multiplier", "next_card_gift_chance", "entry_enemy_stat_subtract_fraction", "entry_steal_attack_fraction", "entry_steal_fraction", "turn_start_hp_steal_fraction", "turn_start_attack_chance", "turn_start_attack_multiplier", "turn_start_full_heal_chance", "damaged_attack_loss_fraction", "target_current_hp_damage_fraction", "action_end_self_hp_loss_fraction", "debuff_attacker_multiplier", "first_hit_per_enemy_multiplier", "field_damage_multiplier", "action_end_ally_stat_multiplier", "kill_ally_stat_multiplier", "normal_attack_dodge_cost", "action_end_incoming_multiplier", "card_rarity", "border_rarity", "border_advantage_multiplier", "border_advantage_incoming_multiplier", "lower_rarity_incoming_multiplier", "younger_target_multiplier", "class_damage_multiplier", "class_incoming_multiplier", "stored_damage_to_next_attack_fraction", "damage_taken_to_party_attack_fraction")
_BATCH5_INTS = ("alternate_rest", "periodic_block_period", "fade_extra_actions", "kill_extra_action", "never_attacks", "debuff_attacker_count", "next_card_freeze_turns", "class_mask", "pack_index", "bonus_vs_class_mask", "guard_vs_class_mask")
_BATCH5_POSITIVE = {'fade_incoming_multiplier', 'turn_start_attack_multiplier', 'action_end_ally_stat_multiplier', 'action_end_incoming_multiplier', 'field_damage_multiplier', 'fade_outgoing_multiplier', 'next_card_stat_multiplier', 'entry_next_card_stat_multiplier', 'kill_ally_stat_multiplier', 'fade_stat_multiplier', 'debuff_attacker_multiplier'}
_BATCH5_CHANCES = {'turn_start_attack_chance', 'next_card_gift_chance', 'turn_start_full_heal_chance'}
_BATCH7_FLOATS = ("death_damage_max_hp_fraction", "death_damage_chance", "death_ally_stat_multiplier", "periodic_attack_multiplier",
                  "periodic_attack_heal_max_hp_fraction", "followup_multiplier", "frozen_target_multiplier", "low_hp_actions_threshold",
                  "poison_fraction", "hit_weakness_multiplier", "low_hp_heal_fraction", "low_hp_trigger_threshold",
                  "poison_target_max_hp_fraction", "delayed_strike_multiplier", "bench_growth_multiplier", "bench_growth_cap",
                  "block_max_hp_fraction", "entry_fossil_attack_multiplier", "turn_start_steal_fraction",
                  "hit_attack_drain_fraction", "turn_start_stat_growth", "turn_start_incoming_reduction",
                  "entry_steal_attack_chance", "incoming_reduction_cap")
_BATCH7_ONES = {"entry_fossil_attack_multiplier", "hit_weakness_multiplier", "bench_growth_multiplier", "poison_fraction", "death_damage_chance", "death_ally_stat_multiplier", "periodic_attack_multiplier", "frozen_target_multiplier"}
_BATCH7_INTS = ("periodic_attack_period", "periodic_attack_hits", "periodic_enemy_slow_turns", "followup_bypass", "followup_on_critical",
                "encounter_freeze_turns", "encounter_confusion_turns", "encounter_class_mask", "low_hp_extra_actions",
                "entry_poison_turns", "entry_poison_all", "hit_poison_turns", "attacked_poison_turns", "lethal_dodge_swap",
                "zero_hp_survival_turns", "encounter_doom_turns", "delayed_strike_turns", "entry_blind", "low_hp_stun_turns",
                "stat_gamble", "block_max", "poison_permanent", "action_end_self_slow_turns",
                "dinosaur", "death_fossils", "counter_min_fossils", "fossil_death_timer", "fossil_deck_hits", "entry_self_slow_turns",
                "entry_disable_enemy", "hit_disable", "cancels_all_abilities", "entry_suppress_enemy_turns", "steal_first_attacked",
                "kill_steal_ability", "copy_fallen_ally", "disable_enemy_class_mask", "entry_disable_enemy_class_mask",
                "copies_enemy_in_play", "entry_hit_next_card")
IDENTITY_FIELDS = ("hp", "attack", "borderless_hp", "borderless_attack", "card_rarity", "border_rarity", "pack_index",
                   "class_mask", "dinosaur", "card_id", "spare")
from .reference import _BATCH12_FLOATS, _BATCH12_INTS, _BATCH12_ONES, _BATCH12_WIDE, MAX_FIGHTERS  # noqa: E402
_BATCH1_ONES = {name for name in _BATCH1_FIELDS if "multiplier" in name}
_BATCH1_POSITIVE = {"action_end_hp_multiplier", "kill_hp_multiplier", "damaged_stat_multiplier",
                    "entry_enemy_hp_multiplier", "entry_enemy_stat_multiplier", "enemy_turn_start_stat_multiplier"}


class CFighter(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        "hp", "attack", "dodge_probability", "outgoing_multiplier", "incoming_multiplier",
        "entry_self_multiplier", "entry_enemy_attack_multiplier", "enemy_entry_steal_fraction",
        "entry_hp_multiplier", "entry_attack_multiplier", "damage_reduction_max_hp", "damage_cap_max_hp",
        "dodge_below_max_hp", "nullify_below_attack", "critical_probability", "critical_multiplier", "entry_hit_multiplier",
        "action_end_attack_multiplier", "action_start_heal_max_hp_fraction", "action_end_heal_max_hp_fraction",
        "heal_damage_dealt_fraction", "heal_damage_taken_fraction", "after_attack_heal_max_hp_fraction",
        "intercept_lethal_multiplier", *_BATCH1_FIELDS, *_BATCH2_FLOATS, "survival_attack_multiplier", *_BATCH4_FLOATS, *_BATCH5_FLOATS, "splash_damage_fraction", "recoil_damage_fraction", *_BATCH7_FLOATS, *_BATCH12_FLOATS)] + [
            ("block_mode", ctypes.c_int32), ("attacks_per_action", ctypes.c_int32), ("heal_on_kill", ctypes.c_int32),
            ("counter_on_damage", ctypes.c_int32), ("lethal_survivals", ctypes.c_int32),
            ("invincible", ctypes.c_int32), ("lifetime_actions", ctypes.c_int32), ("actions_per_turn", ctypes.c_int32),
            ("next_card_dodges", ctypes.c_int32)] + [(name, ctypes.c_int32) for name in _BATCH2_INTS + _BATCH3_INTS + _BATCH4_INTS + _BATCH5_INTS + ("entry_hit_all_enemies", "death_reflects") + _BATCH7_INTS + _BATCH12_INTS]


class CBattle(ctypes.Structure):
    _fields_ = [("fighters", (CFighter * MAX_FIGHTERS) * 2), ("counts", ctypes.c_uint32 * 2), ("first_side", ctypes.c_uint32),
                ("blanks", (CFighter * MAX_FIGHTERS) * 2), ("lineup", ctypes.c_uint32 * 2), ("pool_count", ctypes.c_uint32),
                ("pool", ctypes.POINTER(CFighter)), ("defaults", CFighter)]


class COptions(ctypes.Structure):
    _fields_ = [("mode", ctypes.c_int32), ("rounding", ctypes.c_int32), ("stat_rounding", ctypes.c_int32),
                ("repeat_cycles", ctypes.c_uint32), ("max_steps", ctypes.c_uint32),
                ("max_frontier", ctypes.c_uint32), ("prune_probability", ctypes.c_double),
                ("seed", ctypes.c_uint64), ("rollouts", ctypes.c_uint32), ("rollout_error", ctypes.c_double),
                ("node_budget", ctypes.c_uint32), ("sample_below", ctypes.c_double)]


class CResult(ctypes.Structure):
    _fields_ = [(key, ctypes.c_double) for key in ("p_a", "p_b", "tie", "unresolved")] + [
        ("expanded_states", ctypes.c_uint64), ("merged_states", ctypes.c_uint64), ("estimated", ctypes.c_double)]


def build_library(output_path=None, *, compiler=None):
    """Compile C99 source with the local C compiler, returning its absolute path."""
    path = Path(output_path or DEFAULT_LIBRARY).resolve()
    executable = compiler or os.environ.get("CC") or shutil.which("cc")
    if not executable:
        raise NativeSimulationError("No C compiler found; install a compiler or supply compiler=...")
    path.parent.mkdir(parents=True, exist_ok=True)
    # -ffp-contract=off: fused multiply-add skips an intermediate rounding and breaks
    # bit-for-bit parity with the Python reference (seen with 160 - 320 * 0.35).
    command = [executable, "-std=c99", "-O3", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-fPIC",
               "-dynamiclib" if platform.system() == "Darwin" else "-shared",
               str(SOURCE_DIR / "card_sim.c"), "-lm", "-o", str(path)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeSimulationError(f"C kernel build failed: {getattr(exc, 'stderr', '') or str(exc)}") from exc
    return path


def _integer(value, label, minimum=0, maximum=(1 << 32) - 1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise NativeSimulationError(f"{label}: expected integer in [{minimum}, {maximum}]")
    return value


def _number(value, label, minimum=0, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeSimulationError(f"{label}: expected finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise NativeSimulationError(f"{label}: number exceeds double range") from exc
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum) or (positive and value == 0):
        raise NativeSimulationError(f"{label}: number outside permitted range")
    return 0.0 if value == 0 else value


def _fighter(data):
    allowed = {field for field, _ in CFighter._fields_}
    if not isinstance(data, dict) or set(data) - allowed:
        raise NativeSimulationError("Fighter must be a dict containing only supported kernel fields")
    if "hp" not in data or "attack" not in data:
        raise NativeSimulationError("Fighter requires hp and attack")
    f = CFighter()
    f.hp = _number(data["hp"], "hp", positive=True)
    f.attack = _number(data["attack"], "attack")
    f.dodge_probability = _number(data.get("dodge_probability", 0), "dodge_probability", maximum=1)
    for key in ("outgoing_multiplier", "incoming_multiplier", "entry_enemy_attack_multiplier", "action_end_attack_multiplier"):
        setattr(f, key, _number(data.get(key, 1), key))
    f.entry_self_multiplier = _number(data.get("entry_self_multiplier", 1), "entry_self_multiplier", positive=True)
    f.entry_hp_multiplier = _number(data.get("entry_hp_multiplier", 1), "entry_hp_multiplier", positive=True)
    f.entry_attack_multiplier = _number(data.get("entry_attack_multiplier", 1), "entry_attack_multiplier")
    for key in ("damage_reduction_max_hp", "damage_cap_max_hp", "dodge_below_max_hp", "nullify_below_attack", "entry_hit_multiplier", "action_start_heal_max_hp_fraction",
                "action_end_heal_max_hp_fraction", "heal_damage_dealt_fraction", "heal_damage_taken_fraction",
                "after_attack_heal_max_hp_fraction", "intercept_lethal_multiplier"):
        setattr(f, key, _number(data.get(key, 0), key))
    for key in _BATCH2_FLOATS:
        setattr(f, key, _number(data.get(key, 1 if "multiplier" in key else 0), key, positive=key in _BATCH2_POSITIVE))
    for key in _BATCH2_INTS + _BATCH3_INTS + _BATCH4_INTS:
        setattr(f, key, _integer(data.get(key, 0), key, maximum=16))
    f.splash_damage_fraction = _number(data.get("splash_damage_fraction", 0), "splash_damage_fraction")
    f.recoil_damage_fraction = _number(data.get("recoil_damage_fraction", 0), "recoil_damage_fraction")
    f.entry_hit_all_enemies = _integer(data.get("entry_hit_all_enemies", 0), "entry_hit_all_enemies", maximum=1)
    f.death_reflects = _integer(data.get("death_reflects", 0), "death_reflects", maximum=16)
    for key in _BATCH7_FLOATS:
        setattr(f, key, _number(data.get(key, 1 if key in _BATCH7_ONES else 0), key,
                                maximum=1 if key == "death_damage_chance" else None, positive=key == "death_ally_stat_multiplier"))
    for key in _BATCH7_INTS:
        setattr(f, key, _integer(data.get(key, 0), key, maximum=127 if key.endswith("_mask") else 16))
    for key in _BATCH5_INTS:
        setattr(f, key, _integer(data.get(key, 0), key, maximum=127 if key.endswith("_mask") or key == "pack_index" else 16))
    for key in _BATCH5_FLOATS:
        default = 1 if "multiplier" in key or key in ("next_card_gift_chance", "border_rarity") else 0
        setattr(f, key, _number(data.get(key, default), key, positive=key in _BATCH5_POSITIVE,
                                maximum=1 if key in _BATCH5_CHANCES else None))
    for key in _BATCH4_FLOATS:
        chance = key in ("hit_freeze_chance", "skip_enemy_turn_chance", "chance_survival_probability", "gamble_chance",
                         "entry_borderless_reset_chance", "guardian_chance", "dodge_cap")
        setattr(f, key, _number(data.get(key, 1 if key in _BATCH4_ONES else 0), key, maximum=1 if chance else None))
    f.survival_attack_multiplier = _number(data.get("survival_attack_multiplier", 1), "survival_attack_multiplier", positive=True)
    for key in _BATCH1_FIELDS:
        setattr(f, key, _number(data.get(key, 1 if key in _BATCH1_ONES else 0), key, positive=key in _BATCH1_POSITIVE))
    f.enemy_entry_steal_fraction = _number(data.get("enemy_entry_steal_fraction", 0), "enemy_entry_steal_fraction", maximum=1)
    f.block_mode = _integer(data.get("block_mode", 0), "block_mode", maximum=2)
    f.attacks_per_action = _integer(data.get("attacks_per_action", 1), "attacks_per_action", minimum=1, maximum=16)
    f.critical_probability = _number(data.get("critical_probability", 0), "critical_probability", maximum=1)
    f.critical_multiplier = _number(data.get("critical_multiplier", 2), "critical_multiplier", minimum=1)
    f.heal_on_kill = _integer(data.get("heal_on_kill", 0), "heal_on_kill", maximum=1)
    f.counter_on_damage = _integer(data.get("counter_on_damage", 0), "counter_on_damage", maximum=1)
    f.lethal_survivals = _integer(data.get("lethal_survivals", 0), "lethal_survivals", maximum=16)
    f.invincible = _integer(data.get("invincible", 0), "invincible", maximum=1)
    f.lifetime_actions = _integer(data.get("lifetime_actions", 0), "lifetime_actions", maximum=(1 << 31) - 1)
    f.actions_per_turn = _integer(data.get("actions_per_turn", 1), "actions_per_turn", minimum=1, maximum=16)
    f.next_card_dodges = _integer(data.get("next_card_dodges", 0), "next_card_dodges", maximum=16)
    for key in _BATCH12_FLOATS:
        setattr(f, key, _number(data.get(key, 1 if key in _BATCH12_ONES else 0), key, positive=key in _BATCH12_ONES))
    for key in _BATCH12_INTS:
        setattr(f, key, _integer(data.get(key, 0), key, maximum=1023 if key in _BATCH12_WIDE else 127 if key.endswith("_mask") else 16))
    return f


_POOLS = {}


def _pool(pool):
    """One C array per ability pool, shared by every battle that uses it (kept alive in _POOLS)."""
    cached = _POOLS.get(id(pool))
    if cached is not None and cached[0] is pool:
        return cached[1]
    array = (CFighter * max(1, len(pool)))(*[_fighter(_asdict(f)) for f in pool])
    if len(_POOLS) > 8:
        _POOLS.clear()
    _POOLS[id(pool)] = (pool, array)
    return array


def _asdict(fighter):
    if dataclasses.is_dataclass(fighter) and not isinstance(fighter, type):
        return {f.name: getattr(fighter, f.name) for f in dataclasses.fields(fighter)}
    return fighter


def _battle(data):
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        data = {"teams": [[_asdict(f) for f in team] for team in data.teams], "first_side": data.first_side,
                "lineup": data.lineup, "pool": data.pool}
    if not isinstance(data, dict) or set(data) - {"teams", "first_side", "lineup", "pool"}:
        raise NativeSimulationError("Battle requires teams=[team_a, team_b] and explicit first_side")
    if "first_side" not in data:
        raise NativeSimulationError("first_side must be explicit")
    teams = data.get("teams")
    if not isinstance(teams, (list, tuple)) or len(teams) != 2:
        raise NativeSimulationError("Battle must contain exactly two teams")
    battle = CBattle()
    battle.first_side = _integer(data["first_side"], "first_side", maximum=1)
    for side, team in enumerate(teams):
        if not isinstance(team, (list, tuple)) or not 1 <= len(team) <= MAX_FIGHTERS:
            raise NativeSimulationError(f"Each team must contain 1..{MAX_FIGHTERS} fighters")
        battle.counts[side] = len(team)
        for index, fighter in enumerate(team):
            battle.fighters[side][index] = _fighter(fighter)
            # The card with no ability (identity fields only), for removed or cancelled abilities.
            battle.blanks[side][index] = _fighter({k: fighter[k] for k in IDENTITY_FIELDS if k in fighter})
    lineup = data.get("lineup") or tuple(len(team) for team in teams)
    if len(lineup) != 2 or any(not 1 <= _integer(n, "lineup") <= len(team) for n, team in zip(lineup, teams)):
        raise NativeSimulationError("lineup must give 1..len(team) starting cards per side")
    battle.lineup[0], battle.lineup[1] = lineup
    pool = tuple(data.get("pool") or ())
    if pool:
        battle.pool = ctypes.cast(_pool(pool), ctypes.POINTER(CFighter))
        battle.pool_count = len(pool)
    battle.defaults = _fighter({"hp": 1.0, "attack": 0.0})
    return battle


def _options(data):
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        data = dataclasses.asdict(data)
    if data is None:
        data = {}
    allowed = {field for field, _ in COptions._fields_}
    if not isinstance(data, dict) or set(data) - allowed:
        raise NativeSimulationError("Unknown simulation options")
    mode = data.get("mode", "branch")
    rounding = data.get("rounding", "ceil")
    if isinstance(mode, str):
        if mode not in ("sample", "branch"):
            raise NativeSimulationError("mode must be sample or branch")
        mode = {"sample": 0, "branch": 1}[mode]
    if isinstance(rounding, str):
        modes = {"unrounded": 0, "float": 0, "ceil": 1, "floor": 2, "nearest_half_up": 3}
        if rounding not in modes:
            raise NativeSimulationError("Unknown damage rounding mode")
        rounding = modes[rounding]
    result = COptions()
    result.mode = _integer(mode, "mode", maximum=1)
    result.rounding = _integer(rounding, "rounding", maximum=3)
    stat_rounding = data.get("stat_rounding", "unrounded")
    if isinstance(stat_rounding, str):
        policies = {"float": 0, "unrounded": 0, "ceil": 1, "floor": 2, "nearest_half_up": 3}
        if stat_rounding not in policies:
            raise NativeSimulationError("Unknown stat rounding mode")
        stat_rounding = policies[stat_rounding]
    result.stat_rounding = _integer(stat_rounding, "stat_rounding", maximum=3)
    result.repeat_cycles = _integer(data.get("repeat_cycles", 50), "repeat_cycles", minimum=1, maximum=(1 << 31) - 1)
    result.max_steps = _integer(data.get("max_steps", 1000), "max_steps", minimum=1, maximum=(1 << 31) - 1)
    result.max_frontier = _integer(data.get("max_frontier", 10000), "max_frontier", minimum=1, maximum=(1 << 31) - 1)
    result.prune_probability = _number(data.get("prune_probability", 0), "prune_probability", maximum=1)
    result.seed = _integer(data.get("seed", 1), "seed", maximum=(1 << 64) - 1)
    result.rollouts = _integer(data.get("rollouts", 0), "rollouts", maximum=(1 << 31) - 1)
    result.rollout_error = _number(data.get("rollout_error", 0), "rollout_error", maximum=1)
    result.node_budget = _integer(data.get("node_budget", 0), "node_budget", maximum=(1 << 31) - 1)
    result.sample_below = _number(data.get("sample_below", 0), "sample_below", maximum=1)
    return result


def load_library(path=None):
    path = Path(path or DEFAULT_LIBRARY).resolve()
    if not path.is_file():
        raise NativeSimulationError(f"C kernel is not built: {path}; call build_library() explicitly")
    library = ctypes.CDLL(str(path))
    library.ce_simulate_batch.argtypes = [ctypes.POINTER(CBattle), ctypes.c_size_t, ctypes.POINTER(COptions),
                                         ctypes.POINTER(CResult), ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
    library.ce_simulate_batch.restype = ctypes.c_int
    library.ce_version.argtypes = []
    library.ce_version.restype = ctypes.c_char_p
    if library.ce_version() != b"card-engine-kernel-0.55-provisional":
        raise NativeSimulationError("Native ABI version mismatch; rebuild the C library")
    return library


def simulate_batch(battles, options=None, *, library_path=None):
    """One C call; match i has independent RNG seed (seed+i) modulo 2**64."""
    if not isinstance(battles, (list, tuple)):
        raise NativeSimulationError("battles must be a sequence")
    converted = {}  # repeated battles (Monte Carlo labels) are converted once
    for row in battles:
        if id(row) not in converted:
            converted[id(row)] = _battle(row)
    inputs = (CBattle * len(battles))(*[converted[id(row)] for row in battles])
    settings = _options(options)
    outputs = (CResult * len(battles))()
    error = ctypes.create_string_buffer(512)
    library = load_library(library_path)
    code = library.ce_simulate_batch(inputs, len(battles), ctypes.byref(settings), outputs, error, len(error))
    if code:
        raise NativeSimulationError(error.value.decode("utf-8", errors="replace") or f"Native kernel failure {code}")
    return [dict({name: getattr(row, name) for name, _ in CResult._fields_},
                 rules_status="experimental_subset", training_labels_allowed=False) for row in outputs]


def simulate(battle, options=None, *, library_path=None):
    return simulate_batch([battle], options, library_path=library_path)[0]
