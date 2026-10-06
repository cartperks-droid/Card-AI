"""Train the general win predictor on simulator labels (side A initiates), resumable.

Each step encodes every card description once (289 sequences) and gathers the slot vectors,
so the description encoder trains end to end at a fraction of the per-slot cost. Targets are
the outcome probabilities (exact tablebase results or Monte Carlo frequencies), renormalised
over finished outcomes. The newest label
directory (rule fingerprint) is used; when rules change, training continues on the new labels.
Checkpoints hold the model (model.checkpoint) plus optimizer/step state for exact resumption.
"""

import copy
import functools
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..model import BattleModel, load_model_data
from ..model.config import DescriptionConfig, StrategicConfig
from ..model.checkpoint import load_checkpoint, save_checkpoint
from .labels import FIELDS, BATTLE_FIELDS, GENERATION_SEEDS, SHARD_DIR, battle_arrays, possible_rows, shard_paths

RUN_DIR = Path(__file__).resolve().parents[2] / "data" / "training"
VALIDATION_EVERY = 25  # shard seeds divisible by this are held out


def latest_label_dir(root=SHARD_DIR):
    """The label store (every rules version's shards; rows are filtered by training.flags when loading)."""
    store = Path(root) / "store"
    if not any(store.glob("shard_*.npz")):
        raise FileNotFoundError(f"No label shards in {store}")
    return store


_SHARD_CACHE = {}  # path -> (mtime, rules, validation, rows): each shard is read and checked once per rules version


def cpu_allowance():
    """CPUs this process may use: the container's CPU quota (cgroup v2 cpu.max or v1 cfs quota) when one is set,
    else the cores it may run on. A RunPod pod reported 128 cores but was allowed 13.6 (2026-10-05)."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for quota, period in (("/sys/fs/cgroup/cpu.max", None),
                          ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            values = Path(quota).read_text().split()
            if period is not None:
                values.append(Path(period).read_text().strip())
            if values[0] not in ("max", "-1"):
                return max(1, min(cores, int(float(values[0]) / float(values[1]))))
        except (OSError, ValueError, IndexError):
            continue
    return cores


PARALLEL_LOAD = 2000  # a load with at least this many shards to read uses worker processes
_READER = None  # (catalog, entity hashes, declared changes, rules key, pool cards): set by _reader_init


def _reader_init(context, catalog=None):
    global _READER
    from ..catalog import load_catalog
    from .flags import _pool_cards
    torch.set_num_threads(1)
    catalog = catalog or load_catalog()
    current, changes, rules = context
    _READER = (catalog, current, changes, rules, _pool_cards(catalog))


def _read_shard(item):
    """(path, cache entry) for one shard: its rows still valid under the rules in _READER."""
    from .flags import valid_rows
    path, mtime = item
    catalog, current, changes, rules, pool_cards = _READER
    with np.load(path) as shard:
        arrays = {key: shard[key] for key in (*FIELDS, "probs")}
        arrays.update(battle_arrays(shard, len(arrays["probs"])))
        snapshot_id = str(shard["snapshot"])
    mask = valid_rows(arrays, snapshot_id, current, catalog=catalog, changes=changes, pool_cards=pool_cards)
    finished = arrays["probs"][:, :2].sum(1)  # A win, B win (ties cannot happen; unfinished mass is dropped)
    keep = mask & (finished > 0) & possible_rows(arrays["cards"])
    rows = {key: arrays[key][keep].astype(np.int16) for key in FIELDS}
    rows.update({key: arrays[key][keep] for key in BATTLE_FIELDS})
    rows["target"] = (arrays["probs"][keep, :2] / finished[keep, None]).astype(np.float32)
    # 1: hard examples (the model's disagreements); 2: found by the annealed search (training.hard --select engine)
    rows["hard"] = np.full(int(keep.sum()), 1 if path.name.startswith("hard_") else 2 if path.name.startswith("found_") else 0,
                           dtype=np.int8)
    rows["favourite"] = stat_favourite({key: torch.as_tensor(value) for key, value in rows.items()
                                        if key != "target"}).numpy().astype(np.int8)
    hidden = rows["hidden_side"] >= 0  # one side unseen: no stat favourite, so never an upset
    rows["favourite"][hidden] = rows["target"][hidden].argmax(-1)
    return path, (mtime, rules, int(path.stem.split("_")[1]) % VALIDATION_EVERY == 0, rows)


ROW_KEYS = (*FIELDS, *BATTLE_FIELDS, "target", "hard", "favourite")


def _packs(directory):
    return Path(directory).parent / "packs"


def _read_packs(directory, rules):
    """Restore cache entries from packs written under these rules (see load_split's pack); packs written under other
    rules are deleted."""
    for pack in sorted(_packs(directory).glob("pack_*.npz")):
        with np.load(pack) as data:  # uncompressed: large sequential reads
            if str(data["rules"]) != rules:
                pack.unlink()
                continue
            names, mtimes, validation, counts = data["names"], data["mtimes"], data["validation"], data["counts"]
            arrays = {key: data[key] for key in ROW_KEYS}
        offsets = np.cumsum(counts)[:-1]
        parts = {key: np.split(value, offsets) for key, value in arrays.items()}
        for i, name in enumerate(names):
            path = Path(directory) / str(name)
            if path not in _SHARD_CACHE:
                _SHARD_CACHE[path] = (float(mtimes[i]), rules, bool(validation[i]), {key: parts[key][i] for key in ROW_KEYS})


def _write_pack(directory, rules, entries):
    """One uncompressed file holding these shards' checked rows, read back by _read_packs. Packs written under other
    rules are deleted first; with too little free disk the pack is skipped, never the training (2026-10-05: a new pack
    beside the old one filled the pod's 30 GB disk and stopped the trainer)."""
    import shutil
    folder = _packs(directory)
    folder.mkdir(parents=True, exist_ok=True)
    for old in [*folder.glob("partial_*"), *folder.glob("pack_*.npz")]:
        try:
            with np.load(old) as data:
                keep = old.name.startswith("pack_") and str(data["rules"]) == rules
        except (OSError, ValueError, KeyError):
            keep = False
        if not keep:
            old.unlink(missing_ok=True)
    items = sorted(entries.items())
    size = sum(value.nbytes for _, entry in items for value in entry[3].values())
    free = shutil.disk_usage(folder).free
    if free < size * 1.2 + 2e9:
        print(json.dumps({"pack_skipped": f"needs {size / 1e9:.1f} GB, {free / 1e9:.1f} GB free"}), flush=True)
        return
    started = time.time()
    target = folder / f"pack_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    tmp = target.with_name("partial_" + target.name)
    try:
        _save_pack(tmp, rules, items)
    except OSError as error:  # out of space after all: drop the partial file and train on
        tmp.unlink(missing_ok=True)
        print(json.dumps({"pack_skipped": str(error)}), flush=True)
        return
    os.replace(tmp, target)
    print(json.dumps({"packed_shards": len(items), "seconds": round(time.time() - started)}), flush=True)


def _save_pack(tmp, rules, items):
    with open(tmp, "wb") as handle:
        np.savez(handle, rules=np.array(rules), names=np.array([path.name for path, _ in items]),
                 mtimes=np.array([entry[0] for _, entry in items], dtype=np.float64),
                 validation=np.array([entry[2] for _, entry in items], dtype=bool),
                 counts=np.array([len(entry[3]["target"]) for _, entry in items], dtype=np.int64),
                 **{key: np.concatenate([entry[3][key] for _, entry in items]) for key in ROW_KEYS})


_CAP = {}  # a capped load's share of the store's first shards, and their names, set at the first load


def capped_paths(paths, max_rows):
    """Every hard-example shard, every shard written since the first call (new labels, such as the tower floors'),
    and a share of the others, picked by a hash of their names (so validation keeps its share) and sized so the
    rows come to about max_rows (2,000 per shard, 1,600 per hard shard). The share is fixed at the first call, so a
    reload only adds shards, never swaps them (2026-10-06: loading the pod's and the Mac's 165M rows crashed the
    Mac)."""
    hard = [path for path in paths if path.name.startswith(("hard_", "found_"))]
    rest = [path for path in paths if not path.name.startswith(("hard_", "found_"))]
    if not _CAP:
        _CAP.update(share=min(1.0, max(0, max_rows - 1600 * len(hard)) / max(1, 2000 * len(rest))),
                    first={path.name for path in rest})
    kept = [path for path in rest if path.name not in _CAP["first"]
            or int(hashlib.md5(path.name.encode()).hexdigest()[:8], 16) < _CAP["share"] * 2 ** 32]
    return sorted(hard + kept)


def sample_validation(rows, limit):
    """At most `limit` validation rows to score (a fixed draw per validation size), every hard example among them:
    scoring 830k rows through 18 layers on the Mac took longer than 1,000 training steps (2026-10-06)."""
    count = rows["target"].shape[0]
    if limit is None or count <= limit:
        return rows
    draw = torch.randperm(count, generator=torch.Generator().manual_seed(count))[:limit].to(rows["target"].device)
    keep = torch.zeros(count, dtype=torch.bool, device=draw.device)
    keep[draw] = True
    keep |= rows["hard"] > 0
    return {k: v[keep] for k, v in rows.items()}


def load_split(directory, device, release=None, pack=False, max_rows=None, generations=2):
    """(train, validation) tensors of the rows still valid under the current rules (training.flags);
    validation = shards whose seed is divisible by VALIDATION_EVERY. Every row carries `hard` (1 for training.hard's
    hard-example battles); training draws a set share of each batch from them (train's mix).

    Every row also carries `favourite`, the stat favourite (stat_favourite): upsets are rows it did not win.
    Shards are read and checked once, then cached: a reload reads only new shards (by name: a shard file is never
    rewritten), and rechecks the others only when the rules change. Card fields stay int16 as stored (a quarter of int64's memory); Inputs widens each batch.
    `release` runs just before the tensors are built, once rows are known to exist: the trainer drops its old tensors
    there, so a reload never holds two copies of the store at once.

    pack: a load that reads at least PARALLEL_LOAD shards also writes them, checked, into one uncompressed file in
    the label root's packs/ folder, and later loads (a restart) start from the packs written under the current rules,
    reading only newer shards. On the pod, reading 76,657 small shard files took about 38 minutes whatever the CPU
    did (2026-10-05): its disk is slow per file, fast for large files.

    max_rows: load about this many rows (capped_paths); the rest of the store stays on disk. generations: the field
    generations whose incomplete-mode shards count (newest_generations).
    """
    from ..catalog import load_catalog
    from .flags import entity_hashes, load_changes
    catalog = load_catalog()
    current, changes = entity_hashes(catalog), load_changes()
    rules = json.dumps([sorted(current.items()), changes])
    if pack and not _SHARD_CACHE:
        _read_packs(directory, rules)
    paths, started = sorted(shard_paths(directory)), time.time()
    paths = newest_generations(paths, generations)
    if max_rows is not None:
        paths = capped_paths(paths, max_rows)
    # A shard is written once under a new name (labels, hard: atomic renames, seeds never reused), so a cached shard
    # under the current rules is not opened or even dated again; on the pod's slow disk dating 76k files every
    # reload left the GPU idle.
    stale = [path for path in paths if (_SHARD_CACHE.get(path) or (None, None))[1] != rules]
    stale = [(path, path.stat().st_mtime) for path in stale]
    # A first load reads tens of thousands of shards: worker processes share it (pod, 2026-10-05: 75,727 shards took
    # about 40 minutes in one process, torch spreading each shard's small calculation over every core). A reload's
    # few new shards are read here. Either way torch uses one thread per shard.
    fresh = {}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        context = (current, changes, rules)
        if len(stale) >= PARALLEL_LOAD:
            workers = max(1, min(32, cpu_allowance() - 2))  # the trainer's own thread keeps a core or two
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_reader_init,
                                     initargs=(context,)) as pool:
                for done, (path, cached) in enumerate(pool.map(_read_shard, stale, chunksize=64), 1):
                    fresh[path] = cached
                    if done % 5000 == 0:
                        print(json.dumps({"loading_shards": done, "of": len(stale), "seconds": round(time.time() - started)}),
                              flush=True)
        else:
            _reader_init(context, catalog)
            fresh = dict(map(_read_shard, stale))
    finally:
        torch.set_num_threads(threads)
    if pack and len(fresh) >= PARALLEL_LOAD:
        _write_pack(directory, rules, fresh)
    seen = {path: fresh.get(path) or _SHARD_CACHE[path] for path in paths}
    _SHARD_CACHE.clear()
    _SHARD_CACHE.update(seen)  # shards that disappeared, or old rules' entries, are dropped
    # A reload under the same rules that only adds shards appends their rows to the tensors already on the device
    # (in spare capacity); anything else rebuilds them. Rebuilding 15 GB every 1,000 steps idled the pod's GPU once
    # it trained at 6.7 steps/s (2026-10-05).
    loaded = _TENSORS.get("paths")
    incremental = (loaded is not None and _TENSORS["rules"] == rules and _TENSORS["device"] == str(device)
                   and loaded <= set(seen))

    def parts(selected):
        split = {"train": [], "val": []}
        for path in selected:
            cached = seen[path]
            if len(cached[3]["target"]):
                split["val" if cached[2] else "train"].append(cached[3])
        return split

    split = parts([path for path in paths if path not in loaded]) if incremental else None
    if incremental and any(  # no spare room left: rebuild with room, never holding two copies on the device
            _TENSORS.get(name + "_count", 0) + sum(len(part["target"]) for part in chunk)
            > (len(_TENSORS[name]["target"]) if name in _TENSORS else 0) for name, chunk in split.items() if chunk):
        incremental = False
    if not incremental:
        split = parts(paths)
        _TENSORS.clear()
        if release is not None and split["train"]:
            release()
        if not split["train"]:
            return None, (_stack(split["val"], device) if split["val"] else None)
    for name, chunk in split.items():
        if chunk:
            _append(name, chunk, device)
    _TENSORS.update(paths=set(seen), rules=rules, device=str(device))
    out = {name: ({key: buffer[:_TENSORS[name + "_count"]] for key, buffer in _TENSORS[name].items()}
                  if name in _TENSORS else None) for name in ("train", "val")}
    return out["train"], out["val"]


_TENSORS = {}  # the device tensors load_split last returned: buffers with spare room, their row counts, paths, rules


def _stack(parts, device):
    return {key: torch.as_tensor(np.concatenate([part[key] for part in parts]), device=device) for key in parts[0]}


def _append(name, parts, device):
    """Rows of these shards added to _TENSORS[name] (load_split checks they fit, else it rebuilds)."""
    rows = sum(len(part["target"]) for part in parts)
    count = _TENSORS.get(name + "_count", 0)
    buffers = _TENSORS.get(name)
    if buffers is None:  # a rebuild: a quarter spare, so reloads append for a while
        capacity = int(rows * 1.25) + 1
        _TENSORS[name] = buffers = {key: torch.empty((capacity, *parts[0][key].shape[1:]),
                                                     dtype=torch.as_tensor(parts[0][key][:0]).dtype, device=device)
                                    for key in ROW_KEYS}
    for key in ROW_KEYS:
        buffers[key][count:count + rows] = torch.as_tensor(np.concatenate([part[key] for part in parts]), device=device)
    _TENSORS[name + "_count"] = count + rows


def stat_favourite(rows):
    """Per row: the side (0/1) with more sqrt(HP x ATK) after supports, the naive baseline (ties go to A)."""
    stats = card_stats(rows, card_stat_tables())  # float64 on the CPU, so near-ties resolve as the simulator's stats
    strength = (stats[..., 0] * stats[..., 1]).sqrt().sum(-1)
    return (strength[:, 1] > strength[:, 0]).long()


JURASSIC_WORLD = 13  # blue: Prehistoric cards gain value% stats per Prehistoric card on the team


@functools.cache
def card_stat_tables():
    """(base, red, prehistoric, jurassic, hp multiplier) for the model's card stats, from DaddyDrago's engine
    (drago.stat_tables, drago.hp_multipliers; the last for tower floors).

    base [card, border, mutation, 2]: the card's (HP, ATK); red [card, mutation, support, tier, 2]: the red support's
    multiplier on that card (support 0 = none); prehistoric [card]: Prehistoric pack membership; jurassic [tier]:
    Jurassic World's fraction per Prehistoric card. Stat effects of abilities (entry multipliers, deck passives such
    as General Moon Zoo, awakened Toys) are left to the model, like every other ability.
    """
    from ..catalog import load_catalog
    from ..simulator.drago import hp_multipliers, stat_tables
    catalog = load_catalog()
    return tuple(torch.as_tensor(table, dtype=torch.float64) for table in (*stat_tables(catalog), hp_multipliers(catalog)))


def card_stats(rows, tables):
    """[N, 2, 4, 2]: each card's (HP, ATK) as it enters the battle, from label rows and card_stat_tables(). A
    fixed-stat battle (fixed_side 0 or 1) starts that side's four cards at fixed_stats instead, their HP times each
    card's HP multiplier where fixed_hp_mult is set (tower floors on Normal and Impossible)."""
    base, red, prehistoric, jurassic, hp_multiplier = tables
    rows = {key: value.long() if key in FIELDS else value for key, value in rows.items()}
    cards, mutations = rows["cards"], rows["mutations"]
    red = red[cards, mutations, rows["red"][..., None], rows["red_tier"][..., None]]
    member = prehistoric[cards]
    bonus = torch.where(rows["blue"] == JURASSIC_WORLD, jurassic[rows["blue_tier"]], 0.0)
    blue = 1 + member * (bonus * member.sum(-1))[..., None]
    stats = base[cards, rows["borders"], mutations] * red * blue[..., None]
    if "fixed_side" in rows:
        side = rows["fixed_side"].to(stats.device).long()
        fixed = rows["fixed_stats"].to(stats)[:, None, None, :].expand_as(stats)  # [N, 2, 4, 2]
        if "fixed_hp_mult" in rows:
            scale = torch.where(rows["fixed_hp_mult"].to(stats.device).bool()[:, None, None], hp_multiplier[cards], 1.0)
            fixed = torch.stack([fixed[..., 0] * scale, fixed[..., 1]], -1)
        chosen = (side[:, None] == torch.arange(2, device=stats.device))[..., None, None]  # [N, 2, 1, 1]
        stats = torch.where(chosen, fixed, stats)
    return stats


class Inputs:
    """Turns label rows into model keyword inputs."""

    def __init__(self, device):
        self.data = load_model_data(device=device)
        self.device = device
        self.stat_tables = tuple(table.to(device=device, dtype=torch.float32) for table in card_stat_tables())

    def __call__(self, rows, card_table):
        rows = {**{key: rows[key].long() for key in FIELDS},  # stored as int16
                **{key: rows[key] for key in BATTLE_FIELDS if key in rows}}
        cards = rows["cards"]
        index = cards - 1
        identity = self.data.identity_keys[index]
        identity = torch.where(rows["arts"] > 0, self.data.art_identity_keys[rows["arts"]], identity)
        metadata = dict(border_ids=rows["borders"], mutation_ids=rows["mutations"], pack_ids=self.data.pack_ids[index],
                        identity_keys=identity, red_support_ids=rows["red"], blue_support_ids=rows["blue"],
                        support_tiers=torch.stack([rows["red_tier"], rows["blue_tier"]], -1))
        if self.data.class_weights is not None:
            metadata["class_weights"] = self.data.class_weights[index]
        if "hidden_side" in rows:  # incomplete mode: one side unseen, its cards and supports, and MODE says so
            hidden = rows["hidden_side"].long()
            seen = hidden[:, None] != torch.arange(2, device=hidden.device)  # [N, 2]
            metadata.update(card_visible=seen[:, :, None].expand(-1, 2, 4), support_visible=seen[:, :, None].expand(-1, 2, 2),
                            mode_ids=(hidden >= 0).long())
        metadata["card_stats"] = card_stats(rows, self.stat_tables)
        return dict(card_embeddings=card_table[index], **metadata)


def card_table(model, tokens):
    return model.description(tokens)


def evaluate(model, inputs, table, rows, batch_size=4096):
    """Scores rows against a card table computed in eval mode (see `train`'s `eval_table`)."""
    model.eval()
    total = {"loss": 0.0, "accuracy": 0.0, "brier": 0.0, "baseline": 0.0, "upsets": 0.0, "upset_accuracy": 0.0,
             "kl": 0.0, "decisive": 0.0, "decisive_accuracy": 0.0, "probabilistic_error": 0.0}
    count = rows["target"].shape[0]
    with torch.no_grad():
        for start in range(0, count, batch_size):
            part = {k: v[start:start + batch_size] for k, v in rows.items()}
            logits = model(**inputs(part, table))
            target = part["target"]
            total["loss"] += float(-(target * logits.log_softmax(-1)).sum())
            total["accuracy"] += float((logits.argmax(-1) == target.argmax(-1)).sum())
            total["brier"] += float((logits.softmax(-1) - target).square().sum())
            # KL = loss minus the targets' own entropy: the part of the loss the model could still remove.
            total["kl"] += float((target * (target.clamp_min(1e-12).log() - logits.log_softmax(-1))).sum())
            decisive = target.max(-1).values >= 0.999  # deterministic battles: should be right every time
            total["decisive"] += float(decisive.sum())
            total["decisive_accuracy"] += float((decisive & (logits.argmax(-1) == target.argmax(-1))).sum())
            # Probabilistic battles: mean |predicted - estimated| A-win probability (the target is the estimate itself).
            total["probabilistic_error"] += float(((logits.softmax(-1)[:, 0] - target[:, 0]).abs() * (~decisive)).sum())
            winner = target.argmax(-1)
            upset = (part["favourite"] != winner) if "favourite" in part else torch.zeros_like(winner, dtype=torch.bool)  # abilities overturned the stat favourite
            total["baseline"] += float((~upset).sum())
            total["upsets"] += float(upset.sum())
            total["upset_accuracy"] += float((upset & (logits.argmax(-1) == winner)).sum())
    model.train()
    result = {k: v / count for k, v in total.items()
              if k not in ("upsets", "upset_accuracy", "decisive", "decisive_accuracy", "probabilistic_error")}
    result["probabilistic_error"] = total["probabilistic_error"] / max(1.0, count - total["decisive"])
    result["probabilistic_rows"] = count - total["decisive"]
    result["decisive_accuracy"] = total["decisive_accuracy"] / max(1.0, total["decisive"])
    result["upset_accuracy"] = total["upset_accuracy"] / max(1.0, total["upsets"])
    result["upset_rows"] = total["upsets"]
    return result


def set_dropout(model, p):
    """Dropout only at the GELU upscale of each feedforward (user); none on residuals or attention."""
    from ..model.networks import TransformerBlock
    for block in model.modules():
        if isinstance(block, TransformerBlock):
            for layer in block.feedforward:
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = p
            block.residual_dropout.p = 0.0
            block.attention.dropout = 0.0


def freeze_language(model):
    """Stop training the description transformer; its card vectors become fixed."""
    for parameter in model.description.parameters():
        parameter.requires_grad_(False)
    model.description.eval()


def unfreeze_language(model, optimizer):
    """Train the description transformer again, its Adam moments started afresh (none from before the freeze)."""
    for parameter in model.description.parameters():
        parameter.requires_grad_(True)
        optimizer.state.pop(parameter, None)
    model.description.train()


def drift_floor(steps, beta=0.9):
    """The stat MLP's drift (net displacement over path length) when its updates are pure noise: Adam's steps are an
    AR(1) of the gradient noise with coefficient beta, so n of them add to sqrt(n (1 + beta) / (1 - beta)) step
    lengths, against n for steps that all agree."""
    return math.sqrt((1 + beta) / ((1 - beta) * max(1, steps)))


def weight_norm(model):
    return math.sqrt(sum(float(p.detach().square().sum()) for p in model.parameters()))


WATCH_FILE = Path(__file__).with_name("watch.json")


def load_watch(path, device):
    """Named battles scored at every evaluation beside the engine's answer (user, 2026-10-05): DaddyDrago's floor-105
    cheese decks moved 0.08 -> 0.76 -> 0.02 between checks by hand while the summary metrics barely moved. Each entry:
    "ally" (four cards), optional "red"/"blue" supports, and "tower": [floor, difficulty] (its fixed team and stats).
    The ally attacks first. Returns (names, engine win chances, rows on the device)."""
    from .. import tower
    from ..catalog import load_catalog
    from ..teams import parse_side, spec
    from .predict import simulate
    catalog = load_catalog()
    entries = json.loads(Path(path).read_text())
    names, engine, specs, fixed = [], [], [], []
    for entry in entries:
        ally = parse_side(catalog, entry["ally"], entry.get("red"), entry.get("blue"))
        floor, level = entry["tower"]
        enemy = tower.fixed_team(catalog, int(floor))
        enemy.update(borders=[1] * 4, mutations=[0] * 4, red=0, red_tier=0, blue=0, blue_tier=0)
        stats = tower.stats(int(floor), tower.difficulty(level))
        (attack_first, _), _ = simulate(catalog, ally, enemy, enemy_stats=stats)
        names.append(entry["name"]), engine.append(float(attack_first)), specs.append(spec(ally, enemy)), fixed.append(stats)
    rows = {key: torch.tensor([s[key] for s in specs], device=device) for key in FIELDS}
    rows["fixed_side"] = torch.ones(len(specs), dtype=torch.long, device=device)
    rows["fixed_stats"] = torch.tensor([f[:2] for f in fixed], dtype=torch.float32, device=device)
    rows["fixed_hp_mult"] = torch.tensor([int(f[2]) for f in fixed], device=device)
    return names, engine, rows


MIX_KINDS = ("hard", "upset", "fixed", "hidden", "found")


def mix_rows(rows):
    """Indices of the rows each batch takes a set share of (train's mix): hard examples; fixed-stat battles that are
    not hard; upsets (the stat favourite lost) among the rest; incomplete-mode battles (training.incomplete), which
    only their own share draws. A new model learns the first three first (user, 2026-10-04), so "bigger stats win"
    is not learned before the abilities that overturn it."""
    hidden = rows["hidden_side"] >= 0
    found = rows["hard"] == 2  # the annealed search's winners and gap ladder, their own share
    hard, fixed = (rows["hard"] == 1) & ~hidden, (rows["fixed_side"] >= 0) & ~hidden & ~found
    upset = (rows["favourite"].long() != rows["target"].argmax(-1)) & ~hidden & ~found
    return {"hard": hard.nonzero()[:, 0], "fixed": (fixed & ~hard).nonzero()[:, 0],
            "upset": (upset & ~fixed & ~hard).nonzero()[:, 0], "hidden": hidden.nonzero()[:, 0],
            "found": found.nonzero()[:, 0], "is_hidden": hidden}


def newest_generations(paths, keep):
    """The paths without incomplete-mode shards of fields older than the newest `keep` generations: the hidden side
    means the current field of strong teams (user, 2026-10-06), so labels against older fields stop counting."""
    generations = {path: int(path.stem.split("_")[1]) // GENERATION_SEEDS for path in paths if path.name.startswith("hidden_")}
    if not generations:
        return paths
    newest = max(generations.values())
    return [path for path in paths if generations.get(path, newest) > newest - keep]


def mix_shares(step, mix, mix_start=None, mix_until=None):
    """Each kind's share of the batch at a step: mix_start at step 0, moving linearly to mix at mix_until, then held."""
    pad = lambda shares: (*shares, *(0.0,) * (len(MIX_KINDS) - len(shares)))  # three shares: no incomplete mode
    if mix_start is None or mix_until is None or step >= mix_until:
        return pad(tuple(mix))
    t = step / max(1, mix_until)
    return tuple(a + (b - a) * t for a, b in zip(pad(mix_start), pad(mix)))


def train(*, steps=None, batch_size=512, lr=3e-4, warmup=1000, weight_decay=0.05, dropout=0.1, freeze_language_at=None,
          eval_every=1000, init_from=None, language_lr=None, mix=(0.05, 0.0, 0.0), layers=None, architecture=None, language=None, language_from=None,
          mix_start=None, mix_until=None, ema_decay=0.999, pack_labels=False, bf16=False, lr_decay=None, lr_floor=0.05,
          watch=None, max_rows=None, eval_rows=None, field_generations=2, language_after_plateau=False, thaw_min=6000, frozen_min=12000,
          language_budget=60000, plateau_drift=2.0, plateau_evals=2,
          checkpoint_every=1000, reload_every=1000, device=None, run_dir=RUN_DIR, label_root=SHARD_DIR):
    """Train in run_dir, resuming its model and optimizer if both are there. Otherwise init_from (a model checkpoint,
    e.g. one downloaded from another machine) gives the starting weights and step, with a fresh optimizer whose learning
    rate warms up again; with neither, training starts from random weights.

    language_lr: the description transformer's learning rate (default lr), its own parameter group. Its Adam state
    starts fresh whenever the saved optimizer has no such group, so unfreezing it later (a --freeze-language-at past
    the current step) does not resume momentum from before the freeze.

    language_after_plateau (new run; user, 2026-10-06): the description transformer's last layer starts at zero and
    stays frozen, so the strategic transformer runs without card text (every card vector 0) while the stat MLP absorbs
    the stat prior. It then thaws and freezes in turns, by the stat MLP: a frozen spell lasts at least frozen_min
    steps and ends when the MLP has stopped learning; a thawed spell lasts at least thaw_min steps and ends at an
    evaluation where the MLP is learning again. Once its thawed steps reach language_budget in all, it freezes for the
    rest of the run ("language_thawed", "language_refrozen", "language_done" in the log; kept across restarts).
    The stat MLP's drift, its weights' net displacement over each evaluation interval divided by the length of the path
    they took and by that ratio for pure noise (drift_floor), is logged as "stat_drift" (about 1: the MLP only wanders;
    much more: it is still learning). It has stopped learning after plateau_evals intervals in a row at or under
    plateau_drift, and is learning at an interval over it.

    A new run (random weights) takes `layers` strategic layers (default StrategicConfig's) and `architecture`, other
    StrategicConfig fields (stat_tokens, stat_pairs, stat_width, pack_embedding, mutation_embedding, stat_hidden_width),
    and `language`, DescriptionConfig fields (width, layers, heads, feedforward_width) for its own description transformer. language_from: a model
    checkpoint whose description transformer (card text to card vectors) the new run starts from, frozen from the
    first step.

    mix: (hard, upset, fixed) shares of each batch, drawn from those rows (mix_rows); the rest is drawn uniformly
    from all rows. mix_start: the shares at step 0, moving linearly to mix at step mix_until (the curriculum: a new
    model starts on the battles abilities decide, then moves toward the natural mix). No copies are kept.

    lr_decay: (first, last) steps of a cosine decay of the learning rate to lr_floor times its value, held after
    last (user, 2026-10-05: a fixed rate keeps the weights swinging, so the last part of a model's fit never comes).
    Both steps are given, so restarts keep the same schedule.

    bf16: training steps run under bfloat16 autocast (the pod's GPU sat at its power limit in TF32, 2026-10-05);
    the weights, the loss, the weight average and every evaluation stay float32.

    ema_decay: the trainer also keeps an exponential moving average of the weights (user, 2026-10-05), saved as
    ema.checkpoint beside model.checkpoint. At a fixed learning rate the live weights swing between evaluations (the
    cheese decks moved by 0.35 in 1,000 steps); the average moves slowly, so predict, generate and hard should read
    it. Each evaluation scores it on the validation probe and the hard examples ("ema" in the log)."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if str(device).startswith("cuda"):
        # TF32 tensor cores for float32 matrix products (PyTorch keeps them off by default): the pod's 8-layer run
        # did about 5,100 battles/s without them, two-thirds of what the 4090's 4-layer speed predicted.
        torch.set_float32_matmul_precision("high")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path, state_path, log_path = run_dir / "model.checkpoint", run_dir / "trainer.pt", run_dir / "log.jsonl"
    ema_path = run_dir / "ema.checkpoint"
    warm_from = 0  # step the learning-rate warmup counts from (a fresh optimizer warms up again)
    if model_path.exists() and state_path.exists():
        model, _ = load_checkpoint(model_path, map_location=device)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    elif init_from is not None:
        model, metadata = load_checkpoint(init_from, map_location=device)
        state, warm_from = None, int(metadata["step"])
    else:
        strategic = StrategicConfig(**({"layers": layers} if layers else {}), **(architecture or {}))
        description = DescriptionConfig(**(language or {}))
        model, state = BattleModel(description_config=description, strategic_config=strategic).to(device), None
        if language_from is not None:
            source, _ = load_checkpoint(language_from, map_location=device)
            model.description.load_state_dict(source.description.state_dict())
            del source
        if language_after_plateau:  # the strategic transformer runs without card text until the language side starts
            # (user, 2026-10-06): a zero last layer makes every card vector 0, and the language side grows from there
            last = model.description.projection[-1]
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)
    if language_from is not None and freeze_language_at is None:
        freeze_language_at = 0  # the borrowed language side stays as it was trained
    set_dropout(model, dropout)
    model.train()
    # the average starts from the saved one when resuming, else from the starting weights
    ema = load_checkpoint(ema_path, map_location=device)[0] if ema_path.exists() and state is not None else copy.deepcopy(model)
    ema.eval().requires_grad_(False)
    language = list(model.description.parameters())
    ids = {id(p) for p in language}
    rest = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.AdamW([{"params": rest, "base_lr": lr}, {"params": language, "base_lr": language_lr or lr}],
                                  lr=lr, weight_decay=weight_decay)
    step, best, watch_best = warm_from, None, {}
    # the description transformer's thaw/freeze spells (language_after_plateau): frozen now, since step, thawed steps
    # before this spell, done (frozen for good)
    spells = {"frozen": True, "since": step, "thawed": 0, "done": False} if language_after_plateau else None
    if state is not None:
        saved = state["optimizer"]
        if len(saved["param_groups"]) == 2:
            optimizer.load_state_dict(saved)
        else:  # one group from before: keep the rest's moments, start the description transformer's afresh
            single = torch.optim.AdamW(model.parameters(), lr=lr)
            single.load_state_dict(saved)
            for parameter in rest:
                if parameter in single.state:
                    optimizer.state[parameter] = single.state[parameter]
        for group, base in zip(optimizer.param_groups, (lr, language_lr or lr)):
            group["weight_decay"], group["base_lr"] = weight_decay, base
        step, best, warm_from = state["step"], state.get("best"), state.get("warm_from", 0)
        watch_best = dict(state.get("watch_best") or {})
        spells = state.get("language_spells")
        if freeze_language_at is None:  # a restart keeps the run's freeze unless told otherwise (2026-10-05: one
            freeze_language_at = state.get("freeze_language_at")  # that left it out unfroze the deep run's encoder)
    inputs = Inputs(device)
    tokens = inputs.data.description_tokens
    from .flags import snapshot
    label_dir = latest_label_dir(label_root)
    train_rows, val_rows = load_split(label_dir, device, pack=pack_labels, max_rows=max_rows, generations=field_generations)
    if train_rows is None:
        raise SystemExit("No labels are valid under the current rules: see `python -m card_engine.training.flags status`")
    mixed = mix_rows(train_rows)  # recomputed on each reload
    rules_id = snapshot()  # the current rules; probes and the best checkpoint reset when it changes
    watched = load_watch(watch, device) if watch else None
    if watched is not None:  # the best errors only compare on the same battles: a new list starts its own record
        watch_id = json.dumps(list(zip(watched[0], [round(e, 6) for e in watched[1]])))
        if watch_best and watch_best.get("watch_id") != watch_id:
            for name in ("watch_best", "watch_best_ema"):  # the old list's best weights are kept beside the new ones
                if (run_dir / f"{name}.checkpoint").exists():
                    os.replace(run_dir / f"{name}.checkpoint", run_dir / f"{name}.previous_list.checkpoint")
            print(json.dumps({"watch_list_changed": len(watched[0]), "previous_best": watch_best}), flush=True)
            watch_best = {}
        watch_best["watch_id"] = watch_id
    # Grokking probes: fixed subsets of the training and validation rows, so the curves stay comparable
    # (the full validation set grows with new shards and changes with the rules).
    probe_gen = torch.Generator(device="cpu").manual_seed(0)
    def probe(rows, n):
        picks = torch.randperm(rows["target"].shape[0], generator=probe_gen)[:n].to(device)  # one draw for every field
        return {k: v[picks] for k, v in rows.items()}
    train_probe, val_probe = probe(train_rows, 8192), (probe(val_rows, 8192) if val_rows is not None else None)
    probe_labels = rules_id
    log = log_path.open("a")
    frozen_table = None

    def current_table():
        nonlocal frozen_table
        if spells is not None and not spells["frozen"] and spells["thawed"] + step - spells["since"] >= language_budget:
            spells.update(frozen=True, done=True, thawed=language_budget, since=step)  # the budget is spent
            print(json.dumps({"step": step, "language_done": True}), flush=True)
        if (spells["frozen"] if spells is not None else
                freeze_language_at is not None and step >= freeze_language_at):
            if frozen_table is None:
                freeze_language(model)
                with torch.no_grad():
                    frozen_table = card_table(model, tokens)
                print(json.dumps({"step": step, "language_frozen": True}), flush=True)
            return frozen_table
        return card_table(model, tokens)

    def eval_table():
        """Card vectors for evaluation: the frozen table, or one eval-mode pass shared by all three evaluations."""
        if frozen_table is not None:
            return frozen_table
        model.eval()
        with torch.no_grad():
            table = card_table(model, tokens)
        model.train()
        return table
    def release():
        """Drop the loaded tensors in place (a reload's new rows are about to be built), and hand the memory back."""
        for rows in (train_rows, val_rows):
            if rows:
                rows.clear()
        if device == "mps":
            torch.mps.empty_cache()
    # The stat MLP's drift over each evaluation interval (it thaws and freezes the description transformer)
    stat_parameters = [*model.strategy.stat_projection.parameters(), *model.strategy.stat_mlp.parameters()]
    flat = lambda: torch.nn.utils.parameters_to_vector(stat_parameters).detach().clone()
    window_start, previous, path, window_steps, flat_evals = flat(), flat(), torch.zeros((), device=device), 0, 0
    generator = torch.Generator(device="cpu").manual_seed(step)
    started, last = time.time(), time.time()
    losses = []  # this interval's batch losses, kept on the device (no sync per step); logged as their mean
    try:
        while steps is None or step < steps:
            if step and step % reload_every == 0:  # new shards, or new rules
                label_dir = latest_label_dir(label_root)
                fresh = load_split(label_dir, device, release=release, pack=pack_labels, max_rows=max_rows,
                                   generations=field_generations)
                if fresh[0] is None:  # every row held back (an undeclared engine change): keep the rows already loaded
                    print(json.dumps({"step": step, "labels_held_back": "no rows are valid under the current rules; "
                                      "training continues on the loaded rows until the change is declared (flags status)"}),
                          flush=True)
                else:
                    train_rows, val_rows = fresh
                    mixed = mix_rows(train_rows)
                    rules_id = snapshot()
            count = train_rows["target"].shape[0]
            picks = torch.randint(count, (batch_size,), generator=generator).to(device)
            if len(mixed["hidden"]):  # uniform draws that hit incomplete-mode rows are drawn again (they only enter by
                for _ in range(4):    # their own share); the leftovers are covered by the shares below
                    again = mixed["is_hidden"][picks]
                    if not bool(again.any()):
                        break
                    picks[again] = torch.randint(count, (int(again.sum()),), generator=generator).to(device)
            shares, start = mix_shares(step, mix, mix_start, mix_until), 0
            for kind, share in zip(MIX_KINDS, shares):  # set shares of the batch from each kind, no copies kept
                pool, n = mixed[kind], int(round(batch_size * share))
                if n and len(pool):
                    n = min(n, batch_size - start)
                    picks[start:start + n] = pool[torch.randint(len(pool), (n,), generator=generator).to(device)]
                    start += n
            batch = {k: v[picks] for k, v in train_rows.items()}
            scale = min(1.0, (step - warm_from + 1) / warmup)
            if lr_decay is not None:  # cosine from 1 at the first step to lr_floor at the last, then held
                decay_from, decay_to = lr_decay  # not `last`: that is the rate clock (it read 0.0 steps/s)
                progress = min(1.0, max(0.0, (step - decay_from) / max(1, decay_to - decay_from)))
                scale *= lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            table = current_table()  # card vectors in float32, outside autocast (the model checks their dtype)
            with torch.autocast(str(device).split(":")[0], dtype=torch.bfloat16, enabled=bf16):
                logits = model(**inputs(batch, table))
            loss = -(batch["target"] * logits.float().log_softmax(-1)).sum(-1).mean()
            losses.append(loss.detach())
            total = loss + model.strategy.identity_penalty()
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for average, live in zip(ema.parameters(), model.parameters()):
                    average.lerp_(live, 1 - ema_decay)
                current = flat()
                path += (current - previous).norm()
                previous, window_steps = current, window_steps + 1
            step += 1
            if step % 100 == 0:
                now = time.time()
                interval_loss = float(torch.stack(losses).mean())  # one batch's loss swings with its mix of rows
                losses.clear()
                record = {"step": step, "loss": round(interval_loss, 4), "steps_per_s": round(100 / (now - last), 2),
                          "train_rows": count, "labels": rules_id, "lr": float(f"{optimizer.param_groups[0]['lr']:.3g}")}
                if mix_start is not None and mix_until is not None and step <= mix_until:
                    record["mix"] = [round(share, 3) for share in shares]
                last = now
                if spells is not None and spells["done"] and step - spells["since"] <= 100:
                    record["language_done"] = True  # frozen for the rest of the run
                if step % eval_every == 0:
                    drift = float((previous - window_start).norm() / path.clamp_min(1e-12)) / drift_floor(window_steps)
                    record["stat_drift"] = round(drift, 2)
                    flat_evals = flat_evals + 1 if drift <= plateau_drift else 0
                    if spells is not None and not spells["done"]:
                        spell = step - spells["since"]
                        if spells["frozen"] and spell >= frozen_min and flat_evals >= max(1, plateau_evals):
                            unfreeze_language(model, optimizer)  # the stat MLP has stopped: the language side learns
                            frozen_table = None
                            spells.update(frozen=False, since=step)
                            record["language_thawed"] = True
                        elif not spells["frozen"] and spell >= thaw_min and drift > plateau_drift:
                            spells.update(frozen=True, since=step, thawed=spells["thawed"] + spell)  # it learns again:
                            record["language_refrozen"] = True  # the card vectors hold while it absorbs them
                    window_start, path, window_steps = previous.clone(), torch.zeros((), device=device), 0
                if step % eval_every == 0 and val_rows is not None:
                    table = eval_table()
                    sampled = sample_validation(val_rows, eval_rows)
                    complete = sampled["hidden_side"] < 0  # incomplete mode is its own task: scored apart (val_hidden)
                    scored = {k: v[complete] for k, v in sampled.items()}
                    record["val"] = {k: round(v, 4) for k, v in evaluate(model, inputs, table, scored).items()}
                    record["val_rows"] = int(scored["target"].shape[0])
                    for name, source, subset in (("val_fixed", scored, scored["fixed_side"] >= 0),
                                                 ("val_hard", scored, scored["hard"] == 1),
                                                 ("val_found", scored, scored["hard"] == 2),
                                                 ("val_hidden", sampled, ~complete)):
                        if bool(subset.any()):  # fixed-stat, hard and incomplete-mode battles on their own
                            part = {k: v[subset] for k, v in source.items()}
                            record[name] = {k: round(v, 4) for k, v in evaluate(model, inputs, table, part).items()}
                            record[f"{name}_rows"] = int(subset.sum())
                    if rules_id != probe_labels:  # new rules: new probes
                        train_probe, val_probe, probe_labels = probe(train_rows, 8192), probe(val_rows, 8192), rules_id
                    record["grok"] = {"train_probe": {k: round(v, 4) for k, v in evaluate(model, inputs, table, train_probe).items()
                                                      if k in ("kl", "accuracy", "decisive_accuracy")},
                                      "val_probe": {k: round(v, 4) for k, v in evaluate(model, inputs, table, val_probe).items()
                                                    if k in ("kl", "accuracy", "upset_accuracy", "decisive_accuracy", "probabilistic_error")},
                                      "weight_norm": round(weight_norm(model), 2)}
                    with torch.no_grad():
                        ema_table = card_table(ema, tokens)
                    hard = val_rows["hard"] == 1
                    record["ema"] = {}
                    if val_probe is not None:
                        record["ema"]["val_probe"] = {k: round(v, 4) for k, v in evaluate(ema, inputs, ema_table, val_probe).items()
                                                      if k in ("kl", "accuracy", "upset_accuracy", "probabilistic_error")}
                    if bool(hard.any()):
                        part = {k: v[hard] for k, v in val_rows.items()}
                        record["ema"]["val_hard"] = {k: round(v, 4) for k, v in evaluate(ema, inputs, ema_table, part).items()
                                                     if k in ("kl", "accuracy", "upset_accuracy", "probabilistic_error")}
                    ema.eval()  # evaluate() leaves a model in training mode
                    if watched is not None:  # the watch list: engine, live and averaged weights per battle
                        names, engine, rows = watched
                        with torch.no_grad():
                            live = model.eval()(**inputs(rows, table)).float().softmax(-1)[:, 0].tolist()
                            average = ema(**inputs(rows, ema_table)).float().softmax(-1)[:, 0].tolist()
                            residuals = None
                            if ema.strategy.config.stat_prior:  # the averaged weights' correction to the stat rule
                                residuals = ema.strategy.prior_residual(
                                    ema.strategy.build_sequence(**inputs(rows, ema_table))).tolist()
                        model.train()
                        if residuals is not None:
                            record["watch_residual"] = [round(r, 2) for r in residuals]
                        record["watch"] = [{"name": n, "engine": round(e, 3), "model": round(m, 3), "ema": round(a, 3)}
                                           for n, e, m, a in zip(names, engine, live, average)]
                        # The weights closest to the engine on the watch list are kept apart (user, 2026-10-06: the
                        # live weights hit 0.27 on Drago's Fate deck at step 33,000 and were overwritten at 34,000).
                        for weights, predicted, name in ((model, live, "watch_best"), (ema, average, "watch_best_ema")):
                            error = float(np.mean(np.abs(np.array(predicted) - np.array(engine))))
                            if error < watch_best.get(name, float("inf")):
                                watch_best[name] = error
                                save_checkpoint(weights, run_dir / f"{name}.checkpoint",
                                                metadata={"step": step, "labels": rules_id, "watch_error": round(error, 4)})
                                record[name] = round(error, 4)
                    kl = record["val"]["kl"]
                    if best is None or kl < best["kl"] or best.get("labels") != rules_id:
                        best = {"kl": kl, "step": step, "labels": rules_id}
                        save_checkpoint(model, run_dir / "best.checkpoint", metadata={**best, "val": record["val"]})
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            if step % checkpoint_every == 0:
                save_checkpoint(model, model_path, metadata={"step": step, "labels": rules_id,
                                                             "objective": "A initiates; outcome frequencies"})
                save_checkpoint(ema, ema_path, metadata={"step": step, "labels": rules_id, "ema_decay": ema_decay,
                                                         "objective": "A initiates; outcome frequencies"})
                tmp = state_path.with_suffix(".tmp")
                torch.save({"optimizer": optimizer.state_dict(), "step": step, "best": best, "warm_from": warm_from,
                            "freeze_language_at": freeze_language_at, "watch_best": watch_best,
                            "language_spells": spells}, tmp)
                tmp.replace(state_path)
    finally:
        log.close()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1, help="at the GELU upscale only")
    parser.add_argument("--freeze-language-at", type=int, help="step after which the description transformer is frozen; "
                        "a step past the current one unfreezes it until then")
    parser.add_argument("--language-after-plateau", action="store_true",
                        help="new run: the description transformer stays frozen at its fresh initialisation until the "
                        "stat MLP plateaus (it has absorbed the stat prior), then learns")
    parser.add_argument("--thaw-min", type=int, default=6000,
                        help="with --language-after-plateau: least steps of a thawed spell; it refreezes at the first "
                        "evaluation after them where the stat MLP is learning")
    parser.add_argument("--frozen-min", type=int, default=12000,
                        help="least steps of a frozen spell; it thaws at the first evaluation after them where the stat "
                        "MLP has stopped learning")
    parser.add_argument("--language-budget", type=int, default=60000,
                        help="thawed steps in all, after which the description transformer stays frozen")
    parser.add_argument("--plateau-drift", type=float, default=2.0,
                        help="the stat MLP has plateaued when its drift (stat_drift in the log; about 1 is noise) stays "
                        "at or under this for --plateau-evals evaluations")
    parser.add_argument("--plateau-evals", type=int, default=2, help="evaluations in a row")
    parser.add_argument("--language-lr", type=float, help="the description transformer's learning rate (default --lr)")
    parser.add_argument("--mix", type=float, nargs="+", default=(0.05, 0.0, 0.0), metavar="SHARE",
                        help="HARD UPSET FIXED [HIDDEN [FOUND]]: shares of each batch drawn from hard examples "
                        "(training.hard), upsets, fixed-stat battles, incomplete-mode battles (training.incomplete, "
                        "drawn only by their share) and the annealed search's finds (hard.py --select engine); the "
                        "rest uniformly from the others")
    parser.add_argument("--mix-start", type=float, nargs="+", metavar="SHARE",
                        help="the shares at step 0, moving linearly to --mix at --mix-until (a curriculum)")
    parser.add_argument("--field-generations", type=int, default=2,
                        help="newest field generations whose incomplete-mode labels count (training.incomplete)")
    parser.add_argument("--mix-until", type=int, help="step at which the shares reach --mix")
    parser.add_argument("--layers", type=int, help="strategic transformer layers, for a run starting from random weights")
    parser.add_argument("--stat-width", type=int, default=0,
                        help="new run: the stats' own channels of each card token, beside the card's (0: added to it)")
    parser.add_argument("--stat-tokens", action="store_true",
                        help="new run: each card's stats as a token of its own beside the card's")
    parser.add_argument("--stat-pairs", action="store_true",
                        help="new run: the stat MLP compares each card's stats with each of the other 7 cards'")
    parser.add_argument("--stat-prior", action="store_true",
                        help="new run: the stat rule (the larger total sqrt(HP x ATK) wins, log-odds +-2.36) as a "
                        "prior; the network adds a residual in log-odds")
    parser.add_argument("--width", type=int, default=768,
                        help="new run: the token width (user, 2026-10-06: 256 is far faster and makes the 289 cards share "
                        "directions); attention width / 2 (at least 64), feed-forward and stat MLP 4 x width")
    parser.add_argument("--stat-hidden", type=int,
                        help="new run: the stat MLP's hidden width (default 4 x --width)")
    parser.add_argument("--language-width", type=int, default=128,
                        help="new run: the description transformer's width (heads: width / 64, feed-forward: 4 x width)")
    parser.add_argument("--language-layers", type=int, default=4, help="new run: the description transformer's layers")
    parser.add_argument("--no-pack-embedding", action="store_true", help="new run: no card-pack embedding")
    parser.add_argument("--no-mutation-embedding", action="store_true",
                        help="new run: no mutation embedding (mutations still set the card's stats)")
    parser.add_argument("--language-from", help="model checkpoint whose description transformer a new run starts "
                        "from, frozen (default --freeze-language-at 0)")
    parser.add_argument("--watch", default=str(WATCH_FILE),
                        help="named battles scored at every evaluation beside the engine (JSON; '' for none)")
    parser.add_argument("--lr-decay", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="cosine decay of the learning rate from step FIRST to step LAST (then held at --lr-floor)")
    parser.add_argument("--lr-floor", type=float, default=0.05, help="the decayed rate, as a fraction of --lr")
    parser.add_argument("--bf16", action="store_true",
                        help="training steps in bfloat16 (autocast; weights, loss and evaluation stay float32)")
    parser.add_argument("--pack-labels", action="store_true",
                        help="keep the checked labels in large pack files too, so a restart reads few files (for disks "
                        "slow per file, like the pod's)")
    parser.add_argument("--max-rows", type=int, help="load about this many rows: every hard example and a fixed "
                        "random share of the other shards (for a machine whose memory cannot hold the whole store)")
    parser.add_argument("--eval-rows", type=int, help="score at most this many validation rows at each evaluation "
                        "(a random draw, plus every hard example); default all")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="decay of the weight average saved as ema.checkpoint (about 1 / (1 - decay) steps)")
    parser.add_argument("--init-from", help="model checkpoint to start from when the run directory has no trainer state "
                        "(its step is kept; the optimizer starts fresh)")
    parser.add_argument("--device")
    parser.add_argument("--reload-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000,
                        help="steps between evaluations; each scores the whole validation set, so it grows with the data")
    parser.add_argument("--run-dir", default=RUN_DIR, help="checkpoints and log; a new directory starts from random weights")
    parsed = parser.parse_args()
    for shares in (parsed.mix, parsed.mix_start):
        if shares is not None and len(shares) not in (3, 4, 5):
            parser.error("--mix and --mix-start take HARD UPSET FIXED [HIDDEN [FOUND]]")
    train(steps=parsed.steps, batch_size=parsed.batch_size, lr=parsed.lr, device=parsed.device, reload_every=parsed.reload_every,
          weight_decay=parsed.weight_decay, dropout=parsed.dropout, freeze_language_at=parsed.freeze_language_at,
          eval_every=parsed.eval_every, init_from=parsed.init_from, language_lr=parsed.language_lr, mix=parsed.mix,
          layers=parsed.layers, architecture={"stat_width": parsed.stat_width, "stat_tokens": parsed.stat_tokens, "stat_pairs": parsed.stat_pairs,
          "pack_embedding": not parsed.no_pack_embedding, "mutation_embedding": not parsed.no_mutation_embedding,
          "stat_hidden_width": parsed.stat_hidden or 4 * parsed.width, "stat_prior": parsed.stat_prior,
          **({} if parsed.width == 768 else {"width": parsed.width, "attention_width": max(64, parsed.width // 2),
                                             "feedforward_width": 4 * parsed.width})},
          language={"width": parsed.language_width, "layers": parsed.language_layers,
                    "output_width": parsed.width, "projection_hidden_width": parsed.width,
                    "heads": max(1, parsed.language_width // 64) if parsed.language_width > 128 else 4,
                    "feedforward_width": 4 * parsed.language_width},
          language_from=parsed.language_from, mix_start=parsed.mix_start, mix_until=parsed.mix_until, ema_decay=parsed.ema_decay, pack_labels=parsed.pack_labels,
          bf16=parsed.bf16, lr_decay=parsed.lr_decay, lr_floor=parsed.lr_floor, watch=parsed.watch or None, max_rows=parsed.max_rows, eval_rows=parsed.eval_rows,
          field_generations=parsed.field_generations, plateau_drift=parsed.plateau_drift, plateau_evals=parsed.plateau_evals,
          language_after_plateau=parsed.language_after_plateau, thaw_min=parsed.thaw_min,
          frozen_min=parsed.frozen_min, language_budget=parsed.language_budget,
          run_dir=parsed.run_dir)
