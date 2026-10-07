"""Training progress from a run's log: the latest step, rows and speed, then the last validation rows.

    python3 scripts/perf.py                                   # the pod run, as pod_sync.sh brings it down
    python3 scripts/perf.py data/training/log.jsonl 12        # another log, last 12 evaluations
    python3 scripts/perf.py data/training_18p.out              # a trainer's printed output works too

text A / B: the KL the probe gains with every card vector 0 (A overall, B on upsets): what the model draws
from the description transformer, 0 while it is unused.
up_kl and fav_kl: the KL on the upsets alone and on the other battles; kl is about 9% the first and 91% the
second, so it can fall while the upsets get worse.
val_rows and upsets are the validation set's size and its upset slice: when new shards change them, the
metrics move with the data, not the model.
stat_drift: the stat MLP's progress over the interval, on the run's own scale: it falls while the MLP learns and
flattens at a plateau (the 10s run: 1.01 at step 2,000, about 0.45 from 6,000). With
--language-after-plateau it thaws and freezes the description transformer (thawed / refrozen / done).
"""

import json
import sys
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else "data/training_pod/log.jsonl"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 6
lines = open(path).readlines() if Path(path).exists() else []
# A trainer's .out file (its printed output) works too: only its log records are read.
lines = [l for l in lines if l.startswith('{"step"') and '"loss"' in l]
if not lines:
    raise SystemExit(f"{path} is empty: the trainer is still loading its labels (see its .out file)")
last = json.loads(lines[-1])
print(f"latest step {last['step']:,}  rows {last['train_rows']:,}  {last.get('steps_per_s')} steps/s")
print("step       rows        val_rows  upsets  acc    base   upset  kl     up_kl  fav_kl decisive  prob_err  train_acc  stat_drift")
for r in [json.loads(l) for l in lines[-200 * count:] if '"val"' in l][-count:]:
    v, t = r["val"], r.get("grok", {}).get("train_probe", {})
    print(f"{r['step']:<10,} {r['train_rows']:<11,} {r.get('val_rows', 0):<9,} {int(v.get('upset_rows', 0)):<7,} "
          f"{v['accuracy']:.3f}  {v['baseline']:.3f}  {v['upset_accuracy']:.3f}  {v['kl']:.3f}  "
          f"{v.get('upset_kl', float('nan')):.3f}  {v.get('favourite_kl', float('nan')):.3f}  "
          f"{v['decisive_accuracy']:.3f}     {v['probabilistic_error']:.3f}     {t.get('accuracy', float('nan')):.3f}      {r.get('stat_drift', float('nan')):.2f}"
          + (f"  text {gain['kl']:+.3f} / {gain['upset_kl']:+.3f}" if (gain := r.get("grok", {}).get("language_gain")) else "")
          + "".join(f"  {k[9:]}" for k in ("language_thawed", "language_refrozen", "language_done") if r.get(k)))
for name, title in (("val_fixed", "fixed-stat battles only"), ("val_hard", "hard examples only (generator-proposed)"),
                    ("val_hidden", "incomplete mode only (against the unseen field)"),
                    ("val_found", "found by the annealed search only (hard.py --select engine)")):
    subset = [json.loads(l) for l in lines[-200 * count:] if f'"{name}"' in l][-count:]
    if not subset:
        continue
    print(f"\n{title}")
    print("step       val_rows  upsets  acc    base   upset  kl     decisive  prob_err")
    for r in subset:
        v = r[name]
        print(f"{r['step']:<10,} {r[name + '_rows']:<9,} {int(v.get('upset_rows', 0)):<7,} {v['accuracy']:.3f}  "
              f"{v['baseline']:.3f}  {v['upset_accuracy']:.3f}  {v['kl']:.3f}  {v['decisive_accuracy']:.3f}     "
              f"{v['probabilistic_error']:.3f}")
averaged = [json.loads(l) for l in lines[-200 * count:] if '"ema"' in l][-count:]
if averaged:  # the weight average (ema.checkpoint) beside the live weights, on the same rows
    print("\nlive weights vs their average (ema.checkpoint)")
    print("step       probe_kl  ema    probe_err ema    hard_kl  ema    hard_upset ema")
    for r in averaged:
        live_probe, ema_probe = r.get("grok", {}).get("val_probe", {}), r["ema"].get("val_probe", {})
        live_hard, ema_hard = r.get("val_hard", {}), r["ema"].get("val_hard", {})
        cell = lambda d, k: f"{d[k]:.3f}" if k in d else "  -  "
        print(f"{r['step']:<10,} {cell(live_probe, 'kl')}     {cell(ema_probe, 'kl')}  {cell(live_probe, 'probabilistic_error')}     "
              f"{cell(ema_probe, 'probabilistic_error')}  {cell(live_hard, 'kl')}    {cell(ema_hard, 'kl')}  "
              f"{cell(live_hard, 'upset_accuracy')}      {cell(ema_hard, 'upset_accuracy')}")
watched = [json.loads(l) for l in lines[-200 * count:] if '"watch"' in l][-count:]
if watched:  # the named battles of training/watch.json, one per row: the engine's answer, then live / average per step
    names = [w["name"] for w in watched[-1]["watch"]]
    shown = [r for r in watched if [w["name"] for w in r["watch"]] == names][-4:]  # the current list's evaluations
    print("\nwatch list (ally win chance; engine, then live / average at each step)")
    print(f"{'battle':<32} engine  " + "  ".join(f"{r['step']:>11,}" for r in shown))
    for i, name in enumerate(names):  # residual: the averaged weights' log-odds correction to the stat rule (--stat-prior)
        print(f"{name[:32]:<32} {shown[-1]['watch'][i]['engine']:.2f}    "
              + "  ".join(f"{r['watch'][i]['model']:.2f} / {r['watch'][i]['ema']:.2f}" for r in shown)
              + (f"   residual {shown[-1]['watch_residual'][i]:+.2f}" if "watch_residual" in shown[-1] else ""))
    print(f"{'mean |model - engine|':<32}         " + "  ".join(
        f"{sum(abs(w['model'] - w['engine']) for w in r['watch']) / len(names):.2f} / "
        f"{sum(abs(w['ema'] - w['engine']) for w in r['watch']) / len(names):.2f}" for r in shown))
