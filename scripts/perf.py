"""Training progress from a run's log: the latest step, rows and speed, then the last validation rows.

    python3 scripts/perf.py                                   # the pod run, as pod_sync.sh brings it down
    python3 scripts/perf.py data/training/log.jsonl 12        # another log, last 12 evaluations

val_rows and upsets are the validation set's size and its upset slice: when new shards change them, the
metrics move with the data, not the model.
"""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/training_pod/log.jsonl"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 6
lines = open(path).readlines()
last = json.loads(lines[-1])
print(f"latest step {last['step']:,}  rows {last['train_rows']:,}  {last.get('steps_per_s')} steps/s")
print("step       rows        val_rows  upsets  acc    base   upset  kl     decisive  prob_err  train_acc")
for r in [json.loads(l) for l in lines[-200 * count:] if '"val"' in l][-count:]:
    v, t = r["val"], r.get("grok", {}).get("train_probe", {})
    print(f"{r['step']:<10,} {r['train_rows']:<11,} {r.get('val_rows', 0):<9,} {int(v.get('upset_rows', 0)):<7,} "
          f"{v['accuracy']:.3f}  {v['baseline']:.3f}  {v['upset_accuracy']:.3f}  {v['kl']:.3f}  "
          f"{v['decisive_accuracy']:.3f}     {v['probabilistic_error']:.3f}     {t.get('accuracy', float('nan')):.3f}")
