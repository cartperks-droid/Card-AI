"""Progress of bench runs: one table per log (evaluations), the best by KL and by upset accuracy, and whether it runs.

    python3 scripts/bench_status.py                                   # every log in data/bench_discrete_mac
    python3 scripts/bench_status.py data/bench_discrete_mac/multibit4_30k.log
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evaluations(path):
    rows = []
    for line in open(path):
        if line.startswith("{") and '"kl"' in line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def running(path):
    out = subprocess.run(["pgrep", "-fl", "learning_rule_bench"], capture_output=True, text=True).stdout
    stem = path.stem
    return any(stem.replace("_", "") in line.replace("_", "").replace("-", "") or f"ck_{stem}" in line for line in out.splitlines())


def last_step(path):
    for line in reversed(open(path).read().splitlines()):
        if "] step " in line:
            return int(line.split("] step ")[1].split()[0])
    return None


def show(path):
    rows = evaluations(path)
    state = "running" if running(path) else "stopped"
    step = last_step(path)
    print(f"\n== {path.name}  ({state}{f', step {step}' if step else ''})")
    if not rows:
        print("   no evaluations yet")
        return
    print(f"   {'step':>6} {'KL':>7} {'upsets':>7} {'determ.':>7} {'prob err':>8} {'train':>7}")
    for r in rows:
        print(f"   {r['step']:>6} {r['kl']:>7.4f} {r['upset_accuracy']:>7.1%} {r['decisive_accuracy']:>7.1%} "
              f"{r['probabilistic_error']:>8.3f} {r.get('train_loss', float('nan')):>7.4f}")
    best_kl = min(rows, key=lambda r: r["kl"])
    best_up = max(rows, key=lambda r: r["upset_accuracy"])
    print(f"   best KL {best_kl['kl']:.4f} at step {best_kl['step']}; best upsets {best_up['upset_accuracy']:.1%} "
          f"at step {best_up['step']}; {rows[-1]['seconds'] / rows[-1]['step']:.2f} s/step")


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "data/bench_discrete_mac").glob("*.log"),
                                                    key=lambda p: p.stat().st_mtime)[-3:]
    for path in paths:
        show(path)


if __name__ == "__main__":
    main()
