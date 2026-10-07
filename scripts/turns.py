"""The watch list's decks in both turn orders: the model against DaddyDrago's engine, attacking first and defending.

    python3 scripts/turns.py data/training_10s/step48k_live.checkpoint
    python3 scripts/turns.py data/training_10s/ema.checkpoint card_engine/training/watch.json

The player always starts in the tower, but the reversed battle tests whether the model learned the matchup or only
the turn order it saw (user, 2026-10-07: at step 48,000, Parallax / Judgement Day / Judgement Day / Robin Hood with
Fate scored 0.54 attacking and 0.01 defending at floor 105; the engine gives 0.316 both ways). The engine's answers
are the same each run (fixed seeds); a deck takes a few seconds to minutes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the project root holds card_engine
from card_engine import tower
from card_engine.catalog import load_catalog
from card_engine.teams import parse_side
from card_engine.training.predict import Classifier, simulate
from card_engine.training.train import WATCH_FILE

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "data/training_10s/model.checkpoint"
watch = Path(sys.argv[2]) if len(sys.argv) > 2 else WATCH_FILE
catalog = load_catalog()
classifier = Classifier(checkpoint)
print(f"checkpoint {checkpoint}, step {classifier.metadata.get('step')}")
print(f"{'battle':<32} {'attacking: model / engine':>27} {'defending: model / engine':>27}")
errors = []
for entry in json.loads(watch.read_text()):
    ally = parse_side(catalog, entry["ally"], entry.get("red"), entry.get("blue"))
    floor, level = entry["tower"]
    enemy = tower.fixed_team(catalog, int(floor))
    enemy.update(borders=[tower.enemy_border(catalog, level)] * 4, mutations=[0] * 4, red=0, red_tier=0, blue=0, blue_tier=0)
    stats = tower.stats(int(floor), tower.difficulty(level))
    model = classifier.ally_win([(ally, enemy)], stats)[0]
    engine, exact = simulate(catalog, ally, enemy, enemy_stats=stats)
    errors.append([abs(float(m) - float(e)) for m, e in zip(model, engine)])
    mark = lambda e, x: f"{e:.3f}{'' if x else '~'}"
    print(f"{entry['name']:<32} {model[0]:>17.3f} / {mark(engine[0], exact[0]):<6} {model[1]:>18.3f} / {mark(engine[1], exact[1]):<6}",
          flush=True)
mean = [sum(column) / len(column) for column in zip(*errors)]
print(f"{'mean |model - engine|':<32} {mean[0]:>26.3f} {mean[1]:>27.3f}")
print("~ marks an engine answer estimated with playouts rather than exact")
