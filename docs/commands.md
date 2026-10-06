# Commands

Every command for running this project on your own, by task. Run them from the project folder (`card_engine`) on the Mac. Use `python3` on the Mac; on a pod it's `python`. Code blocks have no inline comments, so they paste straight into zsh.

## 1. Setup and updates

Get the latest code:
```
git pull origin claude/loving-darwin-3jk4cq
```

Fetch and patch DaddyDrago's engine and build the C engine. Run it after every pull that touches `sim_js/` or `sim_c/`, or after changing `sim_js/ENGINE_COMMIT`:
```
bash sim_js/setup.sh
```

Run all tests:
```
python3 -m unittest $(ls tests/test_*.py | sed 's#/#.#;s#\.py$##')
```

Re-import the card data after editing `card_engine/data_corrections.py` or `card_engine/import_snap.py` (card texts, weathers, borders):
```
python3 -m card_engine.import_snap --raw-dir data/raw --output-dir data/clean
```

## 2. Engine rule changes (labels stay valid)

After any change to the engine's code: **setup first, then declare**. Declaring before the rebuild records a half-built engine and holds back every label (2026-10-06).
```
bash sim_js/setup.sh
python3 -m card_engine.training.flags declare --cards "Supreme Ozzy" --note "what changed"
python3 -m card_engine.training.flags status
```

The declaration can name cards, supports or mutations, or say nothing changed or everything did:
```
python3 -m card_engine.training.flags declare --cards 231 "Shuten-dōji" --note "..."
python3 -m card_engine.training.flags declare --supports red27 blue8 --note "..."
python3 -m card_engine.training.flags declare --supports blue13:5 --note "one tier only"
python3 -m card_engine.training.flags declare --mutations Eclipse --note "..."
python3 -m card_engine.training.flags declare --none --note "refactor, no outcome changes"
python3 -m card_engine.training.flags declare --all --note "everything changed"
```

`status` should read "N entities changed" for the big snapshots, never "held back". Card data changes (stats, abilities in his JSON) are detected on their own; only code changes need a declaration. A change to the support code (`auras.label.ts`) can be declared the same way, naming the supports it affects; undeclared, it drops every battle with a support.

Changes made so far live in `sim_js/codemod.mjs` (TypeScript) and `sim_c/engine.c` (C); both must change together. `tests/test_kernel.py` checks they agree. The rule log is `docs/simulator_questions.md`.

## 3. Making labels

Each loop restarts itself when a run ends or a worker dies. With a trainer running, keep the workers low (2–3): six crashed the Mac for memory on 2026-10-06.

Ordinary random battles:
```
caffeinate -i bash -c 'while true; do python3 -m card_engine.training.labels --shards 1000 --rows 2000 --workers 3; done' >> labels.out 2>&1 &
```

Fixed-stat battles (a third are tower floors). `--prior` is the chance a weaker-side card comes from the stat-ignoring list; `--prior 0` drops it:
```
caffeinate -i bash -c 'while true; do python3 -m card_engine.training.labels --fixed --shards 1000 --rows 2000 --workers 2 --prior 0.5; done' >> fixed.out 2>&1 &
```

Hard examples (annealed engine search, then the model's disagreements). `--checkpoint` should be the model you are training, its averaged weights:
```
caffeinate -i nice -n 19 env OMP_NUM_THREADS=2 bash -c 'while true; do python3 -m card_engine.training.hard --shards 2 --workers 2 --device cpu --prior 0 --checkpoint data/training_18t/ema.checkpoint; done' >> hard.out 2>&1 &
```
Without a model scoring the search (cheaper next to a trainer), keep the engine's winners and random battles from every gap:
```
caffeinate -i nice -n 19 bash -c 'while true; do python3 -m card_engine.training.hard --shards 2 --workers 2 --select engine --prior 0; done' >> hard.out 2>&1 &
```
Options: `--candidates 12000` (battles per enemy; 6000 is half the time and still climbs floor 105), `--rounds 40` (enemies per shard), `--keep 16`, `--generator-every 4` (0: no generator rounds).

Incomplete mode (PvP): build a field generation, then mine labels against it. Rebuild the field with newer weights as the model improves:
```
python3 -m card_engine.training.incomplete field --checkpoint data/training_18t/ema.checkpoint --workers 3
caffeinate -i nice -n 19 bash -c 'while true; do python3 -m card_engine.training.incomplete labels --shards 5 --checkpoint data/training_18t/ema.checkpoint --workers 3; done' >> incomplete.out 2>&1 &
```
Field options: `--size 48`, `--candidates 160`, `--opponents 24`. Label options: `--rounds 40`, `--opponents 16`.

See what is running, and stop loops:
```
ps -eo pid,etime,pcpu,args | grep -E 'training\.(train|labels|hard|incomplete)' | grep -v grep
pkill -f 'while true; do python3 -m card_engine.training.labels'
pkill -INT -f card_engine.training.labels
```
Use `hard` or `incomplete` instead of `labels` to stop those loops. Stop the shell loop first, or it restarts the job.

Count shards (the store is too big for `ls` with a wildcard):
```
ls data/labels/store | sed 's/_.*//' | sort | uniq -c
find data/labels/store -name 'fixed_*' -newermt '-1 hour' | wc -l
```

If a label loop crashes on a battle, trace it with one worker; the last `{"seed": ...}` line before the error is the battle:
```
CARD_ENGINE_TRACE=1 python3 -m card_engine.training.labels --fixed --shards 1 --rows 2000 --workers 1 2> trace.txt; tail -c 1500 trace.txt
```

## 4. Training

The current run (18p: the original 18 layout, stat comparisons in the stat MLP, its own description transformer learned from scratch, the found share). The same command resumes it after a stop:
```
python3 -m card_engine.training.train --run-dir data/training_18p --layers 18 --stat-pairs --batch-size 512 --lr 3e-4 --weight-decay 0.05 --dropout 0.1 --mix-start 0.03 0.42 0.42 0 0.01 --mix-until 60000 --mix 0.03 0.1 0.1 0 0.06 --max-rows 60000000 --eval-rows 100000 --eval-every 2000 2>&1 | tee -a data/training_18p.out
```

Battles found by the annealed search (`hard.py --select engine`, `found_*` shards) get the fifth share; incomplete mode is the fourth (0 until it has labels):
```
--mix-start 0.03 0.42 0.42 0 0.06 --mix 0.03 0.1 0.1 0 0.06
```

Once incomplete-mode labels exist, add their share as a fourth number:
```
--mix-start 0.03 0.45 0.45 0.03 --mix 0.03 0.1 0.1 0.05
```

For the last stretch of a run, decay the learning rate between two steps (pick FIRST and LAST from the current step and speed):
```
--lr-decay FIRST LAST
```

Every trainer option:
- `--run-dir DIR`: where the run lives. A new directory starts a new model.
- New-run settings, saved in the run, ignored when resuming: `--width W` (256 is fast), `--stat-prior`, `--language-width W`, `--language-layers N`, `--stat-hidden H`, `--layers N`, `--stat-tokens`, `--stat-pairs`, `--stat-width S`, `--no-pack-embedding`, `--no-mutation-embedding`, `--language-from CHECKPOINT`.
- Rates: `--lr`, `--language-lr`, `--lr-decay FIRST LAST`, `--lr-floor 0.05`, `--weight-decay`, `--dropout`, `--batch-size`.
- Description encoder: a new run learns its own from scratch. `--freeze-language-at STEP` freezes it from that step (a later step than the current one trains it until then); once frozen, it trains again when the stat MLP plateaus (`stat_drift` at or under `--plateau-drift 2` for `--plateau-evals 2` evaluations; `--plateau-evals 0` keeps it frozen); `--language-from CHECKPOINT` borrows another run's instead, frozen from the start.
- Batch mix: `--mix HARD UPSET FIXED [HIDDEN]`, `--mix-start ...`, `--mix-until STEP`.
- Data: `--max-rows N` (36 GB Mac: 60M at most), `--field-generations 2` (incomplete mode), `--reload-every 1000`, `--pack-labels` (slow pod disks only).
- Evaluation: `--eval-every`, `--eval-rows`, `--watch FILE` (`''` for none), `--ema-decay 0.999`.
- Device: `--device cuda|mps|cpu`, `--bf16` (CUDA GPUs).
- Moving machines: `--init-from CHECKPOINT` (a run directory without trainer state starts from these weights).

Stop a trainer with Ctrl-C. Checkpoints are saved every 1,000 steps, so at most the steps since the last one are lost.

## 5. Watching progress

Tables: general, fixed-stat, hard, incomplete mode, live against averaged weights, and the watch list. The number is how many evaluations to show:
```
python3 scripts/perf.py data/training_18t/log.jsonl 6
```

The raw log, and the trainer's output (loading progress and errors):
```
tail -3 data/training_18t/log.jsonl
tail -5 data/training_18t.out
```

Real speed over one minute:
```
a=$(grep -o '"step": [0-9]*' data/training_18t/log.jsonl | tail -1 | cut -d' ' -f2); sleep 60; b=$(grep -o '"step": [0-9]*' data/training_18t/log.jsonl | tail -1 | cut -d' ' -f2); echo "$(( (b - a) / 60 )) steps/s"
```

Hard examples learned or memorised (training KL against held-out KL):
```
python3 scripts/hard_fit.py data/training_18t/ema.checkpoint
```

Memory (keep "Memory Pressure" green in Activity Monitor):
```
sysctl -n hw.memsize | awk '{print $1/2^30 " GB"}'
top -l 2 -n 0 | tail -12
```

The checkpoints in a run directory:
- `model.checkpoint`: the live weights.
- `ema.checkpoint`: the weight average; the best default for tools.
- `best.checkpoint`: the lowest validation KL.
- `watch_best_ema.checkpoint` and `watch_best.checkpoint`: closest to the engine on the watch list (`card_engine/training/watch.json`).
- `trainer.pt`: the optimizer state, needed to resume.

When the watch list changes, the old list's best weights are kept as `*.previous_list.checkpoint`. To see the step a checkpoint was saved at:
```
python3 -c 'import torch; print(torch.load("data/training_18t/watch_best_ema.checkpoint", map_location="cpu", weights_only=True)["metadata"])'
```

## 6. Using the model

A matchup's win chance in both turn orders. `--simulate` adds the engine's answer:
```
python3 -m card_engine.training.predict --checkpoint data/training_18t/ema.checkpoint --ally "Fate Seamstress" "Judgement Day" Parallax "Robin Hood" --ally-blue Fate --tower 105 Impossible --simulate
python3 -m card_engine.training.predict --checkpoint data/training_18t/ema.checkpoint --ally "Vampire Lord@GaPl" Set "Good Boy@Pl/Storm" Archer --ally-red Stormcaller@Galaxy --enemy "Immortal Witch" Archer "Good Boy" Set --enemy-blue Fate@Crystal --simulate
```
Cards are written `Name[@Border][/Mutation]`, supports `Name[@Tier]`. A fixed-stat enemy: `--enemy-stats HP ATK`.

Generate counters (the engine verifies every team):
```
python3 -m card_engine.training.generate --checkpoint data/training_18t/ema.checkpoint --tower 105 Impossible
python3 -m card_engine.training.generate --checkpoint data/training_18t/ema.checkpoint --enemy "Immortal Witch" Archer "Good Boy" Set --pool own
python3 -m card_engine.training.generate --checkpoint data/training_18t/ema.checkpoint --enemies 8 --pool restricted --role defend
```
Generator options:
- Pools: `--pool own|custom|restricted|all`, `--enemy-pool ...`.
- Masks: `--borders none Pl Cr`, `--mutations None Storm`, `--support-tiers base Crystal`, `--max-rarity 2.5M`, `--no-limited`.
- Role: `--role attack|defend`.
- Search: `--counters 32`, `--restarts`, `--steps`, `--noise-levels`, `--sigma-max 8`, `--temperature`, `--nearest`.
- Engine: `--engine-search` (adds an engine search, slower), `--no-verify`.
- Close the trainer's GPU work first, or add `--device cpu`.

Model against engine on a suite of battles:
```
python3 -m card_engine.training.verify --checkpoint data/training_18t/ema.checkpoint --suite tower --difficulty Impossible --n 200 --show 20
```

## 7. Your collection and the player base

Your deck (`data/my_deck.json`), then the custom pool for someone else's collection:
```
python3 -m card_engine.deck add "Malik The Sovereign" --border RuCrPl
python3 -m card_engine.deck add 70 --border GaPl --count 4 --mutation Storm
python3 -m card_engine.deck remove Malik --border RuCr
python3 -m card_engine.deck add-support "Desmond Of Despair" --tier Platinum
python3 -m card_engine.deck list
python3 -m card_engine.deck --custom copy-deck
python3 -m card_engine.deck --custom add "Vampire Lord" --border GaPl
```

The restricted pool (what the player base can own):
```
python3 -m card_engine.restricted show
python3 -m card_engine.restricted add-card Hera
python3 -m card_engine.restricted remove-card Pandora
python3 -m card_engine.restricted enable-border Malik --border GaRuCrPl
python3 -m card_engine.restricted disable-border Odin --border GaRuCr
python3 -m card_engine.restricted reset-card Odin
```

Ownership likelihood of a team: `python3 -m card_engine.ownership --help` lists its commands (`team`, `show`, `calibrate`, ...).

## 8. A GPU pod (RunPod)

Read `docs/runpod.md` first. Copy the code up once, then keep a sync loop running on the Mac. The loop sends label shards up and brings logs, checkpoints, `trainer.pt` and the pod's own shards down:
```
bash scripts/pod_sync.sh IP PORT code
bash scripts/pod_sync.sh IP PORT
```

On the pod, after copying the code, build the engine, then start the trainer with the GPU options:
```
cd /workspace/card_engine && bash sim_js/setup.sh
python -m card_engine.training.train --run-dir data/training_18t --device cuda --bf16 --batch-size 512 --lr 3e-4 --weight-decay 0.05 --dropout 0.1 --mix 0.03 0.1 0.1 --eval-rows 100000 2>&1 | tee -a data/training_18t.out
```
To continue a Mac run there, copy `data/training_18t/` up first (it holds the checkpoints and `trainer.pt`).

Stop the pod at a set time (UTC), archiving every run to `/workspace` first:
```
cd /workspace/card_engine && nohup setsid scripts/runpod_finish.sh "+6 hours" > /workspace/finish.log 2>&1 < /dev/null &
cat /workspace/finish.log
```

Before the balance runs out, bring everything down from the Mac:
```
rsync -az --progress -e "ssh -p PORT" root@IP:/workspace/card_engine/data/training_18t/ data/training_18t/
rsync -az --ignore-existing --progress -e "ssh -p PORT" --exclude 'partial_*' root@IP:/workspace/card_engine/data/labels/store/ data/labels/store/
```

Pod labelling must use its own seed ranges: `--first-seed 50000000` for battles and fixed-stat battles, `--first-seed 5000000` for hard examples. The Mac stays below 5,000,000 on its own.

## 9. Other tools

- Our engine against DaddyDrago's unmodified code, on fixed-stat battles: `python3 -m card_engine.training.crosscheck --battles 200`.
- Counter-team search on a scenario file: `python3 -m card_engine.training.counter SCENARIO --deck data/my_deck.json --mode own`.
- Class membership report: `python3 -m card_engine.class_report`.
- Daily in-game checks to run: `docs/daily_checks/queue.md`.

## Rules of thumb
- Only one trainer per GPU, and no generator or predict on the GPU while it trains.
- After an engine change: `setup.sh`, then `flags declare`, then `flags status`.
- Every rule needs evidence (a video, a screenshot, a stated fact), logged in `docs/simulator_questions.md`. A model or engine result is a prediction until it's checked in-game.
