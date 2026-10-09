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

Fixed-stat battles (a third are tower floors, a sixth depths floors). `--prior` is the chance a weaker-side card comes from the stat-ignoring list; `--prior 0` drops it:
```
caffeinate -i bash -c 'while true; do python3 -m card_engine.training.labels --fixed --shards 1000 --rows 2000 --workers 2 --prior 0.5; done' >> fixed.out 2>&1 &
```

Hard examples (annealed engine search, then the model's disagreements). `--checkpoint` should be the model you are training, its averaged weights:
```
caffeinate -i nice -n 19 env OMP_NUM_THREADS=2 bash -c 'while true; do python3 -m card_engine.training.hard --shards 2 --workers 2 --device cpu --prior 0 --checkpoint data/training_10s/ema.checkpoint; done' >> hard.out 2>&1 &
```
Without a model scoring the search (cheaper next to a trainer), keep the engine's winners and random battles from every gap:
```
caffeinate -i nice -n 19 bash -c 'while true; do python3 -m card_engine.training.hard --shards 2 --workers 2 --select engine --prior 0; done' >> hard.out 2>&1 &
```
Options: `--candidates 12000` (battles per enemy; 6000 is half the time and still climbs floor 105), `--rounds 40` (enemies per shard), `--keep 16`, `--generator-every 4` (0: no generator rounds). `--tower 105 Impossible`: every enemy on that floor (its fixed team, or random enemies on floors without one) instead of drawn ones. `--depths`: every enemy a depths floor's draw (mostly hard depths).

PvP duels: one defender built blind (incomplete mode) against 32 attackers near its rolls and luck, each countering it from its own pool (complete mode). It writes `pvp_*` (complete rows, hard examples) and `hidden_*` (the defender's mean) shards. Point it at the newest checkpoint as the model improves:
```
caffeinate -i nice -n 19 bash -c 'while true; do python3 -m card_engine.training.incomplete --shards 5 --checkpoint data/training_10s/ema.checkpoint --workers 3; done' >> pvp.out 2>&1 &
```
Options: `--duels 8` (per shard), `--device cpu` beside a trainer.

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

The current run (10s: 10 layers at width 256, stat comparisons in the stat MLP, the smooth stat prior, its own description transformer thawed and frozen by the stat MLP, all five batch shares). The same command resumes it after a stop; new-run settings (layers, width, priors) are read from the run, the rest (rates, shares, data, the language budget) take effect on a restart:
```
python3 -m card_engine.training.train --run-dir data/training_10s --layers 10 --width 256 --stat-pairs --smooth-prior --language-after-plateau --language-budget 120000 --batch-size 512 --lr 3e-4 --lr-decay 330000 370000 --lr-floor 0.33 --weight-decay 0.05 --dropout 0.1 --mix-start 0.03 0.42 0.42 0 0.01 --mix-until 60000 --mix 0.03 0.1 0.1 0.05 0.06 --max-rows 75000000 --eval-rows 100000 --eval-every 2000 >> data/training_10s.out 2>&1 &
```
The learning rate decayed from 3e-4 between steps 330,000 and 370,000 and holds at `--lr-floor` of it since: 0.33 is about 1e-4 (2026-10-07: at 0.1, 3e-5, new hard, PvP and depths rows were absorbed too slowly and the curves went flat). If the averaged weights' probe KL climbs and stays up, lower it to 0.2.

The batch shares are `--mix HARD UPSET FIXED HIDDEN FOUND`: hard examples (`hard_*`, and the PvP duels' `pvp_*`), upsets, fixed-stat battles, incomplete mode (the duels' `hidden_*`, drawn only by its share) and the annealed search's finds (`found_*`). `--mix-start` and `--mix-until` set a curriculum for a new run; past `--mix-until` they do nothing.

Every trainer option:
- `--run-dir DIR`: where the run lives. A new directory starts a new model.
- New-run settings, saved in the run, ignored when resuming: `--width W` (256 is fast), `--stat-prior`, `--smooth-prior` (the prior fitted to the rows at the start), `--language-after-plateau`, `--language-width W`, `--language-layers N`, `--stat-hidden H`, `--layers N`, `--stat-tokens`, `--stat-pairs`, `--stat-width S`, `--no-pack-embedding`, `--no-mutation-embedding`, `--language-from CHECKPOINT`.
- Rates: `--lr`, `--language-lr`, `--lr-decay FIRST LAST`, `--lr-floor 0.05`, `--weight-decay`, `--dropout`, `--batch-size`.
- Description encoder: a new run learns its own from scratch. `--freeze-language-at STEP` freezes it from that step (a later step than the current one trains it until then); `--language-after-plateau` (new run) keeps it frozen, every card vector 0, until the stat MLP plateaus, then thaws and freezes it by the stat MLP (`--frozen-min 12000`, `--thaw-min 6000`; `stat_drift` flat within `--plateau-spread 0.15` over `--plateau-evals 3` readings thaws it, a rise over `--learning-rise 1.5` × that level refreezes it) until `--language-budget 60000` thawed steps, then frozen for good; `--language-from CHECKPOINT` borrows another run's instead, frozen from the start.
- Batch mix: `--mix HARD UPSET FIXED [HIDDEN [FOUND]]`, `--mix-start ...`, `--mix-until STEP`.
- Data: `--max-rows N` (the 10s run: 75M), `--pvp-generations 2` (incomplete mode), `--reload-every 1000`, `--pack-labels` (slow pod disks only).
- Evaluation: `--eval-every`, `--eval-rows`, `--watch FILE` (`''` for none), `--ema-decay 0.999`.
- Device: `--device cuda|mps|cpu`, `--bf16` (CUDA GPUs).
- Moving machines: `--init-from CHECKPOINT` (a run directory without trainer state starts from these weights).

Stop a trainer: Ctrl-C in its terminal, or for one started in the background, find its PID and kill it. Checkpoints are saved every 1,000 steps, so at most the steps since the last one are lost:
```
ps -eo pid,etime,args | grep 'training\.train' | grep -v grep
kill PID
```

## 5. Watching progress

Tables: general, fixed-stat, hard, incomplete mode, live against averaged weights, and the watch list. The number is how many evaluations to show:
```
python3 scripts/perf.py data/training_10s/log.jsonl 6
```

The watch list in both turn orders, the model against the engine (the player always starts in the tower; the reversed order tests whether the model learned the matchup or only the order it saw):
```
python3 scripts/turns.py data/training_10s/ema.checkpoint
```

Tune the generator's settings by the engine: each trial generates counters to tower floors (105, 100, 95 Impossible by default) with one setting, the engine plays the best 8 per floor attacking first, and the trials print best first (trial 0 is the current defaults; `M10000` and `M50000` are the model search, an evolution scored by the model, instead of the ascent). One JSON line per trial goes to `data/tune_generator.jsonl` as it finishes. A large `model_mean` with engine wins near 0 means the setting found teams the model is wrong about, not counters:
```
python3 scripts/tune_generator.py --checkpoint data/training_10s/step298k_ema.checkpoint --trials 24 --workers 2
```
The model search's own settings (population, parents, children, stat-ignoring prior, size), with no ascent trials; the winner's values go to `generate.py --search-population --search-parents --search-children --search-prior --model-search`:
```
python3 scripts/tune_generator.py --checkpoint data/training_10s/report_ema.checkpoint --trials 0 --model-search 50000 --search-trials 12 --workers 2
```

The raw log, and the trainer's output (loading progress and errors):
```
tail -3 data/training_10s/log.jsonl
tail -5 data/training_10s.out
```

Real speed over one minute:
```
a=$(grep -o '"step": [0-9]*' data/training_10s/log.jsonl | tail -1 | cut -d' ' -f2); sleep 60; b=$(grep -o '"step": [0-9]*' data/training_10s/log.jsonl | tail -1 | cut -d' ' -f2); echo "$(( (b - a) / 60 )) steps/s"
```

Hard examples learned or memorised (training KL against held-out KL):
```
python3 scripts/hard_fit.py data/training_10s/ema.checkpoint
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
python3 -c 'import torch; print(torch.load("data/training_10s/watch_best_ema.checkpoint", map_location="cpu", weights_only=True)["metadata"])'
```

## 6. Using the model

A matchup's win chance in both turn orders. `--simulate` adds the engine's answer:
```
python3 -m card_engine.training.predict --checkpoint data/training_10s/ema.checkpoint --ally "Fate Seamstress" "Judgement Day" Parallax "Robin Hood" --ally-blue Fate --tower 105 Impossible --simulate
python3 -m card_engine.training.predict --checkpoint data/training_10s/ema.checkpoint --ally "Vampire Lord@GaPl" Set "Good Boy@Pl/Storm" Archer --ally-red Stormcaller@Galaxy --enemy "Immortal Witch" Archer "Good Boy" Set --enemy-blue Fate@Crystal --simulate
```
Cards are written `Name[@Border][/Mutation]`, supports `Name[@Tier]`. A fixed-stat enemy: `--enemy-stats HP ATK`.

Generate counters (the engine verifies every team). Use the model search (`--model-search`, an evolution scored by the model) and the engine search (`--engine-search`, which keeps improving the model's best teams by real battles); the gradient ascent alone never finds cheese teams (every ascent setting scored 0 at floors 95-105 in the 2026-10-07 tuning). You own no Limited cards, so add `--no-limited`:
```
python3 -m card_engine.training.generate --checkpoint data/training_10s/depths_ema.checkpoint --tower 65 Impossible --pool restricted --no-limited --borders none --support-tiers base --model-search 50000 --engine-search 4000 --counters 8 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); [print(' ', c['model'], c['simulator'], c['cards'], c['red'], c['blue']) for c in d['counters'][:8]]"
python3 -m card_engine.training.generate --checkpoint data/training_10s/depths_ema.checkpoint --enemy "Immortal Witch" Archer "Good Boy" Set --pool own --model-search 50000
python3 -m card_engine.training.generate --checkpoint data/training_10s/depths_ema.checkpoint --enemies 8 --pool restricted --no-limited --role defend --model-search 50000
```
Each printed line is the model's win chance, the engine's, the four cards, and the two supports. Your real borders and support tiers matter more than any search setting: with your deck entered use `--pool own`, otherwise widen the masks (`--borders none Pl Cr --support-tiers base Platinum Crystal`).

If the engine column stays at 0.0, the floor may have no win within those masks. The annealed miner discovers wins best: it starts from random teams at a fraction of the enemy's stats and raises them as its best teams start winning (how floor 105's wins were found). It writes the battles as training data; in its log, `mean_best_engine_win` is how well its best teams did and `reached_full_stats` how often they got to the floor's real stats (0: nothing within its borders wins there):
```
python3 -m card_engine.training.hard --select engine --tower 65 Impossible --prior 0 --rounds 4 --shards 1 --workers 4 --first-seed 9400000
```
Pick a `--first-seed` no other miner uses (the running ones: 9,000,000 floor 105, 9,500,000 floor 100, 9,600,000 floor 105 engine, 9,700,000 to 9,900,000 blind spots, 8,500,000 depths).

Generator options:
- Pools: `--pool own|custom|restricted|all`, `--enemy-pool ...`.
- Masks: `--borders none Pl Cr`, `--mutations None Storm`, `--support-tiers base Crystal`, `--max-rarity 2.5M`, `--no-limited`.
- Role: `--role attack|defend`.
- Search: `--model-search N` (teams the model scores; 50,000 is about a minute), `--search-population 512`, `--search-parents 128`, `--search-children 4`, `--counters 32`; the gradient ascent's `--restarts`, `--steps`, `--noise-levels`, `--sigma-max 8`, `--temperature`, `--nearest`.
- Engine: `--engine-search N` (engine battles that keep improving the model's best teams; slower), `--no-verify`.
- Close the trainer's GPU work first, or add `--device cpu`.

Depths (hard depths by default: the floor's enemy pool, the stats of floor × 10; `--normal` for normal depths). A team's depth curve, the model's and with `--simulate` the engine's on the same enemy draws, and the model search for the team that goes deepest (its best `--top` engine-verified):
```
python3 -m card_engine.depths run --checkpoint data/training_10s/ema.checkpoint --ally Parallax "Judgement Day" "Judgement Day" "Robin Hood" --ally-blue Fate --simulate
python3 -m card_engine.depths search --checkpoint data/training_10s/ema.checkpoint --pool own --device cpu
```
Speedrun teams (aura packs per hour, Drago's timing and rewards, battle lengths from our engine; it searches with the engine, since the model gives no battle length, starting from `--ally`):
```
python3 -m card_engine.depths run --checkpoint data/training_10s/ema.checkpoint --ally Tricerotops Velociraptor "Julius Leader" "Julius Leader" --bans speedrun --simulate
python3 -m card_engine.depths search --objective speed --checkpoint data/training_10s/ema.checkpoint --ally Tricerotops Velociraptor "Julius Leader" "Julius Leader" --bans speedrun --pool own --device cpu
```
A card slot can be a whole pack, `pack:Prehistoric` (with `@Border` or `/Mutation` if wanted), wherever the lineup puts it: `run` then plays one team per card of the pack, the lineup as typed, and ranks them (packs per hour with `--simulate`, else the model's expected floors):
```
python3 -m card_engine.depths run --checkpoint data/training_10s/ema.checkpoint --device cpu --ally Tricerotops pack:Prehistoric "Julius Leader" "Julius Leader" --bans speedrun --simulate --workers 2
```
Search for a faster team and its bans together: `--optimize-bans N` alternates a ban step (the N bans that raise your team's packs per hour most) and a team step (the engine search under those bans), `--rounds` times, then plays the final team and bans on fresh draws. Bans you give with `--bans` are kept and count toward the 14. A ban step plays no new battles: banning a card leaves every remaining draw equally more likely, so a ban list is scored on the team's battles against draws without the banned cards; each ban is chosen on half the draws and kept only if it also helps on the other half. The output shows each round, the final bans and the teams; with `--ally`, also that team under the final bans and, as the baseline, with only your own bans.

It builds its own teams; `--ally` is optional, a starting team to beat (user, 2026-10-08). Each team step is the model's search (`--evaluations`, 5,000 teams by expected floors), whose 32 best distinct teams seed the engine's search for speed (`--engine-evaluations`, 300 teams by packs per hour), since the model gives no battle length. Use a frozen copy of the averaged weights (`cp data/training_10s/ema.checkpoint data/training_10s/depths_ema.checkpoint`) so the model is stable between searches, and check it against the engine with `depths run --simulate` (model and engine floors should agree) before trusting what it builds. Every team the engine scores and every ban step's per-card gains go to `data/depths_search.jsonl`, and grow with each search; `stats` turns that evidence into lifts per card, card pair and support (the mean score of teams holding it minus the mean of all), mean ban gains per card, and the best decks with their bans:
```
python3 -m card_engine.depths search --objective speed --optimize-bans 14 --checkpoint data/training_10s/depths_ema.checkpoint --device cpu --pool own --workers 2
python3 -m card_engine.depths stats
```
`--pool progression` searches what a player at `--rolls` (205M) and `--luck` (100) can expect to own: every card × border × mutation with at least one expected copy, borders rolled each with its own chance (a pro's: Platinum 1/2, Crystal 1/10, Ruby 1/100, Galaxy 1/1,000; `card_engine/training/incomplete.py` Progression), every support tier. A starting team's own cards are always added, since a pro shows his luckiest pulls. For a player whose deck you don't have:
```
python3 -m card_engine.depths search --pool progression --rolls 205e6 --luck 100 --ally Fuxi "Cosmic Pop Star@RuPl" "Typhon@RuCrPl" "Julius Leader@RuCrPl" --ally-red "Desmond Of Despair@Galaxy" --ally-blue "Vampire Matron@Galaxy" --bans drago --checkpoint data/training_10s/depths_ema.checkpoint --device cpu --workers 2
```
Improve a team you have, one change at a time, by the engine (no model needed, so no `--checkpoint`). It finds where the team starts dying, screens every single change (a slot for the strongest owned version of any card, or a support) on the same draws there, plays the best few on the full survival curve with fresh draws, and takes the best if it beats the team; then, with `--optimize-bans N`, it picks the bans for the final team and checks them on fresh draws (a ban that does not hold up on both halves of the draws is not taken, so fewer than N may come back). Give the team's supports and bans as they are, and the player's rolls and luck if you don't have their deck:
```
python3 -m card_engine.depths improve --pool progression --rolls 46e6 --luck 100 --ally Fuxi "Cosmic Pop Star@Pl" "Poison Witch@RuCrPl" "Julius Leader@RuPl" --ally-red "The One Ring@Galaxy" --optimize-bans 7 --workers 4
python3 -m card_engine.depths improve --normal --pool own --ally ... --bans drago --workers 4
```
Options: `--normal` (default hard depths), `--objective depth|speed`, `--rounds 4` (one change per round), `--keep 6` (screened changes verified per round), `--screen-draws 48`, `--verify-draws 512`, `--ban-draws 1024`, `--pool own|custom|restricted|all|progression` (with `--rolls`, `--luck`), `--seed`. A run takes minutes (the C engine plays about 700 battles a second on 4 workers). Each round prints the verified candidates; the last line is the final team, its bans and its expected floors (or packs per hour).

`stats` options: `--objective packs_per_hour|expected_floors`, `--normal`, `--min-count 5`, `--top 15`. Lifts come from teams a search chose, which cluster around what it liked: they say "teams with it scored higher", not that the card caused it.

Its options: `--optimize-bans N`, `--rounds 2`, `--ban-samples 64` (draws per floor the ban step scores on), `--evaluations 5000` and `--engine-evaluations 300` (teams per team step), `--restart-seconds S` (time from a death to the next run's first floor, so fast-dying teams are charged for restarting; 0 until measured). With `--objective depth` the bans raise expected floors instead.

`--certain 0.001`: a model win chance within this of 1 counts as a sure win (the model leaves about 1e-5 to 1e-4 of loss on battles the engine never loses, which compounds over tens of thousands of floors; 0 shows the raw model). Options: `--bans CARD ...` (your Depth bans, up to 14) or a preset (`speedrun`, `pro`), `--cap 10000` (last floor), `--points` (floors sampled, geometric; 24, or 12 for the speed search), `--samples` (enemy draws per floor; 32, or 8), `--evaluations` (5,000 teams by the model; 300 by the engine for speed), `--no-chrono-shard`, `--structure N` (battle-speed structure 0-7), `--skill-tree N` (0-4). Output: `expected_floors` (cleared on average), `median_death_floor`, per sampled floor the mean win chance and the chance to have survived that far; with the engine also `turns` per floor, `minutes` and `packs` per run, `packs_per_hour` and `floors_per_hour` (runs restart from floor 1).

Model against engine on a suite of battles:
```
python3 -m card_engine.training.verify --checkpoint data/training_10s/ema.checkpoint --suite tower --difficulty Impossible --n 200 --show 20
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
python -m card_engine.training.train --run-dir data/training_10s --device cuda --bf16 --batch-size 512 --lr 3e-4 --weight-decay 0.05 --dropout 0.1 --mix 0.03 0.1 0.1 --eval-rows 100000 2>&1 | tee -a data/training_10s.out
```
To continue a Mac run there, copy `data/training_10s/` up first (it holds the checkpoints and `trainer.pt`).

Stop the pod at a set time (UTC), archiving every run to `/workspace` first:
```
cd /workspace/card_engine && nohup setsid scripts/runpod_finish.sh "+6 hours" > /workspace/finish.log 2>&1 < /dev/null &
cat /workspace/finish.log
```

Before the balance runs out, bring everything down from the Mac:
```
rsync -az --progress -e "ssh -p PORT" root@IP:/workspace/card_engine/data/training_10s/ data/training_10s/
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
