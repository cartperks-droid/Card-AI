# Training the general win predictor

## Labels (`card_engine/training/labels.py`)
- **Engine:** DaddyDrago's battle rules (set up with `bash sim_js/setup.sh`; see CLAUDE.md). Our old Python/C simulator was deleted on 2026-10-03.
  - **C engine (`sim_c/`, user 2026-10-03: "build it in C ... don't leave out any abilities"):** labels run on a C port of his battle code and of `sim_js/search.ts`. His TypeScript still builds each battle's start (supports, deck passives, Draconian, Astraeus arts, tweaks); `card_engine/simulator/kernel.py` loads it into C, which runs the search. A branch is one `memcpy` of the battle.
  - **Same answers:** the C search gives bit-identical results to the TypeScript search. Checked on 150 sample battles, every supported card (3 battles each) and support (2 each), and every one of his 324 abilities forced onto cards (1,002 battles); together these run 99.8% of the engine's lines (the rest: an unused helper, a branch his own code never calls). `tests/test_kernel.py` keeps checking.
  - **Speed:** about 217 labels/s per core at the design budget (his TypeScript: 20).
  - **Limits:** 4,096 cards per battle and nesting depth 2,000 (where his engine would throw "Maximum call stack size exceeded"). Past them a battle is abandoned and logged like any engine error, never answered wrongly. Ordinary battles stay under 32 cards, but Pandora's rolls can run away: one label battle created 836. A branch copies only the cards and list entries in use, so the high limit costs nothing in ordinary battles.
- **Battles:** random 4v4 matchups over all 289 cards. Each Astraeus gets a random art, and the art is its own card (user). Each card also gets:
  - a border;
  - a mutation, for eligible (Base-weather) cards only, 50% of the time;
  - red and blue supports, absent 10% of the time, each with a tier: Base, Platinum, Crystal, Ruby or Galaxy (1-5).
- **Borders:** 70% of battles keep every border within two rarity ranks of a shared level. The rest draw borders uniformly.
- **Who starts:** side A always initiates. A draw (double KO, or the 2,000-turn cap) counts as A's loss.
- **Search (user's design, `sim_js/search.ts`, ported in `sim_c/search.c`):** every random draw in his engine is a chance point. A branch is the list of choices made at those points, replayed from the start of the battle.
  - **Large branches first:** the most probable open branch is expanded next, up to 20,000 per label (`drago.SEARCH`).
  - **Saved turns:** like the old C kernel continuing from stored states, a branch is the battle saved at the start of the turn in which it split, plus the choices since then. Expanding it restores that turn and plays on; it never replays from turn 1. Results are identical to replaying (checked in `tests/test_training.py`). Ability lookups read a per-card entry (ability, names, their set) rebuilt only when the card's overrides, bonus abilities or definition change (`codemod.mjs`); results are identical to his unpatched code. Cards whose abilities never changed in battle read a shared entry at their card index.
  - **Small branches:** a branch below 0.1% probability is not expanded. It joins the playout pool.
  - **Leftover branches:** small branches, plus whatever is open when the budget runs out, are resolved by pooled playouts. Each playout picks a branch in proportion to its probability, then plays on at random.
  - **Variance-based count:** playouts continue, up to 1,024, until the standard error of the A-win estimate is at most 3%.
- **Exact vs estimated:**
  - A battle with no estimated probability is **exact** and goes into the tablebase (`data/tablebase/<fingerprint>.sqlite`).
  - Estimated outcomes are never stored. A battle that comes up again is re-estimated with fresh playouts, so over time the model sees the spread of probabilistic outcomes.
  - Unfinished probability is column 3 of `probs` and is left out of training targets.
- **Storage:** shards go to `data/labels/store/shard_<seed>.npz` (`probs`, `exact`), stamped with a rules snapshot. `training/flags.py` decides per row whether it is still valid.

```sh
../.venv/bin/python -m card_engine.training.labels --shards 1000 --rows 2000 --workers 12
```

## Fixed-stat battles (`labels.py --fixed`)
Some battle modes give every enemy card the same stats, borders ignored (user, 2026-10-04). The labels cover the general problem, not one mode (user).
- **Battle:** a random matchup in which one side, A or B at random, is borderless and starts all four cards at one (HP, ATK). The level is the opponent's geometric-mean stats times a factor; the HP/ATK balance moves by 10^U(-0.5, 0.5).
- **Big gaps** (user, 2026-10-04): half the battles take the factor from 10^U(-2, 4); the other half from 10^U(1, 4.5), and each of the weaker side's cards is, with probability 1/2, one whose ability ignores raw stats (`labels.STAT_IGNORING`, read from his ability texts: damage scaled to the enemy's HP, kills, revives, shared damage). Random teams almost never win at such gaps otherwise.
- **Range:** the level reaches 10,000×. Floor 105 Impossible is about 2,700× a borderless cheese deck. The first fixed shards (2026-10-04) spread it only 10^U(-1.5, 1.5) and never reached such gaps; they stay valid.
- **Storage:** `fixed_<seed>.npz`, with `fixed_side`, `fixed_stats` (HP, ATK) and `fixed_hp_mult` per row, on their own seed sequence. They are never put in the tablebase, whose key has no stats. Trainers from before fixed shards read only `shard_*.npz`.
- **Model:** the trainer starts that side's cards at `fixed_stats` (`train.card_stats`). `fixed_hp_mult` multiplies HP by each card's HP multiplier, as tower floors do on Normal and Impossible (`card_engine/tower.py`, used by `--tower`); the labels leave it off.

```sh
python -m card_engine.training.labels --fixed --shards 1000 --rows 2000 --workers N
```

## Hard examples (`training/hard.py`)
Battles where the classifier and the engine disagree most (user, 2026-10-04).
- **Not the generator:** proposing teams by ascending the classifier finds teams the model already gets right. The first such shards had a mean |model − engine| of 0.004.
- **Not random candidates:** against huge stats almost none win (0 of 200 at 10.9M HP). Cheese decks are specific four-card lineups.
- **Round:** a fixed-stat enemy, half a tower floor (weighted toward the top: a quarter on floors 95-105, difficulties Normal 1 : Hard 1 : Extreme 2 : Hell 3 : Impossible 3; its fixed team or random cards), half random cards at HP 10^U(2, 7.5) and ATK half that, moved by 10^U(-0.5, 0.5). An engine-guided search spends 1,024 engine evaluations (`--candidates`, milliseconds each): 64 random teams leaning on stat-ignoring cards (`labels.STAT_IGNORING`); each generation the best 16 get 4 variants each (a card, a support, or the lineup order changed), keeping the best 64. Every team evaluated is scored by the model in one batch. The 16 with the largest |model − engine| (`--keep`), each at least two places (lineup cards, supports) apart from the others, and 4 others (`--keep-random`) are kept, the candidate attacking first. Many enemies, few teams each: one evolution's teams are mostly one-change variants, and keeping 64 per enemy (the first ~1,600 enemies, 129k rows) was memorised (probe at a 0.5 hard share, 2026-10-05: hard training KL 0.04, held-out 0.46; the 0.05 run 0.18 against 0.34).
- **Generator rounds** (2026-10-05): every 4th enemy (`--generator-every`) gets the annealed generator's 32 best teams by the model (sigma 4, 12 noise levels; a random border, mutation and support-tier mask) instead of the engine search's. The engine search finds the winners the model underrates; the generator finds the losers it overrates: borderless at floor 105 Impossible it proposed Kira / Time Lord Stryx / Legends / Hades at 0.92, which the engine loses every time.
- **Storage:** `hard_<seed>.npz` (80 rounds, 1,600 battles by default), the fixed-stat fields plus `model_win`, the model's win chance at mining time. Each shard prints `mean_gap_all` (every team evaluated), `mean_gap_kept` (the kept 16 per round) and `mean_best_engine_win` (the best win chance the search reached per enemy).
- **Training:** a set share of each batch (`--mix HARD UPSET FIXED`, hard 0.05 by default) is drawn from hard rows, the rest uniformly from all rows; no copies are kept (repeating rows, tried first, needed tens of GB at high weights). Validation's hard rows are scored on their own (`val_hard`; `scripts/perf.py`). `scripts/hard_fit.py` compares a checkpoint's KL on the hard rows it trains on with the held-out ones: a widening gap is memorisation.
- **Model:** `--checkpoint`, by default the pod's (`data/training_pod/`, as `pod_sync.sh` brings it down), else `data/training/`; reloaded every shard, so the mining follows training.

```sh
python -m card_engine.training.hard --shards 100 --workers 7
```

## Incomplete mode (`training/incomplete.py`)
PvP (user, 2026-10-06): you don't know who will attack you, and your attackers don't know your team. Building a team in incomplete mode is finding one that works against strong teams in general, so the unseen side means the **field**: strong teams the model itself generates, kept only if the engine confirms them. As the model gets stronger the field does too, and the hidden side's meaning moves with it (user: "the hidden slots meaning should change as the model gets stronger").
- **Model input:** one side is not shown at all (`hidden_side`): its cards, stats and supports become the hidden stand-ins and MODE is 1. A visible attacker is side A against a hidden B; a visible defender is side B against a hidden A. The stored fields of the hidden side are `labels.HIDDEN_TEAM`, never read.
- **The field** (`data/labels/field/field_<generation>.json`), by self-play: per role (attackers and defenders), `--candidates` (160) teams are scored by the engine against `--opponents` (24) of the previous generation's other role, and the best `--size` (48), each two places apart, are kept. Candidates: the previous field's members, the model's own proposals in incomplete mode (`generate.Settings.incomplete`: ascended against the unseen field), counters to a few rival members, one-change variants of members, and random teams. Generation 0 is scored against the generator's broadly ascended teams. Theoretical space: every border, mutation and support tier.
- **Labels** (`hidden_<generation × 1,000,000 + n>.npz`), mined like hard examples: each round takes one role and 48 candidates (12 field members, 12 variants, 12 of the model's incomplete-mode proposals, the rest random), plays each against a fresh draw of 16 rival-field members, and the mean is the soft label. The model scores the same teams in incomplete mode; the 8 it gets most wrong (two places apart) and 4 others are kept. Any rules change invalidates these rows (`flags.valid_rows`: the field teams behind a target aren't named in the row).
- **Training:** incomplete-mode rows enter batches only through their own share, `--mix HARD UPSET FIXED HIDDEN`; uniform draws skip them. Only the newest `--field-generations` (2) generations count, so labels against an outdated field drop out. Validation scores them apart (`val_hidden`, `perf.py`).
- **Using it:** `Classifier.field_win(teams, role)` and the generator with `Settings(incomplete=True)` score or ascend a team against the unseen field.
- **Loop:** train, then build the next generation with the newest weights, then mine labels against it, and repeat as the model improves.

```sh
python -m card_engine.training.incomplete field --checkpoint data/training_18t/ema.checkpoint --workers 6
python -m card_engine.training.incomplete labels --shards 10 --checkpoint data/training_18t/ema.checkpoint --workers 6
```

## Model inputs
- **Supports:** tiers get their own embedding (`support_tiers [B,2,2]`, 1 base .. 5 Galaxy).
- **Astraeus:** each art has its own permanent identity key (`data/annotations/card_keys.json`, keys 290-296).
- **Classes:** the user-verified memberships are used.
- **Card stats (user):** each card's (HP, ATK) as it enters the battle: border, mutation, the red support and the blue Jurassic World support, from his engine's tables (`drago.stat_tables`). They match his battle-start stats exactly, except for the deck passives (General Moon Zoo, Julius Leader). Stat changes from abilities are left to the model. The stats are log-scaled, then centred on one shared mean over all visible cards of both teams. No interaction depends on absolute stat sizes, so only ratios remain, and HP and ATK share the scale, which keeps their ratio (hits to kill). A linear projection to the token width skips around a GELU MLP (3,072 wide), and the sum is added to every card token. Before this, base stats had to be encoded in the card's language vector and identity embedding.
- **Stats beside the card** (user, 2026-10-06; for new runs: `train.py --stat-width 128 --no-pack-embedding --no-mutation-embedding`): with `--stat-width S` the stats get the last S channels of each card token and everything else about the card (description, identity, classes, border, mutation, pack) a learned projection onto the rest, each part layer-normalised on its own. Added in one space, a large stat vector could swamp the description inside each block's layer norm, which is where cheese decks live: floor 105 Impossible is about 2,700× a borderless deck. Separated, the stats and the abilities start orthogonal, and the transformer learns where they interact. The engine reads a card's pack and mutation weather only for its stats and for the stat supports that boost it (`auras.ts` `isStatAuraBoosted`, The One Ring), all of which are in the card's stats already, so their embeddings only add a way to tell cards apart without reading them; borders stay, because Dominate compares border tiers in battle. The settings are saved in the checkpoint, so every tool loads such a model unchanged.
- **Stat tokens** (user, 2026-10-06: two layouts in one token read like two inks overlapping): `--stat-tokens` gives each card a second token for its stats, the stat MLP's output with its own learned position (8 stat positions), placed before MODE and PREDICT; a hidden card's stat token is a learned stand-in. Each token is normalised on its own in every block, so neither can drown the other, and attention relates a card to its own stats and to everyone else's: the sequence grows from 15 to 23 tokens, about 1.5x the work per layer. It excludes `--stat-width`.
- **Stat comparisons** (user, 2026-10-06: an 8x7 comparison matrix): `--stat-pairs` feeds the stat MLP each card's stats against each of the other 7 cards, ordered from the card's own point of view (its side's other three by slot, then the opposing four), so a column means the same for ally and enemy tokens. Per pair: log HP and log ATK ratios, the hits it needs to kill the other (log of the other's HP over its ATK) and the hits the other needs to kill it, and whether the other card is visible (an unseen card's comparisons are zero). With its own normalised stats that is 37 inputs instead of 2; all are log differences, so scaling every stat together still changes nothing. Attention could work these out across layers; given directly, each stat token starts knowing where its card stands against every other.
- **Adding the stat input to a run that predates it:** stop every trainer on the run first, then run `python -m card_engine.training.add_stat_mlp`. It adds the projection and MLP at zero, so predictions are unchanged at first, and keeps Adam's state. The originals are kept as `*.pre-stat-mlp`. Files already upgraded are skipped, so it is safe to run again. The script is one-off: delete it once the run is migrated.

## Training (`card_engine/training/train.py`)
- **Loss:** soft cross-entropy on the outcome frequencies, plus the identity L2 penalty.
- **Optimiser:** AdamW at 3e-4 with 1,000 warm-up steps, on the GPU (MPS).
- **Description encoder:** each step encodes all 289 descriptions once, so it trains end to end.
- **Validation:** shards whose seed is divisible by 25 are held out. Fixed-stat validation battles are also scored on their own (`val_fixed` in the log; `scripts/perf.py` prints them).
- **Data refresh:** every 1,000 steps the newest label directory is reloaded, which picks up new shards or new rules.
- **Resuming:** the model checkpoint, optimiser state and step count live in `data/training/`. The log is `data/training/log.jsonl`.
- **Moving machines:** `--init-from <model.checkpoint>` starts a run directory that has no optimiser state from that model. It keeps the step count and uses a fresh optimiser that warms up again over 1,000 steps. Use it when only the checkpoint can be copied over, e.g. when the pod's own disk is out of reach.
- **Tried and dropped (2026-10-03):** amplifying upsets' gradients for the description encoder only (weight 2) made the two encoders chase different losses. Validation KL rose from 0.055 to 0.186 within 15,000 steps, and upsets fell from 78% to 52%. The same resume without it held at 0.0535.
- **Unfreezing the description transformer for a while** (user, 2026-10-04): restart with `--freeze-language-at` past the current step and a lower `--language-lr` (e.g. 3e-5). It has its own parameter group; its Adam state starts fresh when the saved optimizer predates that group, so momentum from before the freeze is not resumed. Watch validation KL: amplified upset gradients through this encoder once tripled it (above).
- **Fresh deeper run** (user, 2026-10-05): a new model with more strategic layers, trained curriculum-first, on the pod in `data/training_deep` (`pod_sync.sh` brings it down to the same name) while the current model keeps training. `--layers N` sets the depth of a run starting from random weights (saved in its checkpoints, so every tool loads it). `--language-from <model.checkpoint>` copies the current model's description transformer (card text to 768-wide card vectors, trained about 400k steps) and freezes it from step 0: it knows nothing about stat gaps, so the "bigger stats win" habit is not carried over. `--mix-start H U F --mix-until STEP --mix H U F` is the curriculum (2026-10-06, replacing an all-or-nothing focus phase): each batch takes set shares of hard examples, upsets (the stat favourite lost) and fixed-stat battles (`mix_rows`), the rest uniformly from all rows; the shares move linearly from `--mix-start` at step 0 to `--mix` at `--mix-until`, then hold (`mix` in the log). A model that starts on the battles abilities decide does not learn "bigger stats win" first, and the end shares keep the cheese battles from being drowned out once it sees the natural mix. Step cost by depth (CPU, batch 512, frozen description; relative, as the GPU's will be): 4 layers 1.0×, 6 1.1×, 8 1.7×, 12 2.1×; each layer adds 5.3M weights to the 27M of the 4-layer strategic model.
- **Weight average** (user, 2026-10-05): the trainer keeps an exponential moving average of the weights (`--ema-decay`, 0.999: about the last 1,000 steps) and saves it as `ema.checkpoint` beside `model.checkpoint`. Training itself is unchanged (Adam still updates the live weights at a fixed rate); the average only smooths what the tools read. At a fixed learning rate the live weights swing between evaluations (the floor-105 cheese decks moved 0.39 → 0.76 → 0.60 within 2,000 steps), so `predict`, `generate` and `hard` should read `ema.checkpoint`. Each evaluation scores it on the validation probe and the hard examples (`ema` in the log; `scripts/perf.py` prints it beside the live weights). A run without one starts its average from its current weights.
- **Learning-rate decay** (2026-10-05): `--lr-decay FIRST LAST` lowers the rate along a cosine from step FIRST to `--lr-floor` (0.05) of it at step LAST, then holds it; both steps are given, so restarts keep one schedule. A fixed rate keeps the weights swinging (the weight average scores better than them), so a run's last stretch should decay. Log lines show `lr`.
- **Watch list** (user, 2026-10-05): `card_engine/training/watch.json` names battles (DaddyDrago's five floor-105 Impossible cheese decks to start) that every evaluation scores with the live weights and their average beside the engine's answer, computed once at start (`watch` in the log; `scripts/perf.py` prints it). Narrow cases move without the summary metrics moving: the decks went 0.08 → 0.76 → 0.02 between checks by hand. `--watch ''` turns it off. The weights closest to the engine on it (mean |model − engine|) are kept as `watch_best.checkpoint` (live) and `watch_best_ema.checkpoint` (average), replaced only by better ones; the best errors survive restarts.
- **Row cap** (2026-10-06): `--max-rows N` loads about N rows: every hard example, every shard written after the trainer started, and a fixed random share (by file name, so validation keeps its 1 in 25) of the shards already there. Loading the whole store, 165M rows once the pod's shards came down, crashed the Mac, whose CPU and GPU share one memory (the cached shards and the tensors are two copies there). About 110 bytes per row, twice.
- **Evaluation sample** (2026-10-06): `--eval-rows N` scores a random N validation rows (plus every hard example) at each evaluation instead of all of them. On the Mac an 18-layer evaluation of 830k rows took longer than the 1,000 training steps before it. The draw is fixed per validation size, so successive evaluations compare until new shards arrive.
- **Separate runs:** `--run-dir` puts a run elsewhere. A new directory starts from random weights; `predict.py --run-dir` reads that run's model.

```sh
../.venv/bin/python -m card_engine.training.train --batch-size 512
```

## Daily checks
- The scheduled task `card-engine-daily-checks` runs at 09:00. It writes `docs/daily_checks/YYYY-MM-DD.md` with 3-5 in-game tests, rule or class, chosen from `docs/daily_checks/queue.md` and the provisional rules.
- Each test comes with the simulator's prediction and a training status section.
- Fixing a rule changes the fingerprint, so labels regenerate and training continues on them.

## Win rate of a matchup (`card_engine/training/predict.py`)
```sh
../.venv/bin/python -m card_engine.training.predict \
    --ally "Vampire Lord@GaPl" Set "Good Boy@Pl/Storm" Archer --ally-red Stormcaller@Galaxy --ally-blue "Guardian Angel@Ruby" \
    --enemy "Immortal Witch" Archer "Good Boy" Set --enemy-blue Fate@Crystal --simulate
```
- **Teams:** cards are `Name[@Border][/Mutation]` and supports `Name[@Tier]` (`card_engine/teams.py`). A name can be an ID, the exact name, or a unique part of it.
- **Output:** the classifier's ally win chance when the ally attacks first and when the enemy does. `--simulate` adds the engine's answers and whether each is exact. `--checkpoint` picks the model (default `data/training/model.checkpoint`).
- **Fixed enemy stats:** `--enemy-stats HP ATK` (e.g. `2.5M 400k`) starts every enemy card at those stats and drops the enemy's borders, for the classifier and the engine alike. `--tower FLOOR DIFFICULTY` (e.g. `--tower 105 Impossible`) uses a tower floor's stats, and its enemy team on fixed floors when `--enemy` is left out. The generator takes both.

## Generator (`card_engine/training/generate.py`)
```sh
../.venv/bin/python -m card_engine.training.generate --enemy "Immortal Witch" Archer "Good Boy" Set --pool restricted
../.venv/bin/python -m card_engine.training.generate --enemies 4 --pool restricted --borders all --support-tiers all --output counters.json
../.venv/bin/python -m card_engine.training.generate --enemy Odin Kira Set Archer --pool custom
```
- **Slot space:** each of the 4 slots is three vectors, one per factor of a card token. The card factor is description + pack + classes + identity; the others are the border and mutation embeddings. Each side also has a distribution over the pool's supports. Everything starts from noise.
- **Ascent:** Adam maximises the classifier's log win probability in one role, never both at once (user, 2026-10-04): `--role attack` (default; your team attacks first and loses a mutual wipe) or `--role defend` (the enemy attacks first). Broad enemies are ascended in the other role. Weights never change.
  - **Annealing** (user, 2026-10-05): like a diffusion sampler, with the classifier's gradient as the score. The steps run in `--noise-levels` levels (12); each after the first starts by adding fresh noise to the slot vectors and support logits, shrinking geometrically from `--sigma-max` (8, in each factor's spread) to 0.02. At floor 105, borderless, restricted pool: sigma 0.5-1 stayed on the stat-stacking peak, 2-4 reached high model scores, 8 found the first team the engine lets win (0.108). Early levels let a slot leave a poor peak; late ones only polish. One level is the plain ascent. Before it, at floor-105 stats the decoded teams scored 0.004 by the model while a known cheese deck scored 0.26 by the same model: the ascent was stopping on poor peaks.
  - **Stats:** a slot's stats are the expected log stats under p(entry) = softmax(-distance / temperature). The distance factorises over the three factors, each in units of its table's typical neighbour gap.
- **Commitment and entropy check:** the distance to each slot's nearest entry is penalised, more and more during the ascent. While a slot still spreads over more than 1.5 effective entries, the ascent continues with a doubled penalty, up to 3 times. `slot_blur` in the output is the worst slot's effective entry count.
- **Decoding:** an exact factorised nearest-k match over the pool. Each slot keeps its 3 nearest entries and each colour its 2 most likely supports. The classifier scores every combination exactly, and each candidate keeps its best.
- **Counters:** 64 candidates per enemy (`--restarts`). The best 32 distinct teams by the model (`--counters`) are verified by the engine and sorted by the engine's win chance in that role. `model` and `simulator` are the ally's win chance in the chosen role.
- **Low win chances count** (user, 2026-10-04): in tower battles any non-zero chance makes progress by retrying, so nothing is cut off. Teams the engine ties (e.g. all 0.0) keep the model's order. The engine's playouts resolve a win chance to about 3%, so its 0.0 may hide a small non-zero chance.
- **Enemies:** `--enemy` names one. `--enemies N` generates N broad ones: each is ascended against a random pool opponent, then decoded by sampling at temperature 1.
- **Single-copy cards:** at most one Fate Seamstress (user, 2026-10-04: a second copy doesn't work), one Time Lord Stryx (user: Better Days doesn't work twice; his engine lets two copies revive each other in a loop, which once scored a floor-105 team 85%) and one Parallax (user; DaddyDrago's tower tool limits it too) per team, in every pool and in new labels (`teams.SINGLE_COPY`). Older shards' battles with two or more copies (his engine lets them work twice) are dropped when shards load (`labels.possible_rows`).
- **Pools:**
  - `own`: your deck (`python -m card_engine.deck`), with copy counts.
  - `custom`: a second deck-format list for suggestions tailored to one collection, changed on the fly with `python -m card_engine.deck --custom ...`. `reset` empties it and `copy-deck` starts it from your deck.
  - `restricted`: the player base's cards and borders, without mutations.
  - `all`: everything.
- **Masks** (user, 2026-10-04): the pools keep everything, but the generator leaves rare options out unless asked. They narrow `restricted` and `all`, for the counters and for `--enemies`:
  - `--borders`: `none` by default; a list (`none Pl Cr`) or `all`.
  - `--mutations`: `None` by default; a list (`None Storm`) or `all`.
  - `--support-tiers`: `base` by default, red and blue alike, since supports have no index to price them; a list or `all`.
  - `--max-rarity`: leaves out entries rarer than this card × border rarity, the rarity the game shows (1 in N: `2.5M`, `30T`, `10qd`).
  - A deck overrides the masks: `own` and `custom` are used as they are, supports included.

## Validation metrics
- `accuracy`: how often the model picks the most frequent outcome.
- `baseline`: the same accuracy for a naive rule where the side with more total sqrt(HP × ATK) wins. Abilities are ignored.
- `upset_accuracy`: accuracy on the battles that rule gets wrong, where abilities decided the outcome. This is the number that shows the model learning card abilities.

## Counter-team search (`card_engine/training/counter.py`)
```sh
../.venv/bin/python -m card_engine.training.counter data/scenarios/<enemy>.json --mode semi --max-rolls 7.8e10
```
- **Modes:**
  - `own`: only your deck (`data/my_deck.json`, edited with `python -m card_engine.deck`).
  - `semi` (default): your deck plus cards from the restricted deck.
  - `restricted`: only the restricted deck, at every support tier.
- **Restricted deck:** player-base availability (`data/restricted_deck.json`, edited with `python -m card_engine.restricted`). It holds every pack except seasonal and Limited, plus Fate Seamstress, Eclipseborn Luminant, Supreme Ozzy, Frank and The Broken One; `--no-limited` drops those Limited ones. Every border is allowed except GaRuCrPl, and cards and borders can be added or removed per card.
- **Cost:** a card you don't own costs its expected rolls, card rarity × border rarity ÷ the chance of rolling in its weather (the scenario's `weather_availability`). `--max-rolls` caps it.
- **Output:** 8 teams that differ from each other, each with its win chance attacking first and defending. `--dump` saves every evaluated team.
