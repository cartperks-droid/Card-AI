# Training the general win predictor

## Labels (`card_engine/training/labels.py`)
- **Engine:** DaddyDrago's battle engine (`sim_js/`, set up with `bash sim_js/setup.sh`; see CLAUDE.md). Our old Python/C simulator was deleted on 2026-10-03.
- **Battles:** random 4v4 matchups over all 289 cards. Astraeus draws its art in battle, so `arts` is 0. Each card also gets:
  - a border;
  - a mutation, for eligible (Base-weather) cards only, 50% of the time;
  - red and blue supports, absent 10% of the time, each with a tier: Base, Platinum, Crystal or Galaxy (1, 2, 3, 5).
- **Borders:** 70% of battles keep every border within two rarity ranks of a shared level. The rest draw borders uniformly.
- **Who starts:** side A always initiates. A draw (double KO, or the 2,000-turn cap) counts as A's loss.
- **Search (user's design, `sim_js/search.ts`):** every random draw in his engine is a chance point. A branch is the list of choices made at those points, replayed from the start of the battle.
  - **Large branches first:** the most probable open branch is expanded next, up to 2,000 battles per label (`drago.SEARCH`).
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

## Model inputs
- **Supports:** tiers get their own embedding (`support_tiers [B,2,2]`, 1 base .. 5 Galaxy).
- **Astraeus:** the art identity keys (`data/annotations/card_keys.json`, keys 290-296) stay in the model, but labels never set an art: his engine draws it in battle.
- **Classes:** the user-verified memberships are used.
- **Card stats (user):** each card's (HP, ATK) as it enters the battle: border, mutation, the red support and the blue Jurassic World support, from his engine's tables (`drago.stat_tables`). They match his battle-start stats exactly, except for the deck passives (General Moon Zoo, Julius Leader). Stat changes from abilities are left to the model. The stats are log-scaled, then centred on one shared mean over all visible cards of both teams. No interaction depends on absolute stat sizes, so only ratios remain, and HP and ATK share the scale, which keeps their ratio (hits to kill). A linear projection to the token width skips around a GELU MLP (3,072 wide), and the sum is added to every card token. Before this, base stats had to be encoded in the card's language vector and identity embedding.
- **Adding the stat input to a run that predates it:** stop every trainer on the run first, then run `python -m card_engine.training.add_stat_mlp`. It adds the projection and MLP at zero, so predictions are unchanged at first, and keeps Adam's state. The originals are kept as `*.pre-stat-mlp`. Files already upgraded are skipped, so it is safe to run again. The script is one-off: delete it once the run is migrated.

## Training (`card_engine/training/train.py`)
- **Loss:** soft cross-entropy on the outcome frequencies, plus the identity L2 penalty.
- **Optimiser:** AdamW at 3e-4 with 1,000 warm-up steps, on the GPU (MPS).
- **Description encoder:** each step encodes all 289 descriptions once, so it trains end to end.
- **Validation:** shards whose seed is divisible by 25 are held out.
- **Data refresh:** every 1,000 steps the newest label directory is reloaded, which picks up new shards or new rules.
- **Resuming:** the model checkpoint, optimiser state and step count live in `data/training/`. The log is `data/training/log.jsonl`.
- **Tried and dropped (2026-10-03):** amplifying upsets' gradients for the description encoder only (weight 2) made the two encoders chase different losses. Validation KL rose from 0.055 to 0.186 within 15,000 steps, and upsets fell from 78% to 52%. The same resume without it held at 0.0535.
- **Separate runs:** `--run-dir` puts a run elsewhere. A new directory starts from random weights; `predict.py --run-dir` reads that run's model.

```sh
../.venv/bin/python -m card_engine.training.train --batch-size 512
```

## Daily checks
- The scheduled task `card-engine-daily-checks` runs at 09:00. It writes `docs/daily_checks/YYYY-MM-DD.md` with 3-5 in-game tests, rule or class, chosen from `docs/daily_checks/queue.md` and the provisional rules.
- Each test comes with the simulator's prediction and a training status section.
- Fixing a rule changes the fingerprint, so labels regenerate and training continues on them.

## Comparing one matchup
```sh
../.venv/bin/python -m card_engine.training.predict --a 21,104,18,10 --b 205,44,3,1 --borders-a 5,5,5,5 --red-a 12:3 --blue-b 15:2
```
This prints the model's A / B probabilities next to the simulator's: exact from the tablebase when possible, otherwise estimated.

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
