"""Cross-check our engine with DaddyDrago's own (user, 2026-10-04).

Our labels come from a codemodded copy of his battle code, searched over its chance points and ported to C. His own,
unmodified simulateBattleV2 (sim_js/drago_check.ts) can only play a team against fixed-power enemies (tower and
depths battles), so that is what this compares: random teams against random fixed-stat enemies, each scored by our
engine (labels.evaluate) and by --runs of his battles. A gap beyond --sigma standard errors (his sampling error plus
ours, 3% when estimated) is reported.

Teams avoid what his original engine handles differently by design: Astraeus (ours presets each art's ability, his
draws one) and Ruby supports (codemod.mjs adds Ruby; his tables lack it). Enemies are borderless, unmutated and have
no supports, at labels.fixed_battle's stats (levels around the team's own, half of them big gaps against
stat-ignoring cards); their power, which few abilities read, is set to their HP.

    python -m card_engine.training.crosscheck --battles 200 --runs 2000
"""

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

from ..catalog import load_catalog
from ..simulator import drago
from ..teams import ASTRAEUS
from .labels import evaluate, fixed_battle, random_spec

ROOT = Path(__file__).resolve().parents[2]
RUBY = 4


def draw_battle(rng, catalog):
    """(spec with side B as the enemy, (HP, ATK)): labels.fixed_battle's spread (levels around the team's stats, half
    of them big gaps against stat-ignoring cards), so outcomes are uncertain often enough to show differences."""
    while True:
        spec = random_spec(rng, catalog)
        side, stats = fixed_battle(rng, catalog, spec)
        if (side == 1 and ASTRAEUS not in spec["cards"][0] + spec["cards"][1]
                and RUBY not in (spec["red_tier"][0], spec["blue_tier"][0])):
            break
    spec["mutations"][1], spec["arts"][1] = [0] * 4, [0] * 4
    for key in ("red", "red_tier", "blue", "blue_tier"):
        spec[key][1] = 0
    return spec, (round(stats[0]), round(stats[1]))


def run(battles, runs=2000, seed=1, sigma=4.0):
    catalog = load_catalog()
    names = drago.names(catalog)[0]
    rng = random.Random(seed)
    node = subprocess.Popen(["npx", "--no-install", "tsx", "drago_check.ts"], cwd=ROOT / "sim_js", text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    rows = []
    try:
        for index in range(battles):
            spec, (health, attack) = draw_battle(rng, catalog)
            (ours_a, ours_b, _, unfinished), exact = evaluate(catalog, spec, seed * 1_000_003 + index,
                                                             fixed=(1, [(health, attack)] * 4))
            loadout = drago.loadouts(catalog, spec)[0]
            enemies = [{"name": names[card], "power": health, "attack": attack, "health": health} for card in spec["cards"][1]]
            node.stdin.write(json.dumps({"loadout": loadout, "enemies": enemies, "runs": runs, "seed": seed + index}) + "\n")
            node.stdin.flush()
            reply = json.loads(node.stdout.readline())
            his = reply["wins"] / runs
            ours = ours_a / max(ours_a + ours_b, 1e-12)
            error = math.sqrt(max(his * (1 - his), 1 / runs) / runs + (0 if exact else 0.03 ** 2))
            rows.append({"ours": round(ours, 4), "his": round(his, 4), "exact": exact, "z": round((ours - his) / error, 1),
                         "hp": health, "atk": attack, "unsupported": reply["unsupported"], "spec": spec})
    finally:
        node.stdin.close()
        node.wait()
    flagged = [r for r in rows if abs(r["z"]) > sigma and not r["unsupported"]]
    gaps = [abs(r["ours"] - r["his"]) for r in rows if not r["unsupported"]]
    return rows, flagged, gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--battles", type=int, default=200)
    parser.add_argument("--runs", type=int, default=2000, help="his battles per matchup")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=4.0, help="report gaps beyond this many standard errors")
    parser.add_argument("--output", help="write every compared battle here (JSON lines)")
    args = parser.parse_args()
    rows, flagged, gaps = run(args.battles, args.runs, args.seed, args.sigma)
    catalog = load_catalog()
    from ..teams import describe
    uncertain = sum(0.02 < r["his"] < 0.98 for r in rows)
    print(json.dumps({"battles": len(rows), "uncertain_outcomes": uncertain, "unsupported_by_his": sum(bool(r["unsupported"]) for r in rows),
                      "mean_abs_gap": round(sum(gaps) / max(len(gaps), 1), 4), "max_abs_gap": round(max(gaps, default=0), 4),
                      "beyond_sigma": len(flagged)}))
    for r in sorted(flagged, key=lambda r: -abs(r["z"]))[:15]:
        ally = {k: v[0] for k, v in r["spec"].items()}
        enemy = [catalog.card(c).name for c in r["spec"]["cards"][1]]
        print(f"ours {r['ours']:.3f}  his {r['his']:.3f}  z {r['z']}  enemy HP {r['hp']:,} ATK {r['atk']:,}  {describe(catalog, ally)}  vs {enemy}")
    if args.output:
        with open(args.output, "w") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
