// DaddyDrago's own, unmodified battle code (battle-v2.ts, as his site runs it) against fixed-power enemies, for
// card_engine.training.crosscheck. One JSON request per stdin line:
//   {"loadout": <his TeamLoadout>, "enemies": [{"name", "power", "attack", "health"}], "runs": N, "seed": S}
// answered by {"wins", "losses", "draws", "unsupported": [...]} over N battles with his seeded RNG.
import * as readline from 'node:readline'
import cards from './vendor/CardRngExpansionDepths/src/data/cards'
import { simulateBattleV2 } from './vendor/CardRngExpansionDepths/src/engine/battle-v2'
import { SeededRng } from './vendor/CardRngExpansionDepths/src/engine/rng'

const byName = new Map(cards.map((card: any) => [card.name, card] as const))
const lines = readline.createInterface({ input: process.stdin })
lines.on('line', (line) => {
  const request = JSON.parse(line)
  const enemies = request.enemies.map((e: any) => ({ card: byName.get(e.name), power: e.power, attack: e.attack, health: e.health }))
  const rng = new SeededRng(request.seed || 1)
  let wins = 0, losses = 0, draws = 0
  const unsupported = new Set<string>()
  for (let i = 0; i < request.runs; i++) {
    const battle = simulateBattleV2(request.loadout, enemies, Math.floor(rng.next() * 0x7fffffff) || i + 1, 2_000, true, false)
    if (battle.winner === 'Allies') wins++
    else if (battle.winner === 'Enemies') losses++
    else draws++
    for (const ability of battle.unsupportedAbilities) unsupported.add(ability)
  }
  process.stdout.write(JSON.stringify({ wins, losses, draws, unsupported: [...unsupported] }) + '\n')
})
