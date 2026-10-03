import { readFileSync } from 'node:fs'
import { solve } from './search'
const battles = JSON.parse(readFileSync('/tmp/claude-0/-home-user-Card-AI/c8c704aa-4a29-55ec-a0fe-96b73548579c/scratchpad/loadouts.json', 'utf8'))
for (const budget of [100, 2000]) {
  let same = 0, differ = 0, tReplay = 0, tResume = 0, exact = 0
  for (const [a, b] of battles) {
    let t = performance.now(); const r1 = solve(a, b, { nodeBudget: budget, rolloutError: 0.03, snapshots: false }); tReplay += performance.now() - t
    t = performance.now(); const r2 = solve(a, b, { nodeBudget: budget, rolloutError: 0.03, snapshots: true }); tResume += performance.now() - t
    const close = Math.abs(r1.a - r2.a) < 1e-9 && Math.abs(r1.b - r2.b) < 1e-9 && Math.abs(r1.draw - r2.draw) < 1e-9
    if (r1.exact && r2.exact ? close : !r1.exact) same++  // exact answers must agree; merging may finish more battles
    else { differ++; if (differ <= 3) console.log('DIFFER', JSON.stringify(r1), JSON.stringify(r2)) }
    exact += r2.exact ? 1 : 0
  }
  console.log(`budget ${budget}: consistent ${same}/${battles.length}, exact ${exact}, replay ${(battles.length / tReplay * 1000).toFixed(1)}/s, resume ${(battles.length / tResume * 1000).toFixed(1)}/s`)
}
