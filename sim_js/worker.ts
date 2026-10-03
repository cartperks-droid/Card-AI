// Label worker: one JSON request per input line, one JSON result per output line, in order.
//   request: {"a": TeamLoadout, "b": TeamLoadout, "options": {...}}   (side A moves first)
//   result:  {"a": P(A wins), "b": P(B wins), "draw": P(draw), "exact": bool, "nodes": n, "playouts": n, "unsupported": [...]}
import { createInterface } from 'node:readline'
import { solve } from './search'
import { createTwoSidedState } from './vendor/CardRngExpansionDepths/src/engine/battle-v2.label'

const lines = createInterface({ input: process.stdin })
lines.on('line', (line) => {
  let reply: any
  try {
    const { a, b, options } = JSON.parse(line)
    const unsupported = [...createTwoSidedState(a, b).unsupportedAbilities]
    reply = { ...solve(a, b, options || {}), unsupported }
  } catch (error: any) {
    reply = { error: String(error?.message || error) }
  }
  process.stdout.write(JSON.stringify(reply) + '\n')
})
