// Label worker: one JSON request per input line, one JSON result per output line, in order.
//   battle:  {"a": TeamLoadout, "b": TeamLoadout, "options": {...}}   (side A moves first)
//            -> {"a": P(A wins), "b": P(B wins), "draw": P(draw), "exact", "nodes", "playouts", "unsupported": [...]}
//   initial: {"op": "initial", "a": ..., "b": ..., "options": tweaks} -> {"a": [[hp, attack, ability] per card], "b": [...]}
//            at the start of battle
//   tables:  {"op": "tables", "cards": [names], "borders": [[border names] per border id], "mutations": [names],
//             "reds": [aura names], "tiers": [aura border or null per tier]}
//            -> {"base": [card][border][mutation] = [hp, attack], "red": [card][mutation][red][tier] = [hp x, attack x],
//                "prehistoric": [per card], "jurassic": [Jurassic World % per Prehistoric card, per tier]}
//   check:   {"op": "check", "a": ..., "b": ...} -> {"unsupported": [...]} without running the battle
import { createInterface } from 'node:readline'
import { solve, startState } from './search'
import { createTwoSidedState } from './vendor/CardRngExpansionDepths/src/engine/battle-v2.label'
import { getAttack, getHealth } from './vendor/CardRngExpansionDepths/src/engine/stats'
import { getAura, getSkillAuraValue, statAuraPercentForCard } from './vendor/CardRngExpansionDepths/src/engine/auras.label'
import cards from './vendor/CardRngExpansionDepths/src/data/cards'

const byName = new Map(cards.map((card: any) => [card.name, card]))

function tables(req: any) {
  const defs = req.cards.map((name: string) => byName.get(name))
  const base = defs.map((d: any) => req.borders.map((b: any) => req.mutations.map((m: string | null) =>
    [getHealth(d, b, m as any), getAttack(d, b, m as any)])))
  const red = defs.map((d: any) => req.mutations.map((m: string | null) => req.reds.map((name: string) => {
    const aura = getAura(name)
    return req.tiers.map((border: any) => {
      const value = statAuraPercentForCard(aura as any, { definition: d, mutationWeather: m } as any, border)
      return [1 + value / 100, name === 'General Sun Tzu' ? 1 : 1 + value / 100]
    })
  })))
  const prehistoric = defs.map((d: any) => d.pack === 'Prehistoric')
  const jurassic = req.tiers.map((border: any) => getSkillAuraValue(getAura('Jurassic World') as any, border))
  return { base, red, prehistoric, jurassic }
}

const lines = createInterface({ input: process.stdin })
lines.on('line', (line) => {
  let reply: any
  try {
    const req = JSON.parse(line)
    if (req.op === 'tables') reply = tables(req)
    else if (req.op === 'initial') {
      const state = startState(req.a, req.b, req.options || {})
      const cards = (team: any[]) => team.map((c: any) => [c.hp, c.damage, c.abilityOverride === undefined ? c.definition.ability : c.abilityOverride])
      reply = { a: cards(state.teams.Allies), b: cards(state.teams.Enemies) }
    } else if (req.op === 'check') {
      reply = { unsupported: [...createTwoSidedState(req.a, req.b).unsupportedAbilities] }
    } else {
      const unsupported = [...createTwoSidedState(req.a, req.b).unsupportedAbilities]
      reply = { ...solve(req.a, req.b, req.options || {}), unsupported }
    }
  } catch (error: any) {
    reply = { error: String(error?.message || error) }
  }
  process.stdout.write(JSON.stringify(reply) + '\n')
})
