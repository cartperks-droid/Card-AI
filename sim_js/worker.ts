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
//   depths:  {"op": "depths"} -> {"pool": [[name, weight, ATK x HP] per card Depths can field], "legacyBans": [...],
//            "maxBans"}: his Depths pool (depths.ts: eligibility and weather weights), a card unlocking once its
//            ATK x HP is below the floor's budget (card_engine/depths.py)
//   state:   {"op": "state", "a": ..., "b": ..., "options": tweaks} -> the battle as it starts, for the C engine
//            (card_engine/simulator/kernel.py): {"cards": [...], "lists": [Allies, Enemies, fallen Allies, fallen
//            Enemies as card positions], "boosts": [Allies, Enemies], "turn", "moving", "unsupported"}
import { createInterface } from 'node:readline'
import { solve, startState } from './search'
import { createTwoSidedState } from './vendor/CardRngExpansionDepths/src/engine/battle-v2.label'
import { getAttack, getHealth } from './vendor/CardRngExpansionDepths/src/engine/stats'
import { getAura, getSkillAuraValue, statAuraPercentForCard } from './vendor/CardRngExpansionDepths/src/engine/auras.label'
import { depthsMechanics, getDepthsPool } from './vendor/CardRngExpansionDepths/src/engine/depths'
import cards from './vendor/CardRngExpansionDepths/src/data/cards'

const byName = new Map(cards.map((card: any) => [card.name, card]))
const TEAMS = ['Allies', 'Enemies']
const BORDER_ORDER = ['Galaxy', 'Ruby', 'Crystal', 'Platinum']
const CARD_KEYS = new Set(['id', 'definition', 'team', 'index', 'borders', 'mutationWeather', 'power', 'hp', 'maxHp', 'damage',
  'entered', 'dead', 'boss', 'identityOverride', 'abilityOverride', 'bonusAbilities', 'status', 'flags', 'counters'])

// Every card once (shared references stay shared), with names for the definition, identity and abilities; kernel.py
// maps names to the C engine's ids and refuses any field it does not know.
function exportState(state: any) {
  const position = new Map<any, number>()
  const out: any[] = []
  const add = (card: any) => {
    if (position.has(card)) return position.get(card)!
    for (const key in card) if (!CARD_KEYS.has(key)) throw new Error(`card field ${key} has no C counterpart`)
    const entry: any = {
      id: card.id, def: card.definition.name, team: TEAMS.indexOf(card.team), index: card.index,
      borders: BORDER_ORDER.filter((b) => card.borders.includes(b)), power: card.power, hp: card.hp, maxHp: card.maxHp,
      damage: card.damage, entered: card.entered, dead: card.dead, boss: card.boss, identity: card.identityOverride ?? null,
      bonus: card.bonusAbilities || [], status: card.status, flags: card.flags, counters: card.counters,
    }
    if (card.abilityOverride !== undefined) entry.abilityOverride = card.abilityOverride  // absent: never set
    position.set(card, out.length)
    out.push(entry)
    return out.length - 1
  }
  const lists = [...TEAMS.map((t) => state.teams[t]), ...TEAMS.map((t) => state.fallen[t])].map((cards: any[]) => cards.map(add))
  return { cards: out, lists, boosts: TEAMS.map((t) => state.boosts[t]), turn: state.turn, moving: TEAMS.indexOf(state.moving),
    unsupported: [...state.unsupportedAbilities] }
}

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
    } else if (req.op === 'state') {
      reply = exportState(startState(req.a, req.b, req.options || {}))
    } else if (req.op === 'depths') {
      const every = getDepthsPool(Number.MAX_SAFE_INTEGER)  // every eligible card has unlocked by then
      reply = { pool: every.map(({ card, weight }: any) => [card.name, weight, getAttack(card) * getHealth(card)]),
        legacyBans: depthsMechanics.legacyHardExclusions, maxBans: depthsMechanics.maxPlayerBans }
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
