// Chance-tree label search around DaddyDrago's battle engine (patched by codemod.mjs into battle-v2.label.ts).
// Best-first: the most probable open branch is expanded next. A branch is the battle saved at the start of the turn
// in which it split, plus the choices made at the engine's chance points since then; expanding it restores that
// turn and plays on, as the old C kernel continued from stored states (user: same search, different mechanics).
// Branches below sampleBelow, and whatever is open when the node budget runs out, are resolved together by playouts
// (also started from their saved turns) until the A-win estimate's standard error is under rolloutError.
// Side A moves first. Outcomes: A wins, B wins, draw (both wiped out, or the turn cap).
import { createTwoSidedState, rollContext, simulateBattleV2 } from './vendor/CardRngExpansionDepths/src/engine/battle-v2.label'
import type { TurnCounters } from './vendor/CardRngExpansionDepths/src/engine/battle-v2.label'

// Per-battle tweaks, each on side 'a' or 'b': fixed (HP, ATK) per card (event or Tower teams), stats scaled by a factor
// (counter-search stress test), or one card's ability removed with its stats kept.
export interface Tweaks { fixed?: [Side, number[][]]; scale?: [Side, number]; strip?: [Side, number] }
type Side = 'a' | 'b'
export interface Options extends Tweaks { nodeBudget: number; sampleBelow: number; rollouts: number; rolloutError: number
  maxTurns: number; seed: number; snapshots: boolean }  // snapshots false: replay every branch from turn 1 (tests)
export const DEFAULTS: Options = { nodeBudget: 20000, sampleBelow: 1e-3, rollouts: 1024, rolloutError: 0.03, maxTurns: 2000, seed: 1,
  snapshots: true }
export interface Result { a: number; b: number; draw: number; exact: boolean; nodes: number; playouts: number }

// A battle saved at the start of a turn (never mutated: resuming works on a copy) with the battle loop's counters.
interface Saved { state: any; counters: TurnCounters }
interface Node { p: number; saved: Saved | null; choices: number[] }  // choices made since `saved` (or turn 1)

class Branch { saved: Saved | null = null; choices: number[] = []; constructor(public probs: number[]) {} }


// Copy of a battle state, shaped like his BattleState: cards (team members and fallen; each copied once, so shared
// references stay shared), boosts (plain values) and the unsupported-ability set. Card definitions are static game
// data and stay shared, as in his engine. Anything else falls back to a generic deep copy.
function copyValue(value: any, seen: Map<any, any>): any {
  if (value === null || typeof value !== 'object') return value
  if (seen.has(value)) return seen.get(value)
  if (value instanceof Set) { const out = new Set(); seen.set(value, out); value.forEach((v) => out.add(copyValue(v, seen))); return out }
  if (value instanceof Map) { const out = new Map(); seen.set(value, out); value.forEach((v, k) => out.set(k, copyValue(v, seen))); return out }
  if (Array.isArray(value)) { const out: any[] = []; seen.set(value, out); for (const v of value) out.push(copyValue(v, seen)); return out }
  const out: any = {}
  seen.set(value, out)
  for (const key in value) out[key] = key === 'definition' ? value[key] : copyValue(value[key], seen)
  return out
}

// A card: status, flags and counters hold plain values; the definition is static and bonus-ability lists are only ever
// replaced (never edited in place), so both stay shared. Any other object field gets the generic copy.
const PLAIN = new Set(['status', 'flags', 'counters'])
const SHARED = new Set(['definition', 'bonusAbilities', 'borders'])

function copyCard(card: any, seen: Map<any, any>): any {
  let out = seen.get(card)
  if (out) return out
  out = { ...card }
  seen.set(card, out)
  for (const key in out) {
    const value = out[key]
    if (value === null || typeof value !== 'object' || SHARED.has(key)) continue
    out[key] = PLAIN.has(key) ? { ...value } : copyValue(value, seen)
  }
  return out
}

function copyState(state: any): any {
  const seen = new Map<any, any>()
  const team = (cards: any[]) => cards.map((card) => copyCard(card, seen))
  const out: any = {}
  for (const key in state) {
    const value = state[key]
    out[key] = key === 'teams' || key === 'fallen' ? { Allies: team(value.Allies), Enemies: team(value.Enemies) }
      : key === 'boosts' ? { Allies: { ...value.Allies }, Enemies: { ...value.Enemies } }
      : copyValue(value, seen)
  }
  return out
}

function mulberry(seed: number) {
  let s = seed >>> 0
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), s | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }
}

class Tape {
  i = 0
  constructor(private choices: number[], private random: (() => number) | null) {}
  draw(probs: number[]): number {
    let only = -1, open = 0
    for (let k = 0; k < probs.length; k++) if (probs[k] > 0) { open++; only = k }
    if (open <= 1) return Math.max(0, only)           // forced outcome: not a chance point
    if (this.i < this.choices.length) return this.choices[this.i++]
    if (!this.random) throw new Branch(probs)
    let u = this.random(), k = 0
    for (; k < probs.length - 1; k++) { if (u < probs[k]) break; u -= probs[k] }
    this.i++
    return k
  }
}

const cdf = (x: number, fate: number) => { x = Math.min(1, Math.max(0, x)); return (1 - fate) * x + fate * x * x }
const holds = (v: number, cmp: string, t: number) => cmp === '<' ? v < t : cmp === '<=' ? v <= t : cmp === '>' ? v > t : v >= t

function chanceFor(tape: Tape) {
  return {
    roll(runtime: any, team: any, cmp: string, t: number) {
      const ctx = rollContext(runtime, team)
      if (ctx.zero) return holds(0, cmp, t)
      const below = cdf(t, ctx.fate), p = cmp[0] === '<' ? below : 1 - below
      return tape.draw([p, 1 - p]) === 0
    },
    pickRoll(runtime: any, team: any, n: number) {
      const ctx = rollContext(runtime, team)
      if (ctx.zero || n <= 1) return 0
      const probs = Array.from({ length: n }, (_, k) => cdf((k + 1) / n, ctx.fate) - cdf(k / n, ctx.fate))
      return tape.draw(probs)
    },
    raw(cmp: string, t: number) {
      const below = Math.min(1, Math.max(0, t)), p = cmp[0] === '<' ? below : 1 - below
      return tape.draw([p, 1 - p]) === 0
    },
    pick(n: number) { return n <= 1 ? 0 : tape.draw(Array(n).fill(1 / n)) },
  }
}

export function startState(a: any, b: any, tweaks: Tweaks = {}) {
  const state = createTwoSidedState(a, b)
  const team = (side: Side) => state.teams[side === 'a' ? 'Allies' : 'Enemies']
  // Astraeus: each art is its own card with a fixed Constellar ability (user); his engine only draws one when none is set.
  for (const [side, loadout] of [['a', a], ['b', b]] as [Side, any][]) {
    const arts = loadout.cards.filter((entry: any) => entry.constellar).map((entry: any) => entry.constellar)
    team(side).filter((card: any) => card.definition.name === 'Astraeus').forEach((card: any, i: number) => {
      if (arts[i]) card.abilityOverride = arts[i]
    })
  }
  const set = (card: any, hp: number, attack: number) => {
    card.hp = card.maxHp = hp
    card.damage = attack
    card.counters.normalDamage = attack
    card.counters.normalMaxHp = hp
  }
  // A fixed card's power follows his enemies' rule, ATK = power / 2 (tower.ts buildTowerEnemies): Beyond The Grave and
  // Creation and Restoration rebuild cards from power (crosscheck 2026-10-04: Anubis at 19M HP revived at his normal power).
  if (tweaks.fixed) tweaks.fixed[1].forEach(([hp, attack], i) => {
    const card = team(tweaks.fixed![0])[i]
    if (card) { set(card, hp, attack); card.power = 2 * attack }
  })
  if (tweaks.scale) for (const card of team(tweaks.scale[0])) set(card, card.maxHp * tweaks.scale[1], card.damage * tweaks.scale[1])
  if (tweaks.strip) {
    const card = team(tweaks.strip[0])[tweaks.strip[1]]
    if (card) { card.abilityOverride = null; card.bonusAbilities = [] }
  }
  return state
}

function play(a: any, b: any, node: Node, random: (() => number) | null, o: Options): 'a' | 'b' | 'draw' {
  const tape = new Tape(node.choices, random)
  const resume = o.snapshots ? node.saved : null
  const state = resume ? copyState(resume.state) : startState(a, b, o)
  let saved = resume, from = 0  // the latest saved turn and the tape position when it was saved
  const onTurn = o.snapshots && !random
    ? (counters: TurnCounters) => { saved = { state: copyState(state), counters: { ...counters } }; from = tape.i }
    : undefined
  try {
    const r = simulateBattleV2(a, [], 1, o.maxTurns, false, false, undefined,
      { state, chance: chanceFor(tape), resume: resume?.counters, onTurn })
    return r.winner === 'Allies' ? 'a' : r.winner === 'Enemies' ? 'b' : 'draw'
  } catch (e) {
    if (e instanceof Branch) { e.saved = saved; e.choices = node.choices.slice(from) }
    throw e
  }
}

export function solve(a: any, b: any, options: Partial<Options> = {}): Result {
  const o = { ...DEFAULTS, ...options }
  const out = { a: 0, b: 0, draw: 0 }
  const open: Node[] = [{ p: 1, saved: null, choices: [] }]
  const rare: Node[] = []
  let nodes = 0
  while (open.length && nodes < o.nodeBudget) {
    let best = 0
    for (let i = 1; i < open.length; i++) if (open[i].p > open[best].p) best = i
    const node = open[best]; open[best] = open[open.length - 1]; open.pop()
    nodes++
    try { out[play(a, b, node, null, o)] += node.p }
    catch (e) {
      if (!(e instanceof Branch)) throw e
      e.probs.forEach((q, k) => {
        if (q > 0) (node.p * q < o.sampleBelow ? rare : open).push({ p: node.p * q, saved: e.saved, choices: [...e.choices, k] })
      })
    }
  }
  const left = [...open, ...rare]
  const mass = left.reduce((s, n) => s + n.p, 0)
  if (!left.length || mass <= 0) return { ...out, exact: true, nodes, playouts: 0 }
  // pooled playouts: each picks an open branch in proportion to its probability, then plays on at random
  const random = mulberry(o.seed), cum: number[] = []
  left.reduce((s, n) => { cum.push(s + n.p); return s + n.p }, 0)
  const tally = { a: 0, b: 0, draw: 0 }
  let n = 0
  while (n < o.rollouts) {
    for (let j = 0; j < 32 && n < o.rollouts; j++, n++) {
      const u = random() * mass
      let lo = 0, hi = cum.length - 1
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < u) lo = mid + 1; else hi = mid }
      tally[play(a, b, left[lo], random, o)]++
    }
    const pa = tally.a / n
    if (mass * Math.sqrt(Math.max(pa * (1 - pa), 1 / n) / n) <= o.rolloutError) break
  }
  return { a: out.a + mass * tally.a / n, b: out.b + mass * tally.b / n, draw: out.draw + mass * tally.draw / n, exact: false, nodes, playouts: n }
}
