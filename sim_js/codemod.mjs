// Rewrites DaddyDrago's battle engine (src/engine/battle-v2.ts) so every random draw is an explicit chance point
// the label search can branch on, and battles can give both sides supports. His logic is otherwise unchanged.
//   rand(runtime, team) <cmp> T             -> runtime.chance.roll(runtime, team, '<cmp>', T)
//   Math.floor|ceil(rand(runtime, team) * N) -> runtime.chance.pickRoll(runtime, team, N) [+ 1]
//   runtime.rng.next() <cmp> T, * 100 < T    -> runtime.chance.raw('<cmp>', T [/ 100])
//   Math.floor(runtime.rng.next() * N)       -> runtime.chance.pick(N)
// roll/pickRoll keep rand()'s rules (Unlucky and noRng force 0; Fate rerolls), so the probabilities are exact.
// Four sites that keep a roll in a variable are rewritten by exact text and fail loudly if his code changes.
// Supports: his auras.ts has no Ruby border; auras.label.ts (written next to the output) adds it (see below).
// Usage: node codemod.mjs <his battle-v2.ts> <output .ts in the same directory>
import { readFileSync, writeFileSync } from 'node:fs'

const [input, output] = process.argv.slice(2)
let src = readFileSync(input, 'utf8')
const counts = {}
const exact = (from, to) => {
  if (src.split(from).length !== 2) throw new Error(`codemod: expected exactly one occurrence of:\n${from}`)
  src = src.replace(from, to)
}
// rand() itself stays as written (its rules are mirrored in chance.ts); protect it from the rewrites below
const randStart = src.indexOf('function rand(runtime: Runtime, team: BattleTeam): number {')
const randEnd = src.indexOf('\n}\n', randStart) + 3
if (randStart < 0 || randEnd < 3) throw new Error('codemod: rand() not found')
const randBody = src.slice(randStart, randEnd)
src = src.slice(0, randStart) + '/*RAND*/' + src.slice(randEnd)

exact("      const roll = rand(runtime, attacker.team)\n      const change = changes[Math.max(0, Math.min(2, Math.ceil(roll * 3) - 1))]",
      "      const change = changes[runtime.chance.pickRoll(runtime, attacker.team, 3)]")
exact("      const roll = rand(runtime, target.team)\n      const dodged = roll > 0.4",
      "      const dodged = runtime.chance.roll(runtime, target.team, '>', 0.4)\n      const roll = dodged ? 1 : 0")
exact("      const roll = rand(runtime, target.team)\n      const dodged = roll > 0.6",
      "      const dodged = runtime.chance.roll(runtime, target.team, '>', 0.6)\n      const roll = dodged ? 1 : 0")
exact("    const roll = rand(runtime, team)\n    if (roll > 0.5) {",
      "    const roll = runtime.chance.roll(runtime, team, '>', 0.5) ? 1 : 0\n    if (roll > 0.5) {")

const sub = (name, pattern, replace) => {
  src = src.replace(pattern, (...m) => { counts[name] = (counts[name] || 0) + 1; return replace(...m) })
}
const R = String.raw`rand\(runtime, ([^()]+?)\)`
const N = String.raw`runtime\.rng\.next\(\)`
const T = String.raw`([^)&|;{}?:\n]+?)`
sub('pickRoll', new RegExp(String.raw`Math\.floor\(${R} \* ([^)]+?)\)`, 'g'), (_, team, n) => `runtime.chance.pickRoll(runtime, ${team}, ${n})`)
sub('pickRoll+1', new RegExp(String.raw`Math\.ceil\(${R} \* ([^)]+?)\)`, 'g'), (_, team, n) => `(runtime.chance.pickRoll(runtime, ${team}, ${n}) + 1)`)
sub('pick', new RegExp(String.raw`Math\.floor\(${N} \* \(([^()]+)\)\)`, 'g'), (_, n) => `runtime.chance.pick(${n})`)
sub('pick', new RegExp(String.raw`Math\.floor\(${N} \* ([^)]+?)\)`, 'g'), (_, n) => `runtime.chance.pick(${n})`)
sub('roll', new RegExp(String.raw`${R} (<=|>=|<|>) ${T}(?=\s*(?:\)|&&|\|\||\?)|[ \t]*\n)`, 'g'), (_, team, cmp, t) => `runtime.chance.roll(runtime, ${team}, '${cmp}', ${t.trim()})`)
sub('raw%', new RegExp(String.raw`${N} \* 100 (<=|>=|<|>) ${T}(?=\s*(?:\)|&&|\|\||\?)|[ \t]*\n)`, 'g'), (_, cmp, t) => `runtime.chance.raw('${cmp}', (${t.trim()}) / 100)`)
sub('raw', new RegExp(String.raw`${N} (<=|>=|<|>) ${T}(?=\s*(?:\)|&&|\|\||\?)|[ \t]*\n)`, 'g'), (_, cmp, t) => `runtime.chance.raw('${cmp}', ${t.trim()})`)
const leftovers = (src.match(new RegExp(`${R}|${N}`, 'g')) || [])
if (leftovers.length) throw new Error(`codemod: ${leftovers.length} random draws left unconverted: ${leftovers.join(', ')}`)
src = src.replace('/*RAND*/', randBody)

// The runtime carries the chance source; a battle may arrive with a prebuilt (two-sided) state.
exact('  rng: SeededRng\n', '  rng: SeededRng\n  chance: any\n')
// Ability lookups (user: hash the index): a card's ability list depends only on its ability override, identity
// override and bonus abilities (bonus lists are only ever replaced, never edited in place), so it is cached per card
// and rebuilt only when one of those changes. The list itself is computed exactly as before.
exact('function abilityNames(card: CombatCard | undefined): string[] {\n  if (!card) return []\n  return [',
      'const ABILITY_CACHE = new WeakMap<CombatCard, { override: any; identity: any; bonus: any; names: string[] }>()\n\n'
      + 'function abilityNames(card: CombatCard | undefined): string[] {\n  if (!card) return []\n'
      + '  const hit = ABILITY_CACHE.get(card)\n'
      + '  if (hit && hit.override === card.abilityOverride && hit.identity === card.identityOverride && hit.bonus === card.bonusAbilities) return hit.names\n'
      + '  const names = Object.freeze(uncachedAbilityNames(card)) as string[]\n'
      + '  ABILITY_CACHE.set(card, { override: card.abilityOverride, identity: card.identityOverride, bonus: card.bonusAbilities, names })\n'
      + '  return names\n}\n\nfunction uncachedAbilityNames(card: CombatCard): string[] {\n  return [')

// The search saves a battle at the start of each turn (onTurn: the state plus the loop's own counters) and later
// resumes it from there (resume), so expanding a branch continues the battle instead of replaying it from turn 1.
exact('  onProgress?: (turn: number) => void,\n): BattleResult {\n  const state = createBattleStateV2(loadout, enemies)',
      '  onProgress?: (turn: number) => void,\n  inject?: { state?: BattleState; chance?: any; resume?: TurnCounters; onTurn?: (counters: TurnCounters) => void },\n'
      + '): BattleResult {\n  const state = inject?.state ?? createBattleStateV2(loadout, enemies)')
exact('  const runtime: Runtime = { state, rng: new SeededRng(seed), debug, captureDebug, deathEpoch: 0 }\n  resolveConstellarArts(runtime)',
      '  const rng = new SeededRng(seed)\n  const runtime: Runtime = { state, rng, chance: inject?.chance ?? sampledChance(rng), debug, captureDebug, '
      + 'deathEpoch: inject?.resume?.deathEpoch ?? 0 }\n  if (!inject?.resume) resolveConstellarArts(runtime)')
exact('  let turnsWithoutDeaths = 0\n  let lastDeathEpoch = runtime.deathEpoch\n',
      '  let turnsWithoutDeaths = inject?.resume?.turnsWithoutDeaths ?? 0\n  let lastDeathEpoch = inject?.resume?.lastDeathEpoch ?? runtime.deathEpoch\n')
exact('  while (state.teams.Allies.length && state.teams.Enemies.length && state.turn < maxTurns) {\n    state.turn += 1\n',
      '  while (state.teams.Allies.length && state.teams.Enemies.length && state.turn < maxTurns) {\n'
      + '    inject?.onTurn?.({ turnsWithoutDeaths, lastDeathEpoch, deathEpoch: runtime.deathEpoch })\n    state.turn += 1\n')
src += '\nexport interface TurnCounters { turnsWithoutDeaths: number; lastDeathEpoch: number; deathEpoch: number }\n'

// Two-sided battles: both teams are player teams with their own red (stat) and blue (skill) auras; side A moves first.
src += `
export function rollContext(runtime: Runtime, team: BattleTeam): { zero: boolean; fate: number } {
  const activeA = runtime.state.teams.Allies[0]
  const activeE = runtime.state.teams.Enemies[0]
  const zero = hasAbility(runtime, activeA, 'Unlucky') || hasAbility(runtime, activeE, 'Unlucky')
    || Boolean(runtime.state.teams[team][0]?.flags.noRng)
  return { zero, fate: (runtime.state.boosts[team].fate || 0) / 100 }
}

export function sampledChance(rng: SeededRng) {
  const cdf = (x: number, fate: number) => { x = Math.min(1, Math.max(0, x)); return (1 - fate) * x + fate * x * x }
  const holds = (value: number, cmp: string, t: number) => cmp === '<' ? value < t : cmp === '<=' ? value <= t : cmp === '>' ? value > t : value >= t
  return {
    roll(runtime: Runtime, team: BattleTeam, cmp: string, t: number) {
      const ctx = rollContext(runtime, team)
      if (ctx.zero) return holds(0, cmp, t)
      const below = cdf(t, ctx.fate)
      return rng.next() < (cmp[0] === '<' ? below : 1 - below)
    },
    pickRoll(runtime: Runtime, team: BattleTeam, n: number) {
      const ctx = rollContext(runtime, team)
      if (ctx.zero) return 0
      const u = rng.next()
      for (let k = 0; k < n; k++) if (u < cdf((k + 1) / n, ctx.fate)) return k
      return n - 1
    },
    raw(cmp: string, t: number) { return rng.next() < (cmp[0] === '<' ? Math.min(1, Math.max(0, t)) : 1 - Math.min(1, Math.max(0, t))) },
    pick(n: number) { return Math.floor(rng.next() * n) },
  }
}

export function createTwoSidedState(a: TeamLoadout, b: TeamLoadout): BattleState {
  const state = createBattleStateV2(a, [])
  const enemies = b.cards
    .map((slot, index) => makePlayerCard(slot.cardName, slot.borders, index + 1, slot.mutationWeather))
    .filter((card): card is CombatCard => Boolean(card))
  for (const card of enemies) { card.team = 'Enemies'; card.id = card.id.replace(/^Allies:/, 'Enemies:') }
  state.teams.Enemies = enemies
  applyDeckPassives(enemies)
  applyDraconianSetup(enemies)
  const stat = applyStatAura(enemies, b.statAura)
  const skillTeam = applySkillAuraTeamEffects(enemies, b.abilityAura)
  const skill = buildSkillAuraBoosts(b.abilityAura)
  state.boosts.Enemies = { fossils: 0, ...skill.boosts }
  if (skill.aura && !skill.implemented) state.unsupportedAbilities.add(\`Aura: \${skill.aura.name}\`)
  if (skillTeam.aura && !skillTeam.implemented) state.unsupportedAbilities.add(\`Aura: \${skillTeam.aura.name}\`)
  if (stat.aura) { state.boosts.Enemies.statAuraName = stat.aura.name; state.boosts.Enemies.statAuraValue = stat.value }
  const composers = enemies.filter((card) => card.definition.ability === 'Nightmare Melody').length
  if (composers > 0) { state.boosts.Enemies.composerCount = composers; state.boosts.Enemies.composerThreshold = 1 }
  for (const card of enemies) {
    card.counters.normalDamage = card.damage
    card.counters.normalMaxHp = card.maxHp
    if (BENCH_AFFECTING_UNSUPPORTED.has(ability(card) || '')) noteUnsupported(state, card)
  }
  return state
}
`
exact("from './auras'", "from './auras.label'")
writeFileSync(output, src)

// Ruby support cards (user's binder, IMG_0350-0354). Red: his formula floor(2^log10(rarity x multiplier) / 2) with a
// Ruby multiplier of 500 gives every value read (user: The One Ring 256%, Dinosaur King 207% / 334% boosted). Blue: Ruby uses the base values (user).
const aurasPath = input.replace(/battle-v2\.ts$/, 'auras.ts')
src = readFileSync(aurasPath, 'utf8')
exact("  Crystal: 100,\n  Galaxy: 1_000,\n}", "  Crystal: 100,\n  Ruby: 500,\n  Galaxy: 1_000,\n}")
exact("  Crystal: 2,\n  Galaxy: 3,\n}", "  Crystal: 2,\n  Ruby: 0,\n  Galaxy: 3,\n}")
writeFileSync(aurasPath.replace(/\.ts$/, '.label.ts'), src)
console.error(JSON.stringify(counts))
