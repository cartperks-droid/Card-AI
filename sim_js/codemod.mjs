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

// Rule corrections from the user's in-game evidence (docs/simulator_questions.md); his engine is otherwise unchanged.
// Ruler of Humans (Supreme Ozzy) adds 1.25x the HP it actually loses, not the whole hit (user, 2026-10-06: Heaven's
// Armor hit a Dinosaur King Crystal Ozzy with 8,933 HP for 163,840; every ally gained exactly 1.25 x 8,933).
exact("  const appliedHpDamage = Math.min(hpTarget.hp, damage)\n  hpTarget.hp -= appliedHpDamage\n",
      "  const appliedHpDamage = Math.min(hpTarget.hp, damage)\n  hpTarget.hp -= appliedHpDamage\n  ;(target as any).hpTaken = hpTarget === target ? appliedHpDamage : 0\n")
exact("if (alive(ally)) ally.damage += damage * 1.25",
      "if (alive(ally)) ally.damage += ((target as any).hpTaken ?? damage) * 1.25")

// Hades's The Underworld (user, 2026-10-07): it cannot copy Parallax's Paradox, and a fallen Hades offers only its own
// The Underworld, never the ability it copied, so a Hades behind a fallen Hades copies nothing ("two Hades
// consecutively is wrong"). His engine copied both, so Parallax / Hades / Hades / Robin Hood beat floor 105 Impossible
// every time (each Hades a second Parallax). On entry, and when Pandora's Box gains The Underworld. Paradox passed over
// or not makes no difference: Parallax's one activation is global (user).
exact("function resolvePandoraGainedAbility(",
      "function underworldCopy(fallen: CombatCard[], names: (card: CombatCard) => (string | null)[], skip: string[] = []): string | null {\n"
      + "  for (const card of fallen) {\n"
      + "    if ((card.definition as any).ability === 'The Underworld') return null\n"
      + "    const found = names(card).find((name) => name && name !== 'The Underworld' && name !== 'Paradox' && !skip.includes(name))\n"
      + "    if (found) return found\n"
      + "  }\n"
      + "  return null\n"
      + "}\n\n"
      + "// Hades copies a fallen ally's ability when it reaches the front (user, 2026-10-07: a second Hades with Piccolo\n"
      + "// behind it copies Piccolo once Piccolo has swapped in and died; it does not copy every time it goes to the front,\n"
      + "// only when an ally has fallen since its last copy). A Hades coming back with a new fallen ally drops its copied\n"
      + "// ability, so The Underworld copies again and the copy gets its own entry. Other entry abilities fire once.\n"
      + "function underworldReturn(runtime: Runtime) {\n"
      + "  for (const team of ['Allies', 'Enemies'] as BattleTeam[]) {\n"
      + "    const fallen = runtime.state.fallen[team].length\n"
      + "    runtime.state.teams[team].forEach((card, index) => {\n"
      + "      if ((card.definition as any).ability !== 'The Underworld') return\n"
      + "      if (index > 0) { card.flags.underworldFront = false; return }\n"
      + "      if (card.flags.underworldFront) return\n"
      + "      card.flags.underworldFront = true\n"
      + "      if (card.entered && (!alive(card) || card.abilityOverride === null || fallen <= (card.counters.underworldSeen || 0))) return\n"
      + "      card.counters.underworldSeen = fallen\n"
      + "      if (card.entered) { card.abilityOverride = undefined; card.entered = false }\n"
      + "    })\n"
      + "  }\n"
      + "}\n\n"
      + "function resolvePandoraGainedAbility(")
exact("    if (!attacker || !defender) break\n\n    onEntry(runtime, attacker)\n",
      "    if (!attacker || !defender) break\n\n    underworldReturn(runtime)\n    onEntry(runtime, attacker)\n")
exact("    const copied = [...runtime.state.fallen[card.team]].reverse()\n      .flatMap((fallen) => abilityNames(fallen))\n      .find((candidate) => candidate !== 'The Underworld' && candidate !== \"Pandora's Box\")\n",
      "    const copied = underworldCopy([...runtime.state.fallen[card.team]].reverse(), (fallen) => abilityNames(fallen), [\"Pandora's Box\"])\n")
exact("    const copied = [...runtime.state.fallen[card.team]].reverse()\n      .map((fallen) => ability(fallen))\n      .find((candidate) => candidate && candidate !== 'The Underworld')\n",
      "    const copied = underworldCopy([...runtime.state.fallen[card.team]].reverse(), (fallen) => [ability(fallen)])\n")

// Stalemate after 100 turns without a death, not 150 (user, 2026-10-02 Immortal Witch mirror: both cards fell after
// 100 turns, 50 rounds; his turn is one card's action, as the user counted, 2026-10-07).
exact("    if (turnsWithoutDeaths >= 150) {", "    if (turnsWithoutDeaths >= 100) {")
exact("Expansion 150-turn no-progress resolution", "Expansion 100-turn no-progress resolution")

// The runtime carries the chance source; a battle may arrive with a prebuilt (two-sided) state.
exact('  rng: SeededRng\n', '  rng: SeededRng\n  chance: any\n')
// Ability lookups (user: precomputed entries, as a chess engine's lookup tables): an entry holds a card's ability, its
// ability names and their set. A card with no ability or identity override and no bonus abilities reads the shared
// entry at its card index; a card whose abilities changed in battle keeps its own, rebuilt only when what it depends
// on changes (overrides, bonus abilities, definition; bonus lists are only ever replaced, never edited in place). The
// own entry is non-enumerable, so a copied card never inherits it. hasAbility reads entries: same checks, same order.
exact('function ability(card: CombatCard | undefined): string | null {\n  if (!card) return null\n  if (card.abilityOverride !== undefined)',
      'function ability(card: CombatCard | undefined): string | null {\n  return card ? abilityEntry(card).ability : null\n}\n\n'
      + 'function uncachedAbility(card: CombatCard): string | null {\n  if (card.abilityOverride !== undefined)')
exact('function abilityNames(card: CombatCard | undefined): string[] {\n  if (!card) return []\n  return [...new Set([ability(card), ...(card.bonusAbilities || [])].filter((name): name is string => Boolean(name)))]\n}',
      'interface AbilityEntry { override: any; identity: any; bonus: any; definition: any; ability: string | null; names: string[]; set: Set<string> }\n'
      + 'const ENTRY = Symbol(\'abilities\')\n\n'
      + '// By card index (user: entries at the card ID\'s index): the entry of a card with no ability or identity override and\n'
      + '// no bonus abilities depends only on its definition, so it is built once per card and shared by every copy.\n'
      + 'const CARD_INDEX = Symbol(\'index\')\n'
      + 'cards.forEach((card: any, index: number) => Object.defineProperty(card, CARD_INDEX, { value: index }))\n'
      + 'const BASE_ENTRIES: AbilityEntry[] = []\n\n'
      + 'function abilityEntry(card: CombatCard): AbilityEntry {\n'
      + '  const index: number | undefined = (card.definition as any)[CARD_INDEX]\n'
      + '  if (index !== undefined && card.abilityOverride === undefined && card.identityOverride == null && !card.bonusAbilities?.length) {\n'
      + '    return BASE_ENTRIES[index] ??= buildEntry(card)\n  }\n'
      + '  const hit: AbilityEntry | undefined = (card as any)[ENTRY]\n'
      + '  if (hit && hit.override === card.abilityOverride && hit.identity === card.identityOverride && hit.bonus === card.bonusAbilities\n'
      + '      && hit.definition === card.definition) return hit\n'
      + '  const entry = buildEntry(card)\n'
      + '  Object.defineProperty(card, ENTRY, { value: entry, writable: true, configurable: true, enumerable: false })\n'
      + '  return entry\n}\n\n'
      + 'function buildEntry(card: CombatCard): AbilityEntry {\n'
      + '  const own = uncachedAbility(card)\n'
      + '  const names = Object.freeze([...new Set([own, ...(card.bonusAbilities || [])].filter((name): name is string => Boolean(name)))]) as string[]\n'
      + '  return { override: card.abilityOverride, identity: card.identityOverride, bonus: card.bonusAbilities, definition: card.definition,\n'
      + '    ability: own, names, set: new Set(names) }\n}\n\n'
      + 'function abilityNames(card: CombatCard | undefined): string[] {\n  return card ? abilityEntry(card).names : []\n}')
exact("  const ownName = effectiveCardName(card)\n  const opposingName = effectiveCardName(opposingCard)\n  let matched = abilityNames(card).includes(name)\n"
      + "  if (!matched && ability(card) === 'Jealousy' && opposingCard && opposingName !== 'Amenhotep') {\n"
      + "    matched = abilityNames(opposingCard).includes(name)\n  }\n  if (!matched) return false\n"
      + "  if (opposingCard && ability(opposingCard) === 'Jealousy' && ownName !== 'Amenhotep') return false\n"
      + "  const honorActive = [active(runtime, 'Allies'), active(runtime, 'Enemies')].some((activeCard) =>\n"
      + "    activeCard && !activeCard.dead && !activeCard.flags.sealed && abilityNames(activeCard).includes('Honor')\n  )\n"
      + "  if (honorActive && name !== 'Honor') return false\n",
      "  const own = abilityEntry(card)\n  let matched = own.set.has(name)\n"
      + "  if (!matched && own.ability === 'Jealousy' && opposingCard && effectiveCardName(opposingCard) !== 'Amenhotep') {\n"
      + "    matched = abilityEntry(opposingCard).set.has(name)\n  }\n  if (!matched) return false\n"
      + "  if (opposingCard && abilityEntry(opposingCard).ability === 'Jealousy' && effectiveCardName(card) !== 'Amenhotep') return false\n"
      + "  if (name !== 'Honor') {\n    for (const activeCard of [active(runtime, 'Allies'), active(runtime, 'Enemies')]) {\n"
      + "      if (activeCard && !activeCard.dead && !activeCard.flags.sealed && abilityEntry(activeCard).set.has('Honor')) return false\n    }\n  }\n")

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
// The C engine (sim_c/gen_tables.ts) builds its tables from the same pools and sets this file uses.
src += '\nexport { FULLY_SUPPORTED, PANDORA_ABILITY_POOL, RANDOM_CARD_POOL, NUWA_CREATABLE_POOL, CONSTELLAR_ABILITIES, DODGE_ABILITIES }\n'

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
