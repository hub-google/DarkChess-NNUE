/** Self-play using EXACTLY the same DarkChessEngine class as the browser.
 * Production, arena and browser use this identical decision core. Replay
 * legality is checked independently using the Python referee.
 */
import { randomUUID } from 'node:crypto'
import { writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { gzipSync } from 'node:zlib'
import { Board, Color, type Move } from '../../frontend/src/engine/board'
import { DarkChessEngine, type SearchAnalysis, type SearchLimits } from '../../frontend/src/engine/search'
import { loadNNUE } from './model_runtime'

type Replay = {
  id: string; ts: number; ver: string; model: string; hid: number[]
  mov: number[]; q: number[]; v: number[]; qd: number[]
  pi: [number, number][][]; qn: number[]; qt: number[]; res: number; ply: number; draw_reason?: string
  red_model?: string; black_model?: string
}
const VERSION = 'v2.1.0-perpetual-chase'
const num = (key: string, defaultValue: number) => {
  const value = Number(process.env[key] ?? defaultValue)
  if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid ${key}=${value}`)
  return value
}
const encode = (move: Move) => move.from * 32 + move.to
const nHidden = (board: Board) => board.remainingCounts.reduce((a, b) => a + b, 0)

function limits(board: Board): SearchLimits {
  const hidden = nHidden(board)
  return {
    maxDepth: hidden >= 24 ? 5 : hidden >= 12 ? 10 : 12,
    nodeBudget: hidden >= 24 ? num('SEARCH_NODE_BUDGET_EARLY', 6000) :
      hidden >= 12 ? num('SEARCH_NODE_BUDGET_MID', 12000) :
      num('SEARCH_NODE_BUDGET_LATE', 25000),
    maxMs: num('SEARCH_MAX_SECONDS', 3) * 1000,
  }
}

function rankedPolicy(analysis: SearchAnalysis, side: Color): [number, number][] {
  if (!analysis.depth) return []
  const sign = side === Color.RED ? 1 : -1
  const entries = Object.entries(analysis.rootValues)
    .map(([action, score]) => [Number(action), sign * score] as const)
    .sort((a, b) => b[1] - a[1]).slice(0, 16)
  if (!entries.length) return []
  const maximum = entries[0][1]
  const weights = entries.map(([, score]) => Math.exp((score - maximum) / 0.25))
  const total = weights.reduce((a, b) => a + b, 0)
  return entries.map(([action], i) => [action, weights[i] / total])
}

/** Limited exploration ONLY among actions whose public-search value is near best. */
function select(analysis: SearchAnalysis, side: Color, ply: number): Move {
  const best = analysis.move
  if (!best) throw new Error('No legal self-play move')
  if (!analysis.depth || ply >= num('EXPLORE_PLIES', 12)) return best
  const sign = side === Color.RED ? 1 : -1
  const vals = Object.entries(analysis.rootValues)
    .map(([id, score]) => ({ id: Number(id), value: sign * score }))
    .sort((a, b) => b.value - a.value)
  if (!vals.length) return best
  const close = vals.filter(x => vals[0].value - x.value <= num('EXPLORE_VALUE_MARGIN', 0.04))
  if (close.length < 2) return best
  const temperature = Math.max(0.001, num('SELF_PLAY_TEMPERATURE', 0.25))
  const weights = close.map(x => Math.exp((x.value - close[0].value) / temperature))
  let threshold = Math.random() * weights.reduce((a, b) => a + b, 0)
  for (let i = 0; i < close.length; i++) {
    threshold -= weights[i]
    if (threshold <= 0) return { from: close[i].id >> 5, to: close[i].id & 31, flip: (close[i].id >> 5) === (close[i].id & 31) }
  }
  return best
}

function play(engine: DarkChessEngine, version: string): {
  replay: Replay; seconds: number; totalNodes: number; depthAverage: number;
  fallback: number; earlyCount: number; lateDepthAverage: number; lateCount: number
} {
  const board = new Board()
  const id = randomUUID()
  const started = performance.now()
  const record: Replay = {
    id, ts: Date.now(), ver: VERSION, model: version,
    hid: board.hidden.map(x => Number(x) - 1),
    mov: [], q: [], v: [], qd: [], pi: [], qn: [], qt: [], res: 0, ply: 0,
    red_model: version, black_model: version,
  }
  let nodes = 0, completedDepth = 0, fallbacks = 0, searched = 0
  let lateDepth = 0, lateCount = 0, earlyCount = 0
  for (let ply = 0; ply < 512 && !board.isOver; ply++) {
    const hidden = nHidden(board)
    const staticValue = engine.evaluate(board)
    const analysis = engine.analyze(board, limits(board))
    if (!analysis.move) throw new Error('No legal move in non-terminal board')
    const side = board.turn
    const chosen = side === Color.NONE ? analysis.move : select(analysis, side, ply)
    const rawValue = analysis.rootValues[encode(chosen)] ?? analysis.value
    record.mov.push(encode(chosen))
    record.q.push(rawValue)
    record.v.push(staticValue)
    record.qd.push(analysis.depth)
    record.qn.push(analysis.nodes)
    record.qt.push(Math.round(analysis.elapsedMs))
    record.pi.push(side === Color.NONE ? [] : rankedPolicy(analysis, side))
    if (!board.play(chosen)) throw new Error(`Illegal move in shared engine: ${encode(chosen)}`)
    nodes += analysis.nodes
    if (side !== Color.NONE) {
      searched++
      completedDepth += analysis.depth
      if (!analysis.depth) fallbacks++
      if (hidden >= 24) earlyCount++
      if (hidden < 12) { lateDepth += analysis.depth; lateCount++ }
    }
  }
  if (!board.isOver) throw new Error('Unfinished self-play game at 512 plies: no fabricated draw')
  record.ply = record.mov.length
  record.res = board.draw ? 0 : board.winner === Color.RED ? 1 : -1
  if (board.draw) record.draw_reason = board.halfMoveClock >= 60 ? 'sixty_no_progress' : 'other'
  const depthAverage = completedDepth / Math.max(1, searched)
  const lateDepthAverage = lateDepth / Math.max(1, lateCount)
  return { replay: record, seconds: (performance.now() - started) / 1000,
    totalNodes: nodes, depthAverage, fallback: fallbacks,
    earlyCount, lateDepthAverage, lateCount }
}

function main() {
  const { model, version } = loadNNUE()
  const engine = new DarkChessEngine(board => model.evaluate(board))
  const count = num('BATCH_SIZE', 10) * num('NUM_BATCHES', 1)
  const output = process.env.OUTPUT_DIR ?? 'output_data'
  mkdirSync(output, {recursive: true})
  console.log(`[Shared Engine] ${version}; max search ${num('SEARCH_MAX_SECONDS', 3)}s, games=${count}`)
  let saved = 0
  for (let i = 0; i < count; i++) {
    try {
      const game = play(engine, version)
      const pass = game.fallback === 0 && game.depthAverage >= num('MIN_AVG_COMPLETED_DEPTH', 2.5) &&
        (game.lateCount === 0 || game.lateDepthAverage >= num('MIN_LATE_COMPLETED_DEPTH', 2.5))
      console.log(`[Shared Engine] Game ${i + 1}: ply=${game.replay.ply} result=${game.replay.res} ` +
        `seconds=${game.seconds.toFixed(1)} nodes=${game.totalNodes} ` +
        `avg_completed_depth=${game.depthAverage.toFixed(2)} late_avg_depth=${game.lateDepthAverage.toFixed(2)} ` +
        `depth_zero=${game.fallback} quality=${pass ? 'PASS' : 'QUARANTINED'}`)
      if (!pass) continue
      const path = join(output, `data_${Date.now()}_${game.replay.id}.jsonl.gz`)
      writeFileSync(path, gzipSync(JSON.stringify(game.replay) + '\n'))
      saved++
    } catch (error) {
      console.error('[Shared Engine] Incomplete or invalid game quarantined:', error)
    }
  }
  if (!saved) throw new Error('Quality gate blocked every game; refusing to upload weak replay data')
  console.log(`[Shared Engine] VERIFIED ${saved} of ${count} complete games for replay parity validation`)
}
main()
