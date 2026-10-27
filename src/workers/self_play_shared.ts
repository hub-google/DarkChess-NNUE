/** Self-play using EXACTLY the same DarkChessEngine class as the browser.
 * Production, arena and browser use this identical decision core. Replay
 * legality is checked independently using the Python referee.
 */
import { randomUUID } from 'node:crypto'
import { writeFileSync, readFileSync, appendFileSync, existsSync, renameSync, unlinkSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { gzipSync } from 'node:zlib'
import { Board, Color, Piece, type Move } from '../../frontend/src/engine/board'
import { DarkChessEngine, type SearchAnalysis, type SearchLimits } from '../../frontend/src/engine/search'
import { loadNNUE } from './model_runtime'

type Replay = {
  id: string; ts: number; ver: string; model: string; hid: number[]
  mov: number[]; q: number[]; v: number[]; qd: number[]
  pi: [number, number][][]; qn: number[]; qt: number[]; res: number; ply: number; draw_reason?: string
  red_model?: string; black_model?: string
}
const VERSION = 'v2.1.0-perpetual-chase'
// 1-based (row, column): (2,3), (2,6), (3,3), (3,6).
// These four central cells are symmetry-related opening candidates;
// no unmeasured win-rate advantage is claimed.
const OPENING_SQUARES = [10, 13, 18, 21] as const
const SEARCH_STRATEGY = 'fixed-opening-quick-first-ten-v1'
const num = (key: string, defaultValue: number) => {
  const value = Number(process.env[key] ?? defaultValue)
  if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid ${key}=${value}`)
  return value
}
const encode = (move: Move) => move.from * 32 + move.to
const nHidden = (board: Board) => board.remainingCounts.reduce((a, b) => a + b, 0)

/** Search depth belongs to each actual move, counted from move 1. */
function targetDepth(ply: number): number {
  // Test-only override exercises an actual search for every move.
  const smoke = process.env.SELF_PLAY_SMOKE_DEPTH
  if (smoke !== undefined) return Math.max(1, Math.floor(num('SELF_PLAY_SMOKE_DEPTH', 2)))
  if (ply === 0) return 0 // First flip: random from four predetermined central squares.
  if (ply < 10) return 1 // Moves 2–10: single-ply complete search; fast.
  return 12 // Move 11 onward: complete twelve-ply search without time/node limits.
}
function limits(board: Board): SearchLimits {
  // The fixed first flip does not invoke the engine. Other moves still require
  // a fully finished depth, even when the opening depth is only one.
  return {maxDepth: Math.max(1, targetDepth(board.ply)), requireFullDepth: true}
}
function depthAllowed(actual: number, ply: number): boolean {
  // Restored older checkpoints can contain valid earlier moves searched DEEPER
  // than the new minimum. Never discard a legal deep result as a fallback.
  return Number.isInteger(actual) && actual >= targetDepth(ply)
}
function fixedOpening(board: Board): SearchAnalysis {
  const eligible = board.legalMoves().filter(move =>
    move.flip && OPENING_SQUARES.includes(move.from as (typeof OPENING_SQUARES)[number]))
  if (board.ply !== 0 || board.turn !== Color.NONE || eligible.length !== OPENING_SQUARES.length) {
    throw new Error('Fixed opening may only select one of the four legal central flips on an untouched board')
  }
  const move = eligible[Math.floor(Math.random() * eligible.length)]
  // No search was conducted. Record the truth: zero depth, zero nodes,
  // zero duration, no false search-policy targets.
  return {move, depth: 0, nodes: 0, elapsedMs: 0, rootValues: {}, value: 0}
}

type Checkpoint = {
  version: string; strategy?: string; replay: Replay; rootValues: Record<number, number>; boardPly: number
}
const checkpointPath = () => join(process.env.RESUME_STATE_DIR ?? 'resume_state',
  `worker_${process.env.WORKER_ID ?? 'local'}.json`)
const memoPath = () => checkpointPath().replace(/\.json$/, '.memo')
function restoreExactMemo(version: string): Map<string, number> {
  const memo = new Map<string, number>()
  if (!existsSync(memoPath()) || !existsSync(checkpointPath())) return memo
  // A saved memo belongs exclusively to the model and referee deal in the
  // paired JSON checkpoint. No untrusted/historical model values are reused.
  const snapshot = JSON.parse(readFileSync(checkpointPath(), 'utf8')) as Checkpoint
  if (snapshot.version !== version || snapshot.strategy !== SEARCH_STRATEGY) return memo
  for (const line of readFileSync(memoPath(), 'utf8').split('\n')) {
    if (!line.trim()) continue
    try {
      const [key, value] = JSON.parse(line) as [string, number]
      if (typeof key === 'string' && Number.isFinite(value)) memo.set(key, value)
    } catch {
      // A killed process can leave the last JSONL record truncated. Ignore it.
    }
  }
  console.log(`[Resume] Restored ${memo.size} completed exact-depth subtrees`)
  return memo
}
function saveCheckpoint(version: string, replay: Replay, board: Board, rootValues: Record<number, number>) {
  const path = checkpointPath()
  mkdirSync(process.env.RESUME_STATE_DIR ?? 'resume_state', {recursive: true})
  const staging = path + '.tmp'
  writeFileSync(staging, JSON.stringify({version, strategy: SEARCH_STRATEGY, replay, rootValues, boardPly: board.ply} satisfies Checkpoint))
  renameSync(staging, path)
}
function restoreCheckpoint(version: string): {board: Board; replay: Replay; rootValues: Record<number, number>} | null {
  const path = checkpointPath()
  if (!existsSync(path)) return null
  const snapshot = JSON.parse(readFileSync(path, 'utf8')) as Checkpoint
  if (snapshot.version !== version || snapshot.replay?.ver !== VERSION ||
      snapshot.strategy !== SEARCH_STRATEGY) {
    console.log('[Resume] Champion or opening strategy changed; restart to avoid mixing search depths.')
    unlinkSync(path)
    if (existsSync(memoPath())) unlinkSync(memoPath())
    return null
  }
  const replay = snapshot.replay
  if (replay.mov.length !== replay.q.length || replay.mov.length !== replay.qd.length ||
      replay.mov.length !== replay.v.length || replay.mov.length !== replay.pi.length ||
      replay.mov.length !== replay.qn.length || replay.mov.length !== replay.qt.length) {
    throw new Error('Invalid checkpoint replay lengths')
  }
  const board = new Board(false)
  board.hidden = replay.hid.map(p => (p + 1) as Piece)
  for (const move of replay.mov) {
    const from = move >> 5, to = move & 31
    if (!board.play({from, to, flip: from === to})) {
      throw new Error('Invalid replay in self-play checkpoint')
    }
  }
  if (board.ply !== snapshot.boardPly) throw new Error('Checkpoint does not match board ply')
  console.log(`[Resume] Restored ${replay.mov.length} plies and ${Object.keys(snapshot.rootValues).length} complete root branches`)
  return {board, replay, rootValues: snapshot.rootValues}
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
  fallback: number; lateDepthAverage: number; lateCount: number
} {
  const saved = restoreCheckpoint(version)
  const board = saved?.board ?? new Board()
  const record: Replay = saved?.replay ?? {
    id: randomUUID(), ts: Date.now(), ver: VERSION, model: version,
    hid: board.hidden.map(x => Number(x) - 1),
    mov: [], q: [], v: [], qd: [], pi: [], qn: [], qt: [], res: 0, ply: 0,
    red_model: version, black_model: version,
  }
  let rootValues = saved?.rootValues ?? {}
  if (!saved && existsSync(memoPath())) unlinkSync(memoPath())
  const memo = saved ? restoreExactMemo(version) : new Map<string, number>()
  // Subtree records are append-only JSONL. They survive the job timeout even
  // when no top-level root move could finish in the same Actions job.
  let memoBuffer: string[] = []
  const flushMemo = () => {
    if (memoBuffer.length) {
      appendFileSync(memoPath(), memoBuffer.join('\n') + '\n')
      memoBuffer = []
    }
  }
  const saveSubtree = (key: string, value: number) => {
    memoBuffer.push(JSON.stringify([key, value]))
    if (memoBuffer.length >= 50) flushMemo()
  }
  // Always persist the initial referee deal, even before the first branch.
  saveCheckpoint(version, record, board, rootValues)
  const started = performance.now()
  let nodes = 0
  for (let ply = board.ply; ply < 512 && !board.isOver; ply++) {
    const side = board.turn
    const depth = targetDepth(board.ply)
    const staticValue = engine.evaluate(board)
    const analysis = depth === 0 ? fixedOpening(board) : engine.analyze(board, {
      ...limits(board),
      resumeRootValues: rootValues,
      resumeSubtrees: memo,
      onSubtreeCompleted: saveSubtree,
      onRootCompleted: (move: Move, value: number) => {
        rootValues[encode(move)] = value
        flushMemo()
        saveCheckpoint(version, record, board, rootValues)
      },
    })
    if (!analysis.move || analysis.depth !== depth) {
      throw new Error(`Search incomplete at ply ${ply}: depth=${analysis.depth}, expected=${depth}`)
    }
    const chosen = side === Color.NONE ? analysis.move : select(analysis, side, ply)
    const chosenValue = analysis.rootValues[encode(chosen)] ?? analysis.value
    record.mov.push(encode(chosen))
    record.q.push(chosenValue)
    record.v.push(staticValue)
    record.qd.push(analysis.depth)
    record.qn.push(analysis.nodes)
    record.qt.push(Math.round(analysis.elapsedMs))
    record.pi.push(side === Color.NONE ? [] : rankedPolicy(analysis, side))
    if (!board.play(chosen)) throw new Error('Illegal self-play move after exact search')
    nodes += analysis.nodes
    rootValues = {}
    flushMemo()
    saveCheckpoint(version, record, board, rootValues)
    console.log(`[${depth === 0 ? 'Fixed Opening' : depth === 1 ? 'Quick Search' : 'Deep Search'}] ply=${board.ply} COMPLETE depth=${analysis.depth} nodes=${analysis.nodes} ms=${Math.round(analysis.elapsedMs)} move=${encode(chosen)}`)
  }
  if (!board.isOver) throw new Error('Game unfinished: cannot upload')
  flushMemo()
  record.ply = record.mov.length
  record.res = board.draw ? 0 : board.winner === Color.RED ? 1 : -1
  if (board.draw) record.draw_reason = board.halfMoveClock >= 60 ? 'sixty_no_progress' : 'other'
  const searchedDepths = record.qd.slice(1)
  const lateDepths = record.qd.filter((_, i) => i >= 10)
  const depthAverage = searchedDepths.reduce((a, b) => a + b, 0) / Math.max(1, searchedDepths.length)
  const lateDepthAverage = lateDepths.reduce((a, b) => a + b, 0) / Math.max(1, lateDepths.length)
  return {replay: record, seconds: (performance.now() - started) / 1000,
    totalNodes: nodes, depthAverage, fallback: record.qd.some((d, i) => !depthAllowed(d, i)) ? 1 : 0,
    lateDepthAverage, lateCount: lateDepths.length}
}

function validateOpening(): void {
  if (targetDepth(0) !== 0 || targetDepth(1) !== 1 ||
      targetDepth(9) !== 1 || targetDepth(10) !== 12) {
    throw new Error('Incorrect opening search depths')
  }
  for (let sample = 0; sample < 128; sample++) {
    const board = new Board()
    const analysis = fixedOpening(board)
    if (!analysis.move ||
        !OPENING_SQUARES.includes(analysis.move.from as (typeof OPENING_SQUARES)[number]) ||
        analysis.move.to !== analysis.move.from || analysis.depth !== 0 ||
        analysis.nodes !== 0 || analysis.elapsedMs !== 0 ||
        Object.keys(analysis.rootValues).length !== 0) {
      throw new Error('Fixed opening selected an invalid move or fabricated search results')
    }
    if (!board.play(analysis.move)) throw new Error('Fixed opening was illegal')
  }
  console.log('[Opening Test] PASS: 128 legal fixed flips with zero search.')
}
function main() {
  if (process.env.SELF_PLAY_VALIDATE_OPENING === '1') {
    validateOpening()
    return
  }
  const { model, version } = loadNNUE()
  const engine = new DarkChessEngine(board => model.evaluate(board))
  const count = num('BATCH_SIZE', 10) * num('NUM_BATCHES', 1)
  const output = process.env.OUTPUT_DIR ?? 'output_data'
  mkdirSync(output, {recursive: true})
  console.log(`[Shared Engine] ${version}; opening=fixed, moves 2-10=depth-1, moves 11+=depth-12; search-time=unlimited, games=${count}`)
  let saved = 0
  for (let i = 0; i < count; i++) {
    try {
      const game = play(engine, version)
      const pass = game.fallback === 0 && game.replay.qd.every((d, ply) => depthAllowed(d, ply))
      console.log(`[Shared Engine] Game ${i + 1}: ply=${game.replay.ply} result=${game.replay.res} ` +
        `seconds=${game.seconds.toFixed(1)} nodes=${game.totalNodes} ` +
        `avg_completed_depth=${game.depthAverage.toFixed(2)} late_avg_depth=${game.lateDepthAverage.toFixed(2)} ` +
        `depth_zero=${game.fallback} quality=${pass ? 'PASS' : 'QUARANTINED'}`)
      if (!pass) continue
      const path = join(output, `data_${Date.now()}_${game.replay.id}.jsonl.gz`)
      writeFileSync(path, gzipSync(JSON.stringify(game.replay) + '\n'))
      saved++
      if (existsSync(checkpointPath())) unlinkSync(checkpointPath())
      if (existsSync(memoPath())) unlinkSync(memoPath())
    } catch (error) {
      console.error('[Shared Engine] Incomplete or invalid game quarantined:', error)
    }
  }
  if (!saved && !existsSync(checkpointPath())) throw new Error('No complete games and no resumable search checkpoint')
  console.log(`[Shared Engine] VERIFIED ${saved} of ${count} complete games; unfinished moves remain checkpointed`)
}
main()
