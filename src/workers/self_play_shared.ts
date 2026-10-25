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
const num = (key: string, defaultValue: number) => {
  const value = Number(process.env[key] ?? defaultValue)
  if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid ${key}=${value}`)
  return value
}
const encode = (move: Move) => move.from * 32 + move.to
const nHidden = (board: Board) => board.remainingCounts.reduce((a, b) => a + b, 0)

function limits(board: Board): SearchLimits {
  // Production: exactly three plies for opening, twelve thereafter.
  // There are NO node or per-move time budgets. Return a move only after
  // the shared engine evaluates every root move to the requested depth.
  // The explicit smoke override is only for small CI functional games.
  // Production never defines SELF_PLAY_SMOKE_DEPTH.
  const smoke = process.env.SELF_PLAY_SMOKE_DEPTH
  const maxDepth = smoke === undefined ? (board.ply < 3 ? 3 : 12) :
    Math.max(1, Math.floor(num('SELF_PLAY_SMOKE_DEPTH', 2)))
  return {maxDepth, requireFullDepth: true}
}

type Checkpoint = {
  version: string; replay: Replay; rootValues: Record<number, number>; boardPly: number
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
  if (snapshot.version !== version) return memo
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
  writeFileSync(staging, JSON.stringify({version, replay, rootValues, boardPly: board.ply} satisfies Checkpoint))
  renameSync(staging, path)
}
function restoreCheckpoint(version: string): {board: Board; replay: Replay; rootValues: Record<number, number>} | null {
  const path = checkpointPath()
  if (!existsSync(path)) return null
  const snapshot = JSON.parse(readFileSync(path, 'utf8')) as Checkpoint
  if (snapshot.version !== version || snapshot.replay?.ver !== VERSION) {
    console.log('[Resume] Champion changed; previous search values cannot be reused.')
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
    const targetDepth = limits(board).maxDepth!
    const limitsForMove = {
      ...limits(board),
      resumeRootValues: rootValues,
      resumeSubtrees: memo,
      onSubtreeCompleted: saveSubtree,
      onRootCompleted: (move: Move, value: number) => {
        rootValues[encode(move)] = value
        // A completed root branch is durable before the next branch starts.
        // This is the smallest safe interruption boundary; it is never
        // mistaken for a fully calculated move.
        flushMemo()
        saveCheckpoint(version, record, board, rootValues)
      },
    }
    const staticValue = engine.evaluate(board)
    const analysis = engine.analyze(board, limitsForMove)
    if (!analysis.move || analysis.depth !== targetDepth) {
      throw new Error(`Search incomplete at ply ${ply}: depth=${analysis.depth}, expected=${targetDepth}`)
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
    console.log(`[Deep Search] ply=${board.ply} COMPLETE depth=${analysis.depth} nodes=${analysis.nodes} ms=${Math.round(analysis.elapsedMs)}`)
  }
  if (!board.isOver) throw new Error('Game unfinished: cannot upload')
  flushMemo()
  record.ply = record.mov.length
  record.res = board.draw ? 0 : board.winner === Color.RED ? 1 : -1
  if (board.draw) record.draw_reason = board.halfMoveClock >= 60 ? 'sixty_no_progress' : 'other'
  const searchedDepths = record.qd.slice(1)
  const lateDepths = record.qd.filter((_, i) => i >= 3)
  const depthAverage = searchedDepths.reduce((a, b) => a + b, 0) / Math.max(1, searchedDepths.length)
  const lateDepthAverage = lateDepths.reduce((a, b) => a + b, 0) / Math.max(1, lateDepths.length)
  return {replay: record, seconds: (performance.now() - started) / 1000,
    totalNodes: nodes, depthAverage, fallback: record.qd.some((d, i) => d !== (process.env.SELF_PLAY_SMOKE_DEPTH === undefined ? (i < 3 ? 3 : 12) : Math.floor(num('SELF_PLAY_SMOKE_DEPTH', 2)))) ? 1 : 0,
    lateDepthAverage, lateCount: lateDepths.length}
}

function main() {
  const { model, version } = loadNNUE()
  const engine = new DarkChessEngine(board => model.evaluate(board))
  const count = num('BATCH_SIZE', 10) * num('NUM_BATCHES', 1)
  const output = process.env.OUTPUT_DIR ?? 'output_data'
  mkdirSync(output, {recursive: true})
  console.log(`[Shared Engine] ${version}; max search ${process.env.SEARCH_MAX_SECONDS ?? 'unlimited'}s, games=${count}`)
  let saved = 0
  for (let i = 0; i < count; i++) {
    try {
      const game = play(engine, version)
      const pass = game.fallback === 0 && game.replay.qd.every((d, ply) => d === (process.env.SELF_PLAY_SMOKE_DEPTH === undefined ? (ply < 3 ? 3 : 12) : Math.floor(num('SELF_PLAY_SMOKE_DEPTH', 2))))
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
