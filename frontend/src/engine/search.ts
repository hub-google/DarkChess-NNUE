/** Public-belief expectiminimax shared by browser and Node self-play.

    Hidden contents are *never* consulted when choosing a move. A speculative
    reveal overwrites ONLY the selected square with each possible remaining
    public piece type. All subsequent speculative flips are chance branches.
    Evaluation is always red-perspective, with RED max and BLACK min.
*/
import { Board, Color, Piece, type Move, type Evaluator, colorOf } from './board'

export const MATERIAL = [12, 7, 6, 5, 4, 5, 2]
const isFlip = (move: Move) => move.flip === true || move.from === move.to
const fromMaterial = (board: Board): number => {
  let signed = 0
  let total = 0
  for (let index = 0; index < 14; index++) {
    const piece = (index + 1) as Piece
    const shown = board.grid.filter(p => p === piece).length
    const weight = MATERIAL[index % 7]
    const count = shown + 0.65 * board.remainingCounts[index]
    const contribution = weight * count
    total += contribution
    signed += colorOf(piece) === Color.RED ? contribution : -contribution
  }
  return total === 0 ? 0 : Math.tanh(2.5 * signed / total)
}
const moveKey = (m: Move) => m.from * 32 + m.to
const capture = (board: Board, m: Move) => !isFlip(m) && board.grid[m.to] !== Piece.EMPTY
function ordered(board: Board, moves: Move[]) {
  return moves.slice().sort((a, b) => (
    Number(capture(board, b)) - Number(capture(board, a))
    || Number(isFlip(b)) - Number(isFlip(a))
    || moveKey(a) - moveKey(b)
  ))
}
/**
 * Optional bounded-width search for production CPU self-play.
 * This still reaches the requested ply horizon for every retained branch and
 * sums ALL possible identities for every explored flip, but does NOT claim
 * exhaustive minimax over all legal interior moves. Root moves remain complete.
 */
function selectInteriorMoves(board: Board, ranked: Move[], requested: number): Move[] {
  const hidden = board.remainingCounts.reduce((sum, count) => sum + count, 0)
  const limit = hidden >= 14 ? Math.min(requested, 2)
    : hidden >= 7 ? Math.min(Math.max(requested, 3), 3)
    : hidden >= 1 ? Math.max(requested, 4) : Infinity
  if (ranked.length <= limit) return ranked
  const captures = ranked.filter(move => capture(board, move))
  const flips = ranked.filter(isFlip)
  const quiet = ranked.filter(move => !isFlip(move) && !capture(board, move))
  // Use only PUBLIC board state when breaking equally ranked flip options.
  let seed = 2166136261
  for (let i = 0; i < board.grid.length; i++) {
    seed = Math.imul(seed ^ (board.grid[i] + 17 * (i + 1)), 16777619)
  }
  const spread = flips.slice().sort((a, b) => {
    const score = (m: Move) => Math.imul((m.from + 1) ^ seed, 0x9e3779b1) >>> 0
    return score(a) - score(b)
  })
  const chosen: Move[] = []
  const include = (move: Move | undefined) => {
    if (move && chosen.length < limit && !chosen.includes(move)) chosen.push(move)
  }
  include(captures[0]) // never drop the best immediate capture
  include(spread[0])   // never drop all hidden-information opportunities
  include(quiet[0])    // retain a reversible positional option where possible
  for (const move of [...captures, ...spread, ...quiet]) include(move)
  return chosen
}

class BudgetExceeded extends Error {}

/** Periodic, read-only statistics; does not change exact search results. */
export type SearchProgress = {
  chancePrunes: number
  depth: number
  nodes: number
  elapsedMs: number
  rootsDone: number
  rootsTotal: number
  cacheHits: number
  cacheEntries: number
  evaluations: number
}
export type SearchLimits = {
  /** Disable bounded chance-node windows only when comparing to reference search. */
  useChanceBounds?: boolean
  /** Experimental: root remains exhaustive; cap only interior move width. */
  innerMoveLimit?: number
  onProgress?: (progress: SearchProgress) => void
  progressIntervalMs?: number
  maxDepth?: number
  nodeBudget?: number
  maxMs?: number
  requireFullDepth?: boolean
  onRootCompleted?: (move: Move, value: number) => void
  resumeRootValues?: Record<number, number>
  resumeSubtrees?: Map<string, number>
  onSubtreeCompleted?: (key: string, value: number) => void
}

export type SearchAnalysis = {
  move: Move | undefined
  depth: number
  nodes: number
  elapsedMs: number
  rootValues: Record<number, number>
  value: number
}

/** The ONE decision engine used by browser games and Node.js self-play. */
export class DarkChessEngine {
  constructor(private readonly evaluator?: Evaluator) {}

  evaluate(board: Board): number {
    return this.evaluator ? this.evaluator(board) : fromMaterial(board)
  }

  analyze(board: Board, limits: SearchLimits = {}): SearchAnalysis {
  const available = board.legalMoves()
  if (!available.length) return { move: undefined, depth: 0, nodes: 0, elapsedMs: 0, rootValues: {}, value: 0 }
  const maxDepth = Math.max(1, Math.floor(limits.maxDepth ?? 3))
  const strict = limits.requireFullDepth === true
  if (strict && (limits.nodeBudget !== undefined || limits.maxMs !== undefined)) {
    throw new Error('Complete-depth search must not have a node or time cutoff')
  }
  const nodeBudget = strict ? Infinity : Math.max(1, Math.floor(limits.nodeBudget ?? 2500))
  const maxMs = strict ? Infinity : Math.max(1, limits.maxMs ?? 150)
  const startedAt = performance.now()
  const deadline = startedAt + maxMs
  let nodes = 0
  let cacheHits = 0
  let chancePrunes = 0
  let evaluations = 0
  let rootsDone = Object.keys(limits.resumeRootValues ?? {}).length
  let lastProgressAt = startedAt
  const reportProgress = (force = false) => {
    if (!limits.onProgress) return
    const now = performance.now()
    if (!force && now - lastProgressAt < (limits.progressIntervalMs ?? 30000)) return
    lastProgressAt = now
    limits.onProgress({
      chancePrunes, depth: maxDepth, nodes, elapsedMs: now - startedAt, rootsDone,
      rootsTotal: available.length, cacheHits,
      cacheEntries: limits.resumeSubtrees?.size ?? 0, evaluations,
    })
  }
  const evalPosition = (state: Board) => {
    if (state.draw) return 0
    if (state.winner === Color.RED) return 1
    if (state.winner === Color.BLACK) return -1
    evaluations++
    // A large early chance tree is more efficiently scored by cheap material;
    // reserve expensive browser NNUE forwards for later revealed positions.
    return this.evaluator ? this.evaluator(state) : fromMaterial(state)
  }
  const tick = () => {
    nodes += 1
    if (limits.onProgress && nodes % 1024 === 0) reportProgress()
    if (nodes > nodeBudget || (nodes % 16 === 0 && performance.now() > deadline)) {
      throw new BudgetExceeded()
    }
  }
  const branch = (state: Board, move: Move, depth: number, alpha: number, beta: number): number => {
    if (!isFlip(move)) {
      const child = state.clone()
      if (!child.play(move)) throw new Error('search attempted illegal move')
      return search(child, depth - 1, alpha, beta)
    }
    const total = state.remainingCounts.reduce((a, b) => a + b, 0)
    if (!total) throw new Error('chance node without remaining hidden pieces')
    let expected = 0
    let seen = 0
    for (let index = 0; index < 14; index++) {
      const count = state.remainingCounts[index]
      if (count <= 0) continue
      const probability = count / total
      // All terminal and NNUE evaluations are bounded to [-1, 1].
      // After this outcome, later unobserved outcomes weigh at most
      // +/- remaining. Derive a provably safe child alpha/beta window:
      //   S + p*child + R <= alpha  => chance is an upper-bound cutoff
      //   S + p*child - R >= beta   => chance is a lower-bound cutoff.
      // The resulting threshold does not sample/discard any chance outcome.
      const remaining = Math.max(0, 1 - seen - probability)
      const bounded = limits.useChanceBounds !== false
      const childAlpha = bounded && Number.isFinite(alpha)
        ? (alpha - expected - remaining) / probability : -Infinity
      const childBeta = bounded && Number.isFinite(beta)
        ? (beta - expected + remaining) / probability : Infinity
      if (childAlpha >= 1 + 1e-12) {
        chancePrunes++
        return alpha
      }
      if (childBeta <= -1 - 1e-12) {
        chancePrunes++
        return beta
      }
      const child = state.clone()
      // Force THIS hypothetical reveal, never read private face-down layout.
      child.hidden[move.from] = (index + 1) as Piece
      if (!child.play(move)) throw new Error('search attempted illegal flip')
      const outcome = search(child, depth - 1,
        childAlpha <= -1 ? -Infinity : childAlpha,
        childBeta >= 1 ? Infinity : childBeta)
      if (outcome <= childAlpha && Number.isFinite(childAlpha)) {
        chancePrunes++
        return alpha
      }
      if (outcome >= childBeta && Number.isFinite(childBeta)) {
        chancePrunes++
        return beta
      }
      expected += probability * outcome
      seen += probability
      const left = Math.max(0, 1 - seen)
      if (expected + left <= alpha) { chancePrunes++; return expected + left }
      if (expected - left >= beta) { chancePrunes++; return expected - left }
    }
    return Math.max(-1, Math.min(1, expected))
  }
  const search = (state: Board, depth: number, alpha: number, beta: number): number => {
    tick()
    if (state.isOver || depth <= 0) return evalPosition(state)
    // Only cache EXACT (full-window) expectiminimax subtrees.
    // A bound from a pruned alpha-beta window is NEVER safe to reuse.
    // Only reversible history matters for future repetitions: flips/captures
    // permanently change the inventory. Preserve all separate chase state.
    // Never include the private referee's hidden identities in cache keys.
    const cacheable = strict && depth >= 2 && depth <= 6 &&
      alpha === -Infinity && beta === Infinity
    const recentHistory = cacheable && state.halfMoveClock > 0
      ? state.history.slice(-state.halfMoveClock) : []
    const cacheKey = cacheable
      ? JSON.stringify([depth, state.grid, state.remainingCounts, state.turn,
          state.halfMoveClock, recentHistory, state.tokenAtSquare,
          state.chaseThreats, state.pendingChase, state.winner, state.draw]) : ''
    const cached = cacheable ? limits.resumeSubtrees?.get(cacheKey) : undefined
    if (cached !== undefined) { cacheHits++; return cached }
    const saveExact = (value: number): number => {
      if (cacheable && Number.isFinite(value)) {
        limits.resumeSubtrees?.set(cacheKey, value)
        limits.onSubtreeCompleted?.(cacheKey, value)
      }
      return value
    }
    const rankedMoves = ordered(state, state.legalMoves())
    if (!rankedMoves.length) return evalPosition(state)
    const moves = limits.innerMoveLimit !== undefined && limits.innerMoveLimit > 0
      ? selectInteriorMoves(state, rankedMoves, limits.innerMoveLimit) : rankedMoves
    if (state.turn === Color.RED) {
      let value = -Infinity
      for (const move of moves) {
        value = Math.max(value, branch(state, move, depth, alpha, beta))
        alpha = Math.max(alpha, value)
        if (alpha >= beta) break
      }
      return saveExact(value)
    }
    let value = Infinity
    for (const move of moves) {
      value = Math.min(value, branch(state, move, depth, alpha, beta))
      beta = Math.min(beta, value)
      if (alpha >= beta) break
    }
    return saveExact(value)
  }
  if (board.turn === Color.NONE) {
    // Opening is a chance choice: the first revealer becomes the owner of
    // whatever color is turned over. Evaluate the full requested horizon,
    // never consult the referee's true face-down arrangement.
    const total = board.remainingCounts.reduce((a, b) => a + b, 0)
    const values: Record<number, number> = {...(limits.resumeRootValues ?? {})}
    for (const move of ordered(board, available)) {
      const key = moveKey(move)
      if (Number.isFinite(values[key])) continue
      let expected = 0
      for (let index = 0; index < 14; index++) {
        const count = board.remainingCounts[index]
        if (count <= 0) continue
        const child = board.clone()
        child.hidden[move.from] = (index + 1) as Piece
        if (!child.play(move)) throw new Error('illegal speculative first flip')
        const ownColor = colorOf((index + 1) as Piece)
        const value = search(child, maxDepth - 1, -Infinity, Infinity)
        expected += count / total * (ownColor === Color.RED ? value : -value)
      }
      values[key] = expected
      rootsDone++
      limits.onRootCompleted?.(move, expected)
      reportProgress()
    }
    if (Object.keys(values).length !== available.length) {
      if (strict) throw new Error('Opening search incomplete: refusing to move')
      return {move: available[0], depth: 0, nodes,
        elapsedMs: performance.now() - startedAt, rootValues: values, value: 0}
    }
    let best = available[0]
    for (const move of available) if (values[moveKey(move)] > values[moveKey(best)]) best = move
    reportProgress(true)
    return {move: best, depth: maxDepth, nodes,
      elapsedMs: performance.now() - startedAt, rootValues: values, value: values[moveKey(best)]}
  }
  if (strict) {
    // Directly evaluate exactly the required horizon. No shallower
    // iterative-deepening result may become a self-play move.
    const rootValues: Record<number, number> = {...(limits.resumeRootValues ?? {})}
    const root = ordered(board, available)
    let best = root[0]
    let value = board.turn === Color.RED ? -Infinity : Infinity
    for (const move of root) {
      const key = moveKey(move)
      const score = Number.isFinite(rootValues[key])
        ? rootValues[key] : branch(board, move, maxDepth, -Infinity, Infinity)
      if (!Number.isFinite(rootValues[key])) {
        rootValues[key] = score
        rootsDone++
        limits.onRootCompleted?.(move, score)
        reportProgress()
      }
      if ((board.turn === Color.RED && score > value) ||
          (board.turn === Color.BLACK && score < value)) {
        best = move; value = score
      }
    }
    if (Object.keys(rootValues).length !== root.length) throw new Error('Full-depth root was not completed')
    reportProgress(true)
    return {move: best, depth: maxDepth, nodes,
      elapsedMs: performance.now() - startedAt, rootValues, value}
  }
  let completedDepth = 0
  let rootValues: Record<number, number> = {}
  // Deterministic fallback is public-state only and takes no chance samples.
  // It remains legal if even a depth-one search cannot complete in time.
  let best = ordered(board, available)[0]
  const root = ordered(board, available)
  for (let depth = 1; depth <= maxDepth; depth++) {
    if (performance.now() > deadline) break
    try {
      let candidate = best
      let value = board.turn === Color.RED ? -Infinity : Infinity
      const values: Record<number, number> = {}
      for (const move of root) {
        const score = branch(board, move, depth, -Infinity, Infinity)
        values[moveKey(move)] = score
        if ((board.turn === Color.RED && score > value) ||
            (board.turn === Color.BLACK && score < value)) {
          value = score
          candidate = move
        }
      }
      best = candidate // commit only a completely finished iteration
      rootValues = values
      completedDepth = depth
    } catch (error) {
      if (error instanceof BudgetExceeded) break
      throw error
    }
  }
  return { move: best, depth: completedDepth, nodes, elapsedMs: performance.now() - startedAt, rootValues, value: rootValues[moveKey(best)] ?? evalPosition(board) }
  }
}

export function chooseSearchedMove(board: Board, nnue?: Evaluator, limits: SearchLimits = {}): Move | undefined {
  return new DarkChessEngine(nnue).analyze(board, limits).move
}
