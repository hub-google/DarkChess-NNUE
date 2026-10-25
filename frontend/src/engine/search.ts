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
class BudgetExceeded extends Error {}

export type SearchLimits = {
  maxDepth?: number
  nodeBudget?: number
  maxMs?: number
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
  if (board.turn === Color.NONE) return { move: available[0], depth: 0, nodes: 0, elapsedMs: 0, rootValues: {}, value: 0 }
  const maxDepth = Math.max(1, Math.floor(limits.maxDepth ?? 3))
  const nodeBudget = Math.max(1, Math.floor(limits.nodeBudget ?? 2500))
  const maxMs = Math.max(1, limits.maxMs ?? 150)
  const startedAt = performance.now()
  const deadline = startedAt + maxMs
  let nodes = 0
  const evalPosition = (state: Board) => {
    if (state.draw) return 0
    if (state.winner === Color.RED) return 1
    if (state.winner === Color.BLACK) return -1
    // A large early chance tree is more efficiently scored by cheap material;
    // reserve expensive browser NNUE forwards for later revealed positions.
    return this.evaluator ? this.evaluator(state) : fromMaterial(state)
  }
  const tick = () => {
    nodes += 1
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
      const child = state.clone()
      // Force THIS hypothetical reveal, do not read actual private referee mapping.
      child.hidden[move.from] = (index + 1) as Piece
      if (!child.play(move)) throw new Error('search attempted illegal flip')
      const probability = count / total
      expected += probability * search(child, depth - 1, -Infinity, Infinity)
      seen += probability
      const remaining = Math.max(0, 1 - seen)
      // All evaluations are in [-1, 1], so these Star1 bounds are safe.
      if (expected + remaining <= alpha) return expected + remaining
      if (expected - remaining >= beta) return expected - remaining
    }
    return Math.max(-1, Math.min(1, expected))
  }
  const search = (state: Board, depth: number, alpha: number, beta: number): number => {
    tick()
    if (state.isOver || depth <= 0) return evalPosition(state)
    const moves = ordered(state, state.legalMoves())
    if (!moves.length) return evalPosition(state)
    if (state.turn === Color.RED) {
      let value = -Infinity
      for (const move of moves) {
        value = Math.max(value, branch(state, move, depth, alpha, beta))
        alpha = Math.max(alpha, value)
        if (alpha >= beta) break
      }
      return value
    }
    let value = Infinity
    for (const move of moves) {
      value = Math.min(value, branch(state, move, depth, alpha, beta))
      beta = Math.min(beta, value)
      if (alpha >= beta) break
    }
    return value
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
