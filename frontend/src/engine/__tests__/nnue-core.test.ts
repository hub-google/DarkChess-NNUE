import { describe, expect, it } from 'vitest'
import { Board, type Piece } from '../board'
import { BrowserNNUE, extractFeatures } from '../nnue-core'

function referenceEval(model: {
  fc1w: Float32Array; fc1b: Float32Array
  fc2w: Float32Array; fc2b: Float32Array
  fc3w: Float32Array; fc3b: Float32Array
}, board: Board): number {
  // Previous feature-tensor implementation, retained as an independent
  // numerical reference for the new sparse CPU path.
  const input = extractFeatures(board)
  const active: number[] = []
  for (let col = 0; col < 498; col++) if (input[col] !== 0) active.push(col)
  const clip = (v: number) => Math.min(1, Math.max(0, v))
  const h1 = new Float32Array(256)
  for (let row = 0; row < 256; row++) {
    let sum = model.fc1b[row]
    for (const col of active) sum += model.fc1w[row * 498 + col] * input[col]
    h1[row] = clip(sum)
  }
  const h2 = new Float32Array(32)
  for (let row = 0; row < 32; row++) {
    let sum = model.fc2b[row]
    for (let col = 0; col < 256; col++) sum += model.fc2w[row * 256 + col] * h1[col]
    h2[row] = clip(sum)
  }
  let output = model.fc3b[0]
  for (let col = 0; col < 32; col++) output += model.fc3w[col] * h2[col]
  return Math.tanh(output)
}

describe('sparse NNUE exact numerical parity', () => {
  it('matches the original Float32 feature-tensor implementation throughout a game', () => {
    const mk = (size: number, seed: number) =>
      Float32Array.from({length: size}, (_, i) => Math.sin(i * 0.1723 + seed) * 0.055)
    const weights = {
      fc1w: mk(256 * 498, 1), fc1b: mk(256, 2),
      fc2w: mk(32 * 256, 3), fc2b: mk(32, 4),
      fc3w: mk(32, 5), fc3b: mk(1, 6),
    }
    const engine = new BrowserNNUE('parity', weights.fc1w, weights.fc1b,
      weights.fc2w, weights.fc2b, weights.fc3w, weights.fc3b)
    const board = new Board(false)
    for (let ply = 0; ply < 90 && !board.isOver; ply++) {
      expect(engine.evaluate(board)).toBeCloseTo(referenceEval(weights, board), 10)
      const legal = board.legalMoves()
      if (!legal.length) break
      // Deterministic mix of flips, quiet moves and captures.
      const move = legal[(ply * 7 + 5) % legal.length]
      expect(board.play(move)).toBe(true)
    }
  })
})
