import { Board, Color, INITIAL_COUNTS, Piece } from './board'

export type TensorInfo = { shape: number[]; offset: number; length: number }
const reluClipped = (value: number) => Math.min(1, Math.max(0, value))

export class BrowserNNUE {
  constructor(
    public version: string,
    private fc1w: Float32Array, private fc1b: Float32Array,
    private fc2w: Float32Array, private fc2b: Float32Array,
    private fc3w: Float32Array, private fc3b: Float32Array,
  ) {}

  evaluate(board: Board) {
    // Build sparse public features in ascending index order directly.
    // This preserves the previous Float32 feature values and FC1 sum order,
    // while avoiding a 498-element tensor allocation and scan at each leaf.
    const active: number[] = []
    const values: number[] = []
    const add = (col: number, value: number) => {
      if (value !== 0) {
        active.push(col)
        values.push(value)
      }
    }
    for (let square = 0; square < 32; square++) {
      const piece = board.grid[square]
      if (piece === Piece.HIDDEN) add(square * 15 + 14, 1)
      else if (piece >= Piece.RED_KING && piece <= Piece.BLK_PAWN) {
        add(square * 15 + piece - 1, 1)
      }
    }
    for (let piece = 0; piece < 14; piece++) {
      add(480 + piece, Math.fround(board.remainingCounts[piece] / INITIAL_COUNTS[piece]))
    }
    if (board.turn === Color.RED) add(494, 1)
    else if (board.turn === Color.BLACK) add(495, 1)
    add(496, Math.fround(Math.min(board.halfMoveClock / 60, 1)))
    add(497, Math.fround(Math.min(board.repetitionCount() / 3, 1)))
    const hidden1 = new Float32Array(256)
    for (let row = 0; row < 256; row++) {
      let sum = this.fc1b[row]
      const base = row * 498
      for (let i = 0; i < active.length; i++) sum += this.fc1w[base + active[i]] * values[i]
      hidden1[row] = reluClipped(sum)
    }
    const hidden2 = new Float32Array(32)
    for (let row = 0; row < 32; row++) {
      let sum = this.fc2b[row]
      const base = row * 256
      for (let col = 0; col < 256; col++) sum += this.fc2w[base + col] * hidden1[col]
      hidden2[row] = reluClipped(sum)
    }
    let output = this.fc3b[0]
    for (let col = 0; col < 32; col++) output += this.fc3w[col] * hidden2[col]
    return Math.tanh(output)
  }
}

export function extractFeatures(board: Board) {
  const features = new Float32Array(498)
  board.grid.forEach((piece, square) => {
    if (piece === Piece.HIDDEN) features[square * 15 + 14] = 1
    else if (piece >= Piece.RED_KING && piece <= Piece.BLK_PAWN) features[square * 15 + piece - 1] = 1
  })
  for (let piece = 0; piece < 14; piece++) features[480 + piece] = board.remainingCounts[piece] / INITIAL_COUNTS[piece]
  if (board.turn === Color.RED) features[494] = 1
  else if (board.turn === Color.BLACK) features[495] = 1
  features[496] = Math.min(board.halfMoveClock / 60, 1)
  features[497] = Math.min(board.repetitionCount() / 3, 1)
  return features
}

