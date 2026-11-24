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


/** One format contract for browser and native Node transport. */
export type NNUEManifest = {
  format: number; inputSize: number; bytes: number; sha256: string
  tensors: Record<string, TensorInfo>
}
const tensorSchema = [
  ['fc1.weight', [256,498]], ['fc1.bias', [256]],
  ['fc2.weight', [32,256]], ['fc2.bias', [32]],
  ['fc3.weight', [1,32]], ['fc3.bias', [1]],
] as const
export function validateNNUEManifest(input: unknown): NNUEManifest {
  if (typeof input !== 'object' || input === null) throw new Error('Invalid NNUE manifest')
  const m = input as NNUEManifest
  if (m.format !== 1 || m.inputSize !== 498 || typeof m.sha256 !== 'string' ||
      !/^[0-9a-f]{64}$/.test(m.sha256) || !m.tensors) throw new Error('Unsupported NNUE manifest')
  let offset = 0
  for (const [key,shape] of tensorSchema) {
    const t = m.tensors[key]
    let length: number = 1
    for (const dimension of shape) length *= dimension
    if (!t || !Array.isArray(t.shape) || t.shape.length !== shape.length ||
        !t.shape.every((v,i)=>v === shape[i]) || t.offset !== offset || t.length !== length)
      throw new Error('Invalid NNUE tensor: ' + key)
    offset += 4*length
  }
  if (m.bytes !== offset) throw new Error('Invalid NNUE manifest byte count')
  return m
}
export function decodeNNUE(input: unknown, bytes: Uint8Array, actualHash: string): BrowserNNUE {
  const m = validateNNUEManifest(input)
  if (bytes.byteLength !== m.bytes || actualHash !== m.sha256) throw new Error('NNUE binary checksum/size mismatch')
  const view = new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength)
  const tensors = tensorSchema.map(([key])=>{
    const t = m.tensors[key], result = new Float32Array(t.length)
    for (let i=0;i<t.length;i++){
      const value = view.getFloat32(t.offset+4*i,true)
      if (!Number.isFinite(value)) throw new Error('Non-finite NNUE weight: '+key)
      result[i]=value
    }
    return result
  })
  return new BrowserNNUE(m.sha256.slice(0,12),tensors[0],tensors[1],tensors[2],tensors[3],tensors[4],tensors[5])
}
