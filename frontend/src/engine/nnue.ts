import { Board, Color, INITIAL_COUNTS, Piece } from './board'

type TensorInfo = { shape: number[]; offset: number; length: number }
type Metadata = { format: number; inputSize: number; sha256: string; tensors: Record<string, TensorInfo> }

const tensor = (data: ArrayBuffer, info: TensorInfo) => new Float32Array(data, info.offset, info.length)
const reluClipped = (value: number) => Math.min(1, Math.max(0, value))

export class BrowserNNUE {
  constructor(
    public version: string,
    private fc1w: Float32Array, private fc1b: Float32Array,
    private fc2w: Float32Array, private fc2b: Float32Array,
    private fc3w: Float32Array, private fc3b: Float32Array,
  ) {}

  evaluate(board: Board) {
    const input = extractFeatures(board)
    const hidden1 = new Float32Array(256)
    for (let row = 0; row < 256; row++) {
      let sum = this.fc1b[row]
      const base = row * 498
      for (let col = 0; col < 498; col++) if (input[col]) sum += this.fc1w[base + col] * input[col]
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

export async function loadChampion() {
  const base = `${import.meta.env.BASE_URL}models/`
  const [metadataResponse, binaryResponse] = await Promise.all([
    fetch(`${base}champion.json`, { cache: 'no-cache' }),
    fetch(`${base}champion.bin`, { cache: 'no-cache' }),
  ])
  if (!metadataResponse.ok || !binaryResponse.ok) throw new Error('Champion model is unavailable')
  const metadata = await metadataResponse.json() as Metadata
  if (metadata.format !== 1 || metadata.inputSize !== 498) throw new Error('Unsupported champion model')
  const data = await binaryResponse.arrayBuffer()
  const t = metadata.tensors
  return new BrowserNNUE(
    metadata.sha256.slice(0, 8),
    tensor(data, t['fc1.weight']), tensor(data, t['fc1.bias']),
    tensor(data, t['fc2.weight']), tensor(data, t['fc2.bias']),
    tensor(data, t['fc3.weight']), tensor(data, t['fc3.bias']),
  )
}
