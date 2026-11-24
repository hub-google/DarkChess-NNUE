/** Research WASM is opt-in; the existing TS engine remains the default. */
import { Board } from './board'
import { DarkChessEngine, type SearchLimits } from './search'
import { loadChampion } from './nnue'
import { loadWasmEngine, type WasmEngine } from './native-wasm'
import type { PublicMove } from './public-history'

type SearchMessage = { id: number; board: Board; limits: SearchLimits; history: PublicMove[]; backend: 'ts' | 'wasm' }
let champion: ReturnType<typeof loadChampion> | undefined
let wasm: Promise<WasmEngine> | undefined

self.onmessage = async (event: MessageEvent<SearchMessage>) => {
  const { id, limits, backend } = event.data
  try {
    const board = Object.assign(new Board(false), event.data.board)
    if (backend === 'wasm') {
      wasm ??= loadWasmEngine()
      const analysis = (await wasm).analyze(event.data.history, limits)
      if (event.data.history.length !== board.ply ||
          !board.legalMoves().some(m => m.from === analysis.move.from && m.to === analysis.move.to))
        throw new Error('WASM/public referee mismatch')
      self.postMessage({id, ...analysis, backend:'wasm'})
      return
    }
    champion ??= loadChampion()
    const model = await champion
    const engine = new DarkChessEngine(state => model.evaluate(state))
    const analysis = engine.analyze(board, limits)
    self.postMessage({ id, move: analysis.move, depth: analysis.depth,
      nodes: analysis.nodes, elapsedMs: analysis.elapsedMs, backend:'ts' })
  } catch (error) {
    self.postMessage({ id, error: String(error) })
  }
}
