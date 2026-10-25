/** Web Worker runs the SAME DarkChessEngine class as Node self-play.
 * It uses the public-belief game state and the latest verified champion.
 */
import { Board } from './board'
import { DarkChessEngine, type SearchLimits } from './search'
import { loadChampion } from './nnue'

type SearchMessage = { id: number; board: Board; limits: SearchLimits }
const champion = loadChampion().catch(() => null)

self.onmessage = async (event: MessageEvent<SearchMessage>) => {
  const { id, limits } = event.data
  try {
    const model = await champion
    // postMessage strips prototypes: restore Board methods without reshuffling
    // any piece identities or looking at the referee's unrevealed locations.
    const board = Object.assign(new Board(false), event.data.board)
    const engine = new DarkChessEngine(model ? state => model.evaluate(state) : undefined)
    const analysis = engine.analyze(board, limits)
    self.postMessage({ id, move: analysis.move, depth: analysis.depth,
      nodes: analysis.nodes, elapsedMs: analysis.elapsedMs })
  } catch (error) {
    self.postMessage({ id, error: String(error) })
  }
}
