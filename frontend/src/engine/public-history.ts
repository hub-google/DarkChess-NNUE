import { Board, type Move, Piece } from './board'
/** [move id, publicly revealed piece]; 0 for non-flip. No private bag. */
export type PublicMove = [number, number]
export function applyPublicMove(board: Board, move: Move, history: PublicMove[]): boolean {
  if (!board.play(move)) return false
  history.push([move.from * 32 + move.to, move.flip ? board.grid[move.from] : Piece.EMPTY])
  return true
}
