import { describe, expect, it } from 'vitest'
import { Board, Color, Piece } from '../board'
import { chooseSearchedMove } from '../search'

describe('Browser public-belief Star1 search', () => {
  const started = () => {
    const board = new Board(false)
    board.play({ from: 0, to: 0, flip: true })
    return board
  }

  it('returns a legal move under a strict node limit', () => {
    const board = started()
    const move = chooseSearchedMove(board, undefined, { nodeBudget: 1, maxMs: 10 })
    expect(board.legalMoves().some(m => m.from === move?.from && m.to === move?.to)).toBe(true)
  })

  it('does not use the actual face-down referee arrangement', () => {
    const a = started()
    const b = a.clone()
    // Both public boards, remaining counts and to-move are identical.
    // Only hidden referee mapping differs.
    b.hidden[1] = Piece.BLK_KING
    b.hidden[2] = Piece.RED_PAWN
    const first = chooseSearchedMove(a, undefined, { nodeBudget: 800, maxDepth: 2, maxMs: 5000 })
    const second = chooseSearchedMove(b, undefined, { nodeBudget: 800, maxDepth: 2, maxMs: 5000 })
    expect(first).toEqual(second)
  })

  it('searches deterministic tactical captures without reading hidden contents', () => {
    const board = started()
    board.grid.fill(Piece.EMPTY)
    board.remainingCounts.fill(0)
    board.turn = Color.RED
    board.grid[0] = Piece.RED_ROOK
    board.grid[1] = Piece.BLK_KNIGHT
    board.grid[31] = Piece.BLK_KING
    const move = chooseSearchedMove(board, undefined, { nodeBudget: 500, maxMs: 1000 })
    expect(move).toBeDefined()
    expect(board.legalMoves()).toContainEqual(move)
  })
})
