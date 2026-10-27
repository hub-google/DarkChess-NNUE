import { describe, it, expect } from 'vitest'
import { Board, Color, Piece, colorOf, type Move } from '../board'

const id = (m: Move) => m.from * 32 + m.to
function originalLegalMoves(board: Board): number[] {
  if (board.isOver) return []
  const moves: Move[] = []
  for (let i = 0; i < 32; i++) {
    if (board.grid[i] === Piece.HIDDEN) moves.push({from: i, to: i, flip: true})
  }
  if (board.turn === Color.NONE) return moves.map(id)
  for (let from = 0; from < 32; from++) {
    if (colorOf(board.grid[from]) !== board.turn) continue
    for (let to = 0; to < 32; to++) {
      if (board.canMove(from, to)) moves.push({from, to})
    }
  }
  return moves.map(id)
}
const next = (seed: number) => (Math.imul(seed, 1664525) + 1013904223) >>> 0

describe('CPU board optimization parity', () => {
  it('preserves every possible move and clone state across randomized real games', () => {
    let seed = 947182
    let positions = 0
    for (let game = 0; game < 35; game++) {
      const board = new Board(false)
      for (let index = 31; index > 0; index--) {
        seed = next(seed)
        const target = seed % (index + 1)
        ;[board.hidden[index], board.hidden[target]] = [board.hidden[target], board.hidden[index]]
      }
      for (let ply = 0; ply < 115 && !board.isOver; ply++) {
        const expected = originalLegalMoves(board)
        expect(board.legalMoves().map(id)).toEqual(expected)
        const clone = board.clone()
        expect(clone.legalMoves().map(id)).toEqual(expected)
        expect(clone.grid).toEqual(board.grid)
        expect(clone.hidden).toEqual(board.hidden)
        expect(clone.remainingCounts).toEqual(board.remainingCounts)
        expect(clone.chaseThreats).toEqual(board.chaseThreats)
        expect(clone.history).toEqual(board.history)
        positions++
        if (expected.length === 0) break
        seed = next(seed)
        expect(board.play(board.legalMoves()[seed % expected.length])).toBe(true)
      }
    }
    expect(positions).toBeGreaterThan(250)
  })
})
