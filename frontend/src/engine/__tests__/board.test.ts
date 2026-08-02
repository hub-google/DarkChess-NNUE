import { beforeEach, describe, expect, it } from 'vitest'
import { Board, Color, Piece } from '../board'
import { extractFeatures } from '../nnue'

describe('DarkChess Board Engine', () => {
  let board: Board

  beforeEach(() => { board = new Board(false) })

  it('initializes with the standard 32-piece inventory', () => {
    expect(board.hiddenPieces).toHaveLength(32)
    expect(board.grid.every(piece => piece === Piece.HIDDEN)).toBe(true)
    expect(board.remainingCounts).toEqual([1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5])
  })

  it('assigns the first revealed color to the player and passes the turn', () => {
    board.hidden[0] = Piece.RED_KING
    expect(board.flipPiece(0, 0)).toBe(Piece.RED_KING)
    expect(board.turn).toBe(Color.BLACK)
    expect(board.remainingCounts[0]).toBe(0)
  })

  it('uses the same king/pawn exception as the training engine', () => {
    board.grid.fill(Piece.EMPTY); board.turn = Color.RED
    board.grid[0] = Piece.RED_KING; board.grid[1] = Piece.BLK_PAWN
    expect(board.canMove(0, 1)).toBe(false)
    board.grid[0] = Piece.RED_PAWN; board.grid[1] = Piece.BLK_KING
    expect(board.canMove(0, 1)).toBe(true)
  })

  it('requires exactly one screen for a cannon capture', () => {
    board.grid.fill(Piece.EMPTY); board.turn = Color.RED
    board.grid[0] = Piece.RED_CANNON; board.grid[3] = Piece.BLK_ROOK
    expect(board.canMove(0, 3)).toBe(false)
    board.grid[1] = Piece.HIDDEN
    expect(board.canMove(0, 3)).toBe(true)
    board.grid[2] = Piece.RED_PAWN
    expect(board.canMove(0, 3)).toBe(false)
  })

  it('allows a cannon to move one square left into an empty square', () => {
    board.grid.fill(Piece.EMPTY); board.turn = Color.RED
    board.grid[28] = Piece.RED_CANNON
    expect(board.canMove(28, 27)).toBe(true)
  })

  it('records captured pieces outside the board', () => {
    board.grid.fill(Piece.EMPTY); board.turn = Color.RED
    board.grid[0] = Piece.RED_ROOK; board.grid[1] = Piece.BLK_KNIGHT
    expect(board.play({ from: 0, to: 1 })).toBe(true)
    expect(board.captured).toEqual([Piece.BLK_KNIGHT])
  })

  it('allows a chase that keeps advancing into new positions', () => {
    board.grid.fill(Piece.EMPTY); board.turn = Color.RED
    board.grid[8] = Piece.RED_GUARD; board.grid[1] = Piece.BLK_ROOK

    expect(board.play({ from: 8, to: 0 })).toBe(true) // first chase
    expect(board.play({ from: 1, to: 2 })).toBe(true)
    expect(board.play({ from: 0, to: 1 })).toBe(true) // second chase
    expect(board.play({ from: 2, to: 3 })).toBe(true)

    expect(board.canMove(1, 2)).toBe(true)
  })

  it('forbids the chase move that would repeat the same route a third time', () => {
    board.grid.fill(Piece.EMPTY); board.turn = Color.RED
    board.grid[8] = Piece.RED_GUARD; board.grid[1] = Piece.BLK_ROOK
    const route = [
      { from: 8, to: 0 },
      { from: 1, to: 9 }, { from: 0, to: 1 }, { from: 9, to: 8 }, { from: 1, to: 9 },
      { from: 8, to: 0 }, { from: 9, to: 8 }, { from: 0, to: 1 },
    ]
    for (const move of route) expect(board.play(move)).toBe(true)
    expect(board.canMove(8, 0)).toBe(false)
    expect(board.canMove(8, 9)).toBe(false) // changing direction still continues the long chase
    expect(board.play({ from: 8, to: 0 })).toBe(false)
  })

  it('does not adjudicate an ordinary threefold repetition as a draw', () => {
    board.grid.fill(Piece.EMPTY); board.turn = Color.RED
    board.grid[0] = Piece.RED_ROOK; board.grid[31] = Piece.BLK_ROOK
    for (const move of [
      { from: 0, to: 1 }, { from: 31, to: 30 }, { from: 1, to: 0 }, { from: 30, to: 31 },
      { from: 0, to: 1 }, { from: 31, to: 30 }, { from: 1, to: 0 }, { from: 30, to: 31 },
    ]) expect(board.play(move)).toBe(true)
    expect(board.draw).toBe(false)
  })

  it('exports the same 498 public features used during training', () => {
    board.grid.fill(Piece.EMPTY); board.grid[0] = Piece.RED_KING; board.grid[1] = Piece.HIDDEN
    board.remainingCounts[0] = 0; board.turn = Color.BLACK; board.halfMoveClock = 30
    const features = extractFeatures(board)
    expect(features).toHaveLength(498)
    expect(features[0]).toBe(1)
    expect(features[29]).toBe(1)
    expect(features[480]).toBe(0)
    expect(features[495]).toBe(1)
    expect(features[496]).toBe(0.5)
  })
})
