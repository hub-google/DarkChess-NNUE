import { describe, expect, it } from 'vitest'
import { Board, Piece } from '../board'
import { DarkChessEngine, chooseSearchedMove } from '../search'

describe('One shared decision engine for website and self-play', () => {
  it('produces the identical legal choice through class and compatibility function', () => {
    const b = new Board(false)
    b.play({from: 0, to: 0, flip: true})
    const limits = {maxDepth: 2, maxMs: 5000, nodeBudget: 2500}
    const engine = new DarkChessEngine()
    const analysis = engine.analyze(b, limits)
    expect(analysis.move).toEqual(chooseSearchedMove(b, undefined, limits))
    expect(b.legalMoves()).toContainEqual(analysis.move)
    expect(analysis.depth).toBeGreaterThanOrEqual(0)
    expect(analysis.nodes).toBeGreaterThan(0)
  })
  it('never looks at a hidden referee piece during analysis', () => {
    const a = new Board(false)
    a.play({from: 0, to: 0, flip: true})
    const b = a.clone()
    b.hidden[5] = Piece.BLK_KING
    b.hidden[6] = Piece.RED_PAWN
    const e = new DarkChessEngine()
    expect(e.analyze(a, {maxDepth: 2, maxMs: 5000, nodeBudget: 1600}).move)
      .toEqual(e.analyze(b, {maxDepth: 2, maxMs: 5000, nodeBudget: 1600}).move)
  })
  it('reports complete search depth rather than requested depth', () => {
    const b = new Board(false)
    b.play({from: 0, to: 0, flip: true})
    const result = new DarkChessEngine().analyze(b, {nodeBudget: 1, maxDepth: 12, maxMs: 50})
    expect(result.depth).toBe(0)
    expect(result.move).toBeDefined()
  })
})
