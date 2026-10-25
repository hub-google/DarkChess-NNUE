import { describe, expect, it } from 'vitest'
import { Board, Piece, Color } from '../board'
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
  it('rejects budget limits while requiring full depth', () => {
    const b = new Board(false)
    b.play({from: 0, to: 0, flip: true})
    expect(() => new DarkChessEngine().analyze(b,
      {maxDepth: 12, requireFullDepth: true, nodeBudget: 100}))
      .toThrow('Complete-depth search must not have a node or time cutoff')
  })
  it('finishes every legal root to the exact depth with no fallback', () => {
    const b = new Board(false)
    b.grid.fill(Piece.EMPTY)
    b.remainingCounts.fill(0)
    b.turn = Color.RED
    b.grid[0] = Piece.RED_ROOK
    b.grid[1] = Piece.BLK_PAWN
    b.grid[31] = Piece.BLK_KING
    const completed: number[] = []
    const analysis = new DarkChessEngine().analyze(b, {
      maxDepth: 4, requireFullDepth: true,
      onRootCompleted: m => completed.push(m.from * 32 + m.to),
    })
    expect(analysis.depth).toBe(4)
    expect(Object.keys(analysis.rootValues)).toHaveLength(b.legalMoves().length)
    expect(completed).toHaveLength(b.legalMoves().length)
    expect(b.legalMoves()).toContainEqual(analysis.move)
    const resumed = new DarkChessEngine().analyze(b, {
      maxDepth: 4, requireFullDepth: true, resumeRootValues: analysis.rootValues,
    })
    expect(resumed.move).toEqual(analysis.move)
    expect(resumed.depth).toBe(4)
    expect(resumed.nodes).toBe(0)
  })
  it('reports complete search depth rather than requested depth', () => {
    const b = new Board(false)
    b.play({from: 0, to: 0, flip: true})
    const result = new DarkChessEngine().analyze(b, {nodeBudget: 1, maxDepth: 12, maxMs: 50})
    expect(result.depth).toBe(0)
    expect(result.move).toBeDefined()
  })
})
