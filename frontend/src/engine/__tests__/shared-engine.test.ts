import { describe, expect, it } from 'vitest'
import { Board, Piece, Color } from '../board'
import { DarkChessEngine, chooseSearchedMove, type SearchProgress } from '../search'

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
    const progress: SearchProgress[] = []
    const analysis = new DarkChessEngine().analyze(b, {
      maxDepth: 4, requireFullDepth: true,
      onRootCompleted: m => completed.push(m.from * 32 + m.to),
      onProgress: s => progress.push(s),
    })
    expect(analysis.depth).toBe(4)
    expect(Object.keys(analysis.rootValues)).toHaveLength(b.legalMoves().length)
    expect(completed).toHaveLength(b.legalMoves().length)
    expect(progress.at(-1)?.rootsDone).toBe(b.legalMoves().length)
    expect(progress.at(-1)?.depth).toBe(4)
    expect(b.legalMoves()).toContainEqual(analysis.move)
    const resumed = new DarkChessEngine().analyze(b, {
      maxDepth: 4, requireFullDepth: true, resumeRootValues: analysis.rootValues,
    })
    expect(resumed.move).toEqual(analysis.move)
    expect(resumed.depth).toBe(4)
    expect(resumed.nodes).toBe(0)
  })
  it('matches unbounded chance search on full-depth early/mid positions', () => {
    const positions: Board[] = []
    for (let variant = 0; variant < 4; variant++) {
      const b = new Board(false)
      b.grid.fill(Piece.EMPTY)
      b.remainingCounts.fill(0)
      b.turn = variant % 2 ? Color.BLACK : Color.RED
      b.grid[0] = Piece.RED_ROOK
      b.grid[31] = Piece.BLK_ROOK
      b.grid[variant + 1] = Piece.HIDDEN
      b.grid[24 - variant] = Piece.HIDDEN
      b.hidden[variant + 1] = Piece.RED_PAWN
      b.hidden[24 - variant] = Piece.BLK_PAWN
      b.remainingCounts[6] = 1
      b.remainingCounts[13] = 1
      positions.push(b)
    }
    for (const board of positions) {
      const engine = new DarkChessEngine()
      const reference = engine.analyze(board, {
        maxDepth: 4, requireFullDepth: true, useChanceBounds: false,
      })
      const progress: SearchProgress[] = []
      const optimized = engine.analyze(board, {
        maxDepth: 4, requireFullDepth: true, useChanceBounds: true,
        onProgress: p => progress.push(p),
      })
      expect(optimized.depth).toBe(4)
      expect(optimized.move).toEqual(reference.move)
      expect(Object.keys(optimized.rootValues)).toEqual(Object.keys(reference.rootValues))
      for (const key of Object.keys(reference.rootValues)) {
        expect(optimized.rootValues[Number(key)]).toBeCloseTo(reference.rootValues[Number(key)], 9)
      }
      expect(optimized.nodes).toBeLessThanOrEqual(reference.nodes)
      expect(progress.at(-1)?.chancePrunes).toBeGreaterThanOrEqual(0)
    }
  })

  it('keeps six plies and every root option in selective CPU mode', () => {
    const b = new Board(false)
    b.grid.fill(Piece.EMPTY)
    b.remainingCounts.fill(0)
    b.turn = Color.RED
    b.grid[0] = Piece.RED_ROOK
    b.grid[31] = Piece.BLK_ROOK
    b.grid[1] = Piece.HIDDEN
    b.grid[24] = Piece.HIDDEN
    b.hidden[1] = Piece.RED_PAWN
    b.hidden[24] = Piece.BLK_PAWN
    b.remainingCounts[6] = 1
    b.remainingCounts[13] = 1
    const analysis = new DarkChessEngine().analyze(b, {
      maxDepth: 6, requireFullDepth: true, innerMoveLimit: 2,
    })
    expect(analysis.depth).toBe(6)
    expect(Object.keys(analysis.rootValues)).toHaveLength(b.legalMoves().length)
    expect(b.legalMoves()).toContainEqual(analysis.move)
  })

  it('uses only publicly known probabilities for sampled six-ply search', () => {
    const b = new Board(false)
    b.grid.fill(Piece.EMPTY)
    b.remainingCounts.fill(0)
    b.turn = Color.RED
    b.grid[0] = Piece.RED_ROOK
    b.grid[31] = Piece.BLK_ROOK
    const hidden = [1, 2, 20, 22]
    const pieces = [Piece.RED_PAWN, Piece.BLK_PAWN, Piece.RED_CANNON, Piece.BLK_CANNON]
    for (let i = 0; i < hidden.length; i++) {
      b.grid[hidden[i]] = Piece.HIDDEN
      b.hidden[hidden[i]] = pieces[i]
      b.remainingCounts[pieces[i] - 1]++
    }
    const alternate = b.clone()
    for (let i = 0; i < hidden.length; i++) {
      alternate.hidden[hidden[i]] = pieces[(i + 1) % hidden.length]
    }
    const limits = {maxDepth: 4, requireFullDepth: true, innerMoveLimit: 2, chanceSamples: 2}
    const first = new DarkChessEngine().analyze(b, limits)
    const again = new DarkChessEngine().analyze(b, limits)
    const privateShuffle = new DarkChessEngine().analyze(alternate, limits)
    expect(first.depth).toBe(4)
    expect(Object.keys(first.rootValues)).toHaveLength(b.legalMoves().length)
    expect(b.legalMoves()).toContainEqual(first.move)
    expect(first.move).toEqual(again.move)
    expect(first.rootValues).toEqual(again.rootValues)
    expect(first.rootValues).toEqual(privateShuffle.rootValues)
  })

  it('reports complete search depth rather than requested depth', () => {
    const b = new Board(false)
    b.play({from: 0, to: 0, flip: true})
    const result = new DarkChessEngine().analyze(b, {nodeBudget: 1, maxDepth: 12, maxMs: 50})
    expect(result.depth).toBe(0)
    expect(result.move).toBeDefined()
  })
})
