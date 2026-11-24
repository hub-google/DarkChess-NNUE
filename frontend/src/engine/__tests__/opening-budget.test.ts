import {describe, expect, it} from 'vitest'
import {Board, Piece} from '../board'
import {DarkChessEngine} from '../search'
describe('first flip budget handling', () => {
  it('returns a legal fallback without throwing or claiming completed depth', () => {
    const b = new Board(false)
    const a = new DarkChessEngine().analyze(b, {maxDepth: 12, nodeBudget: 1, maxMs: 1000})
    expect(a.depth).toBe(0)
    expect(a.rootValues).toEqual({})
    expect(b.legalMoves()).toContainEqual(a.move)
  })
  it('preserves last complete opening iteration when deeper search runs out of budget', () => {
    const b = new Board(false)
    const e = new DarkChessEngine()
    const shallow = e.analyze(b, {maxDepth: 1, requireFullDepth: true})
    const a = e.analyze(b, {maxDepth: 12, nodeBudget: shallow.nodes + 1, maxMs: 5000})
    expect(a.depth).toBe(1)
    expect(a.rootValues).toEqual(shallow.rootValues)
    expect(a.move).toEqual(shallow.move)
  })
  it('never uses true hidden layout for first reveal analysis', () => {
    const a = new Board(false), b = a.clone()
    b.hidden.fill(Piece.BLK_PAWN)
    const e = new DarkChessEngine()
    expect(e.analyze(a,{maxDepth:1,requireFullDepth:true}).rootValues)
      .toEqual(e.analyze(b,{maxDepth:1,requireFullDepth:true}).rootValues)
  })
})

describe('public search boundary',()=>{
  it('does not pass true hidden identities to custom evaluators',()=>{
    const b=new Board(false)
    const original=b.hidden.slice()
    const engine=new DarkChessEngine(state=>{
      for(let i=0;i<32;i++) if(state.grid[i]===Piece.HIDDEN) expect(state.hidden[i]).toBe(Piece.HIDDEN)
      return 0
    })
    engine.evaluate(b)
    engine.analyze(b,{maxDepth:1,requireFullDepth:true})
    expect(b.hidden).toEqual(original)
  })
})
