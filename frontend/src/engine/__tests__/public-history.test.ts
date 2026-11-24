import {describe,it,expect} from 'vitest'
import {Board,Piece} from '../board'
import {applyPublicMove,type PublicMove} from '../public-history'
describe('public move transport',()=>{
  it('records only revealed facts and excludes failed moves',()=>{
    const b=new Board(false), history:PublicMove[]=[]
    expect(applyPublicMove(b,{from:0,to:0,flip:true},history)).toBe(true)
    expect(history).toEqual([[0,b.grid[0]]])
    expect(b.publicClone().hidden.every(x=>x===Piece.HIDDEN)).toBe(true)
    expect(applyPublicMove(b,{from:0,to:0,flip:true},history)).toBe(false)
    expect(history).toHaveLength(1)
  })
})
