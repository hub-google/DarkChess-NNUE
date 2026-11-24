import {describe,expect,it} from 'vitest'
import {readFileSync} from 'node:fs'
import {createHash} from 'node:crypto'
import {decodeNNUE,validateNNUEManifest} from '../nnue-core'
import {Board} from '../board'
const manifest=JSON.parse(readFileSync('public/models/champion.json','utf8'))
const bytes=readFileSync('public/models/champion.bin')
const hash=createHash('sha256').update(bytes).digest('hex')
describe('shared NNUE format',()=>{
  it('loads actual published model and produces finite evaluation',()=>{
    expect(Number.isFinite(decodeNNUE(manifest,bytes,hash).evaluate(new Board(false)))).toBe(true)
  })
  it('rejects wrong tensor shape, overlap, and binary hash',()=>{
    const bad=structuredClone(manifest);bad.tensors['fc1.weight'].shape=[498,256]
    expect(()=>validateNNUEManifest(bad)).toThrow()
    const overlapping=structuredClone(manifest);overlapping.tensors['fc1.bias'].offset=0
    expect(()=>validateNNUEManifest(overlapping)).toThrow()
    expect(()=>decodeNNUE(manifest,bytes,'0'.repeat(64))).toThrow()
  })
  it('rejects nonfinite values even in correctly checksummed files',()=>{
    const bad=Buffer.from(bytes);bad.writeFloatLE(NaN,0)
    const h=createHash('sha256').update(bad).digest('hex')
    expect(()=>decodeNNUE({...manifest,sha256:h},bad,h)).toThrow('Non-finite')
  })
})
