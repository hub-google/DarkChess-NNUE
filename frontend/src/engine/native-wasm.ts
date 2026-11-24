import {loadChampionBytes} from './nnue'
import type {PublicMove} from './public-history'
import type {SearchLimits} from './search'

type RawResult = {move: number; depth: number; nodes: number; ms: number; value: number; roots: Record<string,number>; error?: string}
type Module = {
  FS: {writeFile(path:string,bytes:Uint8Array):void}
  ccall(name:string,result:string,args:string[],values:unknown[]):unknown
}
export type WasmEngine = { analyze(history:PublicMove[],limits:SearchLimits): {
  move:{from:number;to:number;flip:boolean}; depth:number; nodes:number; elapsedMs:number; value:number
} }

export async function loadWasmEngine(): Promise<WasmEngine> {
  const url = new URL(`${import.meta.env.BASE_URL}engine/darkchess.mjs`, self.location.origin).href
  const imported = await import(/* @vite-ignore */ url) as {default:(options:{locateFile:(name:string)=>string})=>Promise<Module>}
  const [m, weights] = await Promise.all([imported.default({locateFile:name=>new URL(name,url).href}),loadChampionBytes()])
  m.FS.writeFile('/champion.bin',weights.bytes)
  if (m.ccall('dc_init','number',[],[]) !== 1) throw new Error('WASM model initialization failed')
  return {analyze(history,limits){
    const request=[history.length,limits.maxDepth ?? 12,limits.nodeBudget ?? 100000000,limits.maxMs ?? 250,...history.flat()].join(' ')
    const a=JSON.parse(String(m.ccall('dc_analyze','string',['string'],[request]))) as RawResult
    if(a.error||!Number.isInteger(a.move)||a.move<0||a.move>=1024||!Number.isInteger(a.depth)||a.depth<0||
       !Number.isFinite(a.nodes)||!Number.isFinite(a.ms)||!Number.isFinite(a.value)) throw new Error(a.error||'Invalid WASM response')
    const from=Math.floor(a.move/32),to=a.move%32
    return {move:{from,to,flip:from===to},depth:a.depth,nodes:a.nodes,elapsedMs:a.ms,value:a.value}
  }}
}
