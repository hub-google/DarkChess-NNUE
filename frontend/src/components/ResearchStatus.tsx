import {useEffect,useState} from 'react'
type Evidence = {
  asOf:string
  engineMatch:{run:string;games:number;budget:number;score:number;decision:string}[]
  selfPlay:{run:string;games:number;min25:number;max25:number;min100:number;max100:number}
  browser:{run:string;positions:number;rootScores:number;maxError:number}
  training:{run:string;arms:string[];updatesPerArm:number;state:string}
}
export function ResearchStatus(){
 const [data,setData]=useState<Evidence|null>(null)
 const [error,setError]=useState(false)
 useEffect(()=>{
  let active=true
  fetch(import.meta.env.BASE_URL+'research-status.json',{cache:'no-store'})
   .then(r=>{if(!r.ok)throw Error('research snapshot unavailable');return r.json() as Promise<Evidence>})
   .then(r=>{if(active)setData(r)})
   .catch(()=>{if(active)setError(true)})
  return ()=>{active=false}
 },[])
 return <section className="training-card" aria-label="V2 研究實測">
  <div className="training-heading"><strong>V2 研究實測</strong><small>{data ? data.asOf+' 快照' : error?'資料無法讀取':'讀取中'}</small></div>
  {data && <>
   {data.engineMatch.map(r=><a className="metric-row" key={r.budget} href={r.run} target="_blank" rel="noreferrer">
    <span><small>對舊版 · 每手 {r.budget}ms</small><strong>{(r.score*100).toFixed(2)}%</strong></span>
    <span className="metric-detail">{r.games} 局 · {r.decision==='SUPERIOR'?'通過優勢門檻':'未通過門檻'}</span>
   </a>)}
   <a className="metric-row" href={data.selfPlay.run} target="_blank" rel="noreferrer">
    <span><small>合法原生自我對弈</small><strong>{data.selfPlay.games} 局</strong></span>
    <span className="metric-detail">25ms 每台每小時 {data.selfPlay.min25}–{data.selfPlay.max25} 局</span>
   </a>
   <a className="metric-row" href={data.browser.run} target="_blank" rel="noreferrer">
    <span><small>WASM 實機與根分數驗證</small><strong>{data.browser.rootScores} 個</strong></span>
    <span className="metric-detail">Chromium 通過</span>
   </a>
   <a className="metric-row" href={data.training.run} target="_blank" rel="noreferrer">
    <span><small>新舊資料訓練消融</small><strong>{data.training.arms.length} 組</strong></span>
    <span className="metric-detail">{data.training.state}</span>
   </a>
   <p className="honesty">這是研究快照，並非目前正式冠軍的棋力或即時資料總量。點選各列查閱原始 Actions。</p>
  </>}
 </section>
}
