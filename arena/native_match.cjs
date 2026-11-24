// Equal wall-time search, frozen TS opponent, common model, common referee.
// Private deal is NEVER sent to either searcher. Only revealed moves cross the native protocol.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),zlib=require('node:zlib');
const {spawn}=require('node:child_process');
const {createInterface}=require('node:readline');
const {Board,Piece}=require('../.shared-build/frontend/src/engine/board.js');
const {loadNNUE}=require('../.shared-build/src/workers/model_runtime.js');
const arg=(name,def)=>{const i=process.argv.indexOf('--'+name);return i<0?def:process.argv[i+1]};
const pairs=Number(arg('pairs','16')),ms=Number(arg('ms','25')),seed=Number(arg('seed','24061101'));
const output=arg('output','arena/native.json'),base=path.resolve(arg('baseline-dir','frozen'));
const modelDir=path.resolve('frontend/public/models');
const exe=path.resolve(arg('exe','/tmp/dc-cli'));
const frozenSha='643218496863fef81612eab331c2b4d7a4255b12';
if(!Number.isSafeInteger(seed)||!Number.isInteger(pairs)||pairs<1||!Number.isFinite(ms)||ms<=0)throw Error('invalid experiment configuration');
const {execFileSync}=require('node:child_process');
if(execFileSync('git',['-C',base,'rev-parse','HEAD'],{encoding:'utf8'}).trim()!==frozenSha)throw Error('baseline is not frozen commit');
const {Board:OldBoard,Piece:OldPiece}=require(path.join(base,'.shared-build/frontend/src/engine/board.js'));
const {DarkChessEngine}=require(path.join(base,'.shared-build/frontend/src/engine/search.js'));
const {loadNNUE:loadOld}=require(path.join(base,'.shared-build/src/workers/model_runtime.js'));
const old=loadOld(modelDir),current=loadNNUE(modelDir);
const modelHash=crypto.createHash('sha256').update(fs.readFileSync(path.join(modelDir,'champion.bin'))).digest('hex');
if(old.version!==current.version)throw Error('model mismatch');
const frozen=new DarkChessEngine(b=>old.model.evaluate(b));
const report={schema:1,planned_pairs:pairs,pair_scores:[],games:[],incomplete_games:0,seed,budget_ms:ms,baseline_sha:frozenSha,candidate_sha:process.env.GITHUB_SHA||'local',model_sha256:modelHash,stats:[0,1].map(()=>({moves:0,nodes:0,depth_sum:0,depth_zero:0,search_ms:0,decision_ms:0,cpu_ms:0,max_search_ms:0,over_budget_2x:0}))};
let rngState=seed>>>0;
function rng(){rngState=(rngState+0x6D2B79F5)>>>0;let t=rngState;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}
const records=[];
function persist(){
 fs.writeFileSync(output+'.tmp',JSON.stringify(report,null,2));fs.renameSync(output+'.tmp',output);
 fs.writeFileSync(output+'.jsonl.gz.tmp',zlib.gzipSync(records.map(r=>JSON.stringify(r)).join('\n')+'\n'));
 fs.renameSync(output+'.jsonl.gz.tmp',output+'.jsonl.gz');
}
async function main(){
 const native=spawn(exe,[path.join(modelDir,'champion.bin')],{stdio:['pipe','pipe','inherit']});
 const lines=createInterface({input:native.stdout})[Symbol.asyncIterator]();
 let nativeError=null;
 native.on('error',e=>{nativeError=e;});
 native.stdin.on('error',e=>{nativeError=e;});
 async function nextLine(){
  let timer;
  try{
   const r=await Promise.race([lines.next(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('native response deadline')),Math.max(5000,ms*20));})]);
   if(r.done||nativeError)throw nativeError||Error('native process closed');
   return r.value;
  }finally{clearTimeout(timer);}
 }
 try{
  if(await nextLine()!=='READY')throw Error('native startup failed');
  async function play(bag,opening,first,pair){
   const board=new Board(false);board.hidden=bag.slice();
   const history=[],oldBoard=new OldBoard(false);oldBoard.hidden.fill(OldPiece.HIDDEN);
   function apply(m){
    const reveal=m.flip?board.hidden[m.from]:0;
    if(m.flip)oldBoard.hidden[m.from]=reveal;
    if(!board.play(m)||!oldBoard.play(m))throw Error('referee disagreement');
    history.push([m.from*32+m.to,reveal]);
    if(board.turn!==oldBoard.turn||board.winner!==oldBoard.winner||board.draw!==oldBoard.draw||board.grid.some((p,i)=>p!==oldBoard.grid[i]))throw Error('board state mismatch');
   }
   const color=bag[opening]<=Piece.RED_PAWN?0:1;
   apply({from:opening,to:opening,flip:true});
   for(let ply=1;ply<4096&&!board.isOver;ply++){
    const player=board.turn===color?first:1-first;
    const start=performance.now();let a,searchMs,cpuMs;
    if(player===0){
     const cpu=process.cpuUsage();
     const result=frozen.analyze(oldBoard,{maxDepth:12,nodeBudget:100000000,maxMs:ms});
     const usage=process.cpuUsage(cpu);
     searchMs=result.elapsedMs;cpuMs=(usage.user+usage.system)/1000;
     if(!result.move)throw Error('missing TS move');
     a={move:result.move.from*32+result.move.to,depth:result.depth,nodes:result.nodes};
    }else{
     native.stdin.write([history.length,12,100000000,ms,...history.flat()].join(' ')+'\n');
     a=JSON.parse(await nextLine());searchMs=a.ms;cpuMs=a.cpu_ms;
    }
    const elapsed=performance.now()-start;
    if(!Number.isInteger(a.move)||a.move<0||a.move>=1024||!Number.isInteger(a.depth)||a.depth<0||a.depth>12||!Number.isFinite(a.nodes)||a.nodes<0||a.nodes>100000001||!Number.isFinite(searchMs)||searchMs<0||!Number.isFinite(cpuMs)||cpuMs<0)throw Error('invalid engine response');
    const stat=report.stats[player];stat.moves++;stat.nodes+=a.nodes;stat.depth_sum+=a.depth;stat.depth_zero+=Number(a.depth===0);stat.search_ms+=searchMs;stat.decision_ms+=elapsed;stat.cpu_ms+=cpuMs;stat.max_search_ms=Math.max(stat.max_search_ms,searchMs);stat.over_budget_2x+=Number(searchMs>ms*2);
    const from=Math.floor(a.move/32),to=a.move%32;
    apply({from,to,flip:from===to});
   }
   if(!board.isOver)throw Error('incomplete game');
   const score=board.draw?0.5:((board.winner===color?first:1-first)===1?1:0);
   const id='v2-arena-'+seed+'-'+pair+'-'+first;
   records.push({id,ver:'v2.1.0-perpetual-chase',hid:bag,mov:history.map(h=>h[0]),res:board.draw?0:board.winner===0?1:-1,ply:history.length});
   return {first,score,plies:board.ply,id};
  }
  for(let i=0;i<pairs;i++){
   const bag=new Board(false).hidden.slice();
   for(let j=31;j>0;j--){const k=Math.floor(rng()*(j+1));[bag[j],bag[k]]=[bag[k],bag[j]];}
   const opening=Math.floor(rng()*32),order=rng()<0.5?[0,1]:[1,0];
   const deal_hash=crypto.createHash('sha256').update(JSON.stringify([bag,opening])).digest('hex');
   try{
    const games=[];for(const first of order)games.push(await play(bag,opening,first,i));
    report.games.push({pair:i,deal_hash,opening,order,games});report.pair_scores.push(games[0].score+games[1].score);
    persist();console.log('pair '+(i+1)+'/'+pairs+' score='+report.pair_scores.at(-1));
   }catch(e){report.incomplete_games++;report.error=String(e);persist();throw e;}
  }
 }finally{native.stdin.end();native.kill();lines.return?.();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
