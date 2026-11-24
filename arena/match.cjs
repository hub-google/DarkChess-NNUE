// Fixed-horizon research match; no production writes. First flip is a fixed referee action.
const fs=require('node:fs');
const {Board,Color,Piece}=require('../.shared-build/frontend/src/engine/board.js');
const {DarkChessEngine}=require('../.shared-build/frontend/src/engine/search.js');
const {loadNNUE}=require('../.shared-build/src/workers/model_runtime.js');
const arg=(name,def)=>{const i=process.argv.indexOf('--'+name);return i<0?def:process.argv[i+1]};
const pairs=Number(arg('pairs','16')), ms=Number(arg('ms','25')), initialSeed=Number(arg('seed','19660217'));
const output=arg('output','arena/results.json'), candidateProfile=arg('candidate-profile','full');
const basePath=arg('baseline','frontend/public/models'), candidatePath=arg('candidate',basePath);
if(!Number.isInteger(pairs)||pairs<1||!Number.isFinite(ms)||ms<1||!Number.isInteger(initialSeed)||!['full','selective'].includes(candidateProfile))throw Error('Bad config');
let seed=initialSeed>>>0;
function rng(){seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}
const b=loadNNUE(basePath),c=loadNNUE(candidatePath);
const engines=[new DarkChessEngine(x=>b.model.evaluate(x)),new DarkChessEngine(x=>c.model.evaluate(x))];
const report={planned_pairs:pairs,pair_scores:[],games:[],incomplete_games:0,seed:initialSeed,budget_ms:ms,baseline:b.version,candidate:c.version,candidate_profile:candidateProfile,stats:[{moves:0,depth_zero:0,wall_ms:0,cpu_ms:0,max_ms:0},{moves:0,depth_zero:0,wall_ms:0,cpu_ms:0,max_ms:0}]};
function persist(){fs.writeFileSync(output+'.tmp',JSON.stringify(report,null,2));fs.renameSync(output+'.tmp',output);}
function play(bag,opening,first){
 const board=new Board(false);board.hidden=bag.slice();
 const color=bag[opening]<=Piece.RED_PAWN?Color.RED:Color.BLACK;
 if(!board.play({from:opening,to:opening,flip:true}))throw Error('Opening illegal');
 for(let ply=1;ply<4096&&!board.isOver;ply++){
  const player=board.turn===color?first:1-first;
  const publicBoard=board.clone();publicBoard.hidden=Array(32).fill(Piece.HIDDEN);
  const lim={maxDepth:12,nodeBudget:100000000,maxMs:ms};
  if(player===1&&candidateProfile==='selective'){lim.innerMoveLimit=2;lim.chanceSamples=3;}
  const start=performance.now(),cpu=process.cpuUsage();
  const a=engines[player].analyze(publicBoard,lim),wall=performance.now()-start,usage=process.cpuUsage(cpu);
  const s=report.stats[player];s.moves++;s.depth_zero+=Number(a.depth===0);s.wall_ms+=wall;s.cpu_ms+=(usage.user+usage.system)/1000;s.max_ms=Math.max(s.max_ms,wall);
  if(!a.move||!board.play(a.move))throw Error('Illegal or missing move at '+ply);
 }
 if(!board.isOver)throw Error('Incomplete game');
 return {score:board.draw?.5:((board.winner===color?first:1-first)===1?1:0),plies:board.ply};
}
for(let i=0;i<pairs;i++){
 const bag=new Board(false).hidden.slice();
 for(let j=31;j>0;j--){const k=Math.floor(rng()*(j+1));[bag[j],bag[k]]=[bag[k],bag[j]];}
 const opening=Math.floor(rng()*32);
 try{
  // Randomize execution order while retaining matched deals and swapped ownership.
  const order=rng()<.5?[0,1]:[1,0],games=order.map(first=>({first,...play(bag,opening,first)}));
  report.games.push({pair:i,opening,order,games});report.pair_scores.push(games[0].score+games[1].score);
  persist();console.log('pair '+(i+1)+'/'+pairs+' score='+report.pair_scores.at(-1));
 }catch(e){report.incomplete_games++;report.error=String(e);persist();throw e;}
}
