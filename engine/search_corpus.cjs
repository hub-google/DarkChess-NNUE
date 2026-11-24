const fs=require('node:fs');
const {Board,Piece}=require('../.shared-build/frontend/src/engine/board.js');
const {DarkChessEngine}=require('../.shared-build/frontend/src/engine/search.js');
const {loadNNUE}=require('../.shared-build/src/workers/model_runtime.js');
const {model}=loadNNUE();const engine=new DarkChessEngine(b=>model.evaluate(b));
let seed=0xa09123;
function rng(){seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}
const rows=[];
for(let game=0;game<12;game++){
 const board=new Board(false),path=[];
 for(let i=31;i>0;i--){const j=Math.floor(rng()*(i+1));[board.hidden[i],board.hidden[j]]=[board.hidden[j],board.hidden[i]];}
 for(let ply=0;ply<128&&!board.isOver;ply++){
  const depth=ply===0&&game===0?1:ply===12?1:ply===40||ply===80?2:0;
  if(depth){
   const analysis=engine.analyze(board,{maxDepth:depth,requireFullDepth:true,useChanceBounds:false});
   if(analysis.depth!==depth)throw Error('reference incomplete');
   const roots=Object.entries(analysis.rootValues);
   rows.push([path.length,...path.flat(),depth,roots.length,...roots.flatMap(([m,v])=>[m,v])].join(' '));
  }
  const legal=board.legalMoves();
  const capture=legal.filter(m=>!m.flip&&board.grid[m.to]!==Piece.EMPTY);
  const choices=capture.length&&rng()<.75?capture:legal;
  const m=choices[Math.floor(rng()*choices.length)];
  path.push([m.from*32+m.to,m.flip?board.hidden[m.from]:0]);
  if(!board.play(m))throw Error('reference illegal');
 }
}
fs.writeFileSync('engine/search_corpus.txt',rows.length+'\n'+rows.join('\n')+'\n');
console.log('Generated '+rows.length+' complete-depth reference positions');
