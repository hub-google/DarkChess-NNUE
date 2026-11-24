const fs=require('node:fs');
const {Board,Color,Piece}=require('../.shared-build/frontend/src/engine/board.js');
const {loadNNUE}=require('../.shared-build/src/workers/model_runtime.js');
const {model}=loadNNUE();
let seed=0x24a76121;
function rng(){seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}
const games=[];
function record(b,choose,limit){
 const header=[b.turn,b.halfMoveClock,...b.grid,...b.remainingCounts];
 const rows=[];
 const chase=c=>[c[0],c[1],c[2],c[3],c[4].length,...c[4]];
 for(let ply=0;ply<=limit;ply++){
  const legal=b.legalMoves();
  const next=b.isOver||ply===limit?null:choose(b,legal,ply);
  const m=next?next.from*32+next.to:-1;
  const reveal=next?.flip?b.hidden[next.from]:0;
  const row=[b.turn,b.winner,+b.draw,b.halfMoveClock,b.repetitionCount(),model.evaluate(b),
    ...b.grid,...b.tokenAtSquare,...b.remainingCounts,b.captured.length,...b.captured,
    +(b.pendingChase!==null),...(b.pendingChase?chase(b.pendingChase):[]),
    b.chaseThreats.length,...b.chaseThreats.flatMap(chase),
    legal.length,...legal.map(x=>x.from*32+x.to),m,reveal];
  const accepted=next?b.play(next):false;
  rows.push([...row,+accepted].join(' '));
  if(!next)break;
 }
 games.push([rows.length,...header].join(' ')+'\n'+rows.join('\n'));
}
function synthetic(red,black){
 const b=new Board(false);b.grid.fill(Piece.EMPTY);b.remainingCounts.fill(0);b.turn=Color.RED;
 b.grid[red[0]]=red[1];b.grid[black[0]]=black[1];return b;
}
const chaseRoute=[[8,0],[1,9],[0,1],[9,8],[1,9],[8,0],[9,8],[0,1],[8,0]];
record(synthetic([8,Piece.RED_GUARD],[1,Piece.BLK_ROOK]),(_,__,i)=>chaseRoute[i]?{from:chaseRoute[i][0],to:chaseRoute[i][1]}:null,chaseRoute.length);
const quietRoute=[[0,1],[31,30],[1,0],[30,31]];
record(synthetic([0,Piece.RED_ROOK],[31,Piece.BLK_ROOK]),(_,__,i)=>({from:quietRoute[i%4][0],to:quietRoute[i%4][1]}),61);
for(let game=0;game<64;game++){
 const b=new Board(false);
 for(let i=31;i>0;i--){let j=Math.floor(rng()*(i+1));[b.hidden[i],b.hidden[j]]=[b.hidden[j],b.hidden[i]];}
 record(b,(state,legal)=>{if(!legal.length)throw Error('nonterminal without legal moves');const captures=legal.filter(m=>!m.flip&&state.grid[m.to]!==Piece.EMPTY);const choices=captures.length&&rng()<.8?captures:legal;return choices[Math.floor(rng()*choices.length)];},512);
}
fs.writeFileSync('engine/corpus.txt',games.length+'\n'+games.join('\n')+'\n');
console.log('Generated '+games.length+' reference sequences');
