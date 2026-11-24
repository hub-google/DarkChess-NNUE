const fs=require('node:fs');
const {Board}=require('../.shared-build/frontend/src/engine/board.js');
let seed=0x218a23f1;
function random(){seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}
const positions=[],rows=[],n=2048,rounds=100;
for(let game=0;positions.length<n&&game<2000;game++){
 const b=new Board(false);
 for(let i=31;i>0;i--){const j=Math.floor(random()*(i+1));[b.hidden[i],b.hidden[j]]=[b.hidden[j],b.hidden[i]];}
 for(let ply=0;ply<512&&positions.length<n;ply++){
  const legal=b.legalMoves();
  if(!b.pendingChase){
   const p=b.clone();p.hidden=[]; // benchmark input does not need the referee's bag
   positions.push(p);
   rows.push([p.turn,Number(p.isOver),...p.grid,legal.length,...legal.map(m=>m.from*32+m.to)].join(' '));
  }
  if(b.isOver)break;
  if(!legal.length||!b.play(legal[Math.floor(random()*legal.length)]))throw Error('Invalid reference game');
 }
}
if(positions.length!==n)throw Error('Insufficient corpus');
fs.writeFileSync('research_v2/corpus.txt',[n+' '+rounds,...rows].join('\n')+'\n');
let checksum=0;
for(let warm=0;warm<5;warm++)for(const p of positions)p.legalMoves();
const start=performance.now();
for(let r=0;r<rounds;r++)for(const p of positions)for(const m of p.legalMoves())checksum+=m.from*32+m.to+1;
const seconds=(performance.now()-start)/1000;
const report={positions:n,rounds,seconds,positions_per_second:n*rounds/seconds,checksum};
fs.writeFileSync('research_v2/typescript.json',JSON.stringify(report,null,2)+'\n');console.log(report);
