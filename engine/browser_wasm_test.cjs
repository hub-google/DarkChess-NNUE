const fs=require('node:fs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base='http://127.0.0.1:4173/DarkChess-NNUE/';
async function main(){
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage();page.setDefaultTimeout(30000);
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(base+(process.env.DC_DEFAULT_WASM==='1'?'':'?engine=wasm'));
  const corpus=fs.readFileSync('engine/search_corpus.txt','utf8').trim().split('\n');
  const count=Number(corpus.shift());
  const fixtures=corpus.map(line=>{
   const a=line.split(/\s+/).map(Number),n=a.shift(),history=a.splice(0,n*2),depth=a.shift(),k=a.shift();
   if(a.length!==k*2)throw Error('bad fixture');
   const roots={};for(let i=0;i<k;i++)roots[a[i*2]]=a[i*2+1];
   return {n,history,depth,roots};
  });
  if(fixtures.length!==count)throw Error('bad corpus count');
  const report=await page.evaluate(async ({base,fixtures})=>{
   const url=base+'engine/darkchess.mjs',create=(await import(url)).default;
   const m=await create({locateFile:name=>new URL(name,url).href});
   const model=await fetch(base+'models/champion.bin');
   if(!model.ok)throw Error('missing weights');
   m.FS.writeFile('/champion.bin',new Uint8Array(await model.arrayBuffer()));
   if(m.ccall('dc_init','number',[],[])!==1)throw Error('init');
   let roots=0,max_error=0;
   const analyze=(f,depth,nodes,ms)=>JSON.parse(m.ccall('dc_analyze','string',['string'],[[f.n,depth,nodes,ms,...f.history].join(' ')]));
   for(const f of fixtures){
    const a=analyze(f,f.depth,100000000,60000);
    if(a.error||a.depth!==f.depth||Object.keys(a.roots).length!==Object.keys(f.roots).length||!(a.move in f.roots))throw Error('WASM incomplete');
    for(const [move,value] of Object.entries(f.roots)){
     const error=Math.abs(a.roots[move]-value);
     if(!Number.isFinite(error)||error>1e-5)throw Error('WASM root mismatch '+move+' '+error);
     max_error=Math.max(max_error,error);roots++;
    }
    const limited=analyze(f,12,1,1000);
    if(limited.error||limited.depth!==0||limited.nodes>1||!(limited.move in f.roots))throw Error('WASM fallback invalid');
   }
   return {positions:fixtures.length,root_scores:roots,max_error,budget_fallback:true};
  },{base,fixtures});
  const board=page.locator('.board-grid');
  for(let turn=0;turn<2;turn++){
   await board.locator('button.hidden:not([disabled])').first().click();
   await page.waitForFunction(()=>document.querySelector('main')?.getAttribute('data-search-engine')==='wasm' && document.querySelector('.board-heading h2')?.textContent==='輪到你走棋');
   const ply=await page.locator('.board-heading span').textContent();
   if(ply!==('對局 #'+String(3+turn*2).padStart(3,'0')))throw Error('browser turns not applied '+ply);
  }
  if(errors.length)throw Error(errors.join(';'));
  report.browser_ai_turns=2;report.browser_backend='wasm';
  // A fresh navigation recreates the worker and model; no stale history survives.
  await page.locator('.new-game').click();
  await page.waitForFunction(()=>document.querySelector('.board-heading span')?.textContent==='對局 #001');
  await board.locator('button.hidden:not([disabled])').first().click();
  await page.waitForFunction(()=>document.querySelector('main')?.getAttribute('data-search-engine')==='wasm' && document.querySelector('.board-heading h2')?.textContent==='輪到你走棋');
  if((await page.locator('.board-heading span').textContent())!=='對局 #003')throw Error('restart failed');
  report.restart=true;
  await page.waitForFunction(()=>document.querySelector('[aria-label="V2 研究實測"]')?.textContent?.includes('67.58%'));
  report.research_evidence_visible=true;
  await page.route('**/engine/darkchess.mjs',route=>route.abort());
  await page.reload();
  await board.locator('button.hidden:not([disabled])').first().click();
  await page.waitForFunction(()=>document.querySelector('main')?.getAttribute('data-search-engine')==='ts' && document.querySelector('.board-heading h2')?.textContent==='輪到你走棋');
  report.fallback=true;
  if(errors.length)throw Error(errors.join(';'));
  fs.writeFileSync('engine/wasm_report.json',JSON.stringify(report,null,2)+'\n');console.log(report);
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
