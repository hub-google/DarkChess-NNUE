const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
async function main(){
 const url=process.env.SITE_URL,expected=process.env.EXPECTED_MODEL_SHA;
 if(!url||!expected)throw Error('missing deployment identity');
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage();page.setDefaultTimeout(30000);
  await page.goto(url,{waitUntil:'domcontentloaded'});
  const actual=await page.evaluate(async()=>{const r=await fetch(new URL('models/champion.json',location.href),{cache:'no-store'});if(!r.ok)throw Error('manifest');return (await r.json()).sha256;});
  if(actual!==expected)throw Error('published model mismatch');
  await page.locator('.board-grid button.hidden:not([disabled])').first().click();
  await page.waitForFunction(()=>document.querySelector('main')?.getAttribute('data-search-engine')==='wasm'&&document.querySelector('.board-heading h2')?.textContent==='輪到你走棋');
  if(await page.locator('.board-heading span').textContent()!=='對局 #003')throw Error('actual play failed');
  await page.locator('.new-game').click();
  await page.waitForFunction(()=>document.querySelector('.board-heading span')?.textContent==='對局 #001');
  await page.waitForFunction(()=>document.querySelector('[aria-label="V2 研究實測"]')?.textContent?.includes('67.58%'));
  console.log(JSON.stringify({url,model_sha256:actual,backend:'wasm',real_ai_move:true,restart:true,research_evidence_visible:true}));
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
