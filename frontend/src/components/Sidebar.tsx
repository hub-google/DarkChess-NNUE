import type { Board } from '../engine/board'
import { Color } from '../engine/board'

const side = (color: Color) => color === Color.RED ? '紅方' : color === Color.BLACK ? '黑方' : '尚未決定'

export function Sidebar({ human, thinking, onReset }: { game: Board; human: Color; thinking: boolean; onReset: () => void }) {
  return <aside className="panel">
    <div className="eyebrow">PLAY AGAINST THE LAB</div>
    <h1>來測試正在<br/><em>進化的暗棋 AI</em></h1>
    <p className="lede">這不是展示用棋盤。完整支援翻牌、移動、吃子與炮隔子攻擊；AI 會在你的每一步後立即應戰。</p>
    <div className="versus">
      <div><small>你的陣營</small><strong className={human===Color.RED?'red-text':''}>{side(human)}</strong></div>
      <b>VS</b>
      <div><small>對手</small><strong>Nightly AI</strong></div>
    </div>
    <div className="model-card">
      <span className="pulse"/><div><small>目前引擎</small><strong>{thinking ? '搜尋最佳走法中' : 'Belief Search + 最新模型'}</strong></div>
    </div>
    <ul className="rules"><li>第一顆翻出的顏色就是你的陣營</li><li>炮必須隔一顆棋才能吃子</li><li>小兵能吃將帥，將帥不能吃小兵</li></ul>
    <button className="new-game" onClick={onReset}>重新開局 <span>↻</span></button>
    <p className="honesty">模型尚未通過棋力驗證時，網站會使用規則搜尋保底，不會假裝成已訓練完成的 NNUE。</p>
  </aside>
}
