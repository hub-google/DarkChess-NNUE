import type { BrowserNNUE } from '../engine/nnue'
import { Board, Color, colorOf, Piece } from '../engine/board'
import { labels } from './Board'

const side = (color: Color) => color === Color.RED ? '紅方' : color === Color.BLACK ? '黑方' : '尚未決定'

function CapturedPieces({ game }: { game: Board }) {
  const red = game.captured.filter(piece => colorOf(piece) === Color.RED)
  const black = game.captured.filter(piece => colorOf(piece) === Color.BLACK)
  const row = (title: string, pieces: Piece[], className: string) => <div className="captured-row">
    <small>{title}</small><div>{pieces.length ? pieces.map((piece, index) => <span className={className} key={`${piece}-${index}`}>{labels[piece]}</span>) : <em>尚無</em>}</div>
  </div>
  return <section className="captured-card" aria-label="已被吃掉的棋子">
    <strong>已被吃掉的棋子</strong>
    {row('紅方損失', red, 'red-piece')}
    {row('黑方損失', black, 'black-piece')}
  </section>
}

export function Sidebar({ game, human, thinking, model, modelError, onReset }: {
  game: Board; human: Color; thinking: boolean; model: BrowserNNUE | null; modelError: boolean; onReset: () => void
}) {
  const engine = model ? `NNUE v2 · ${model.version}` : modelError ? '規則搜尋（模型載入失敗）' : '正在載入 NNUE 模型…'
  return <aside className="panel">
    <div className="eyebrow">PLAY AGAINST THE LAB</div>
    <h1>來測試正在<br /><em>進化的暗棋 AI</em></h1>
    <p className="lede">完整支援翻牌、移動、吃子、炮隔子攻擊；AI 會在你的每一步後立即應戰。</p>
    <div className="versus">
      <div><small>你的陣營</small><strong className={human === Color.RED ? 'red-text' : ''}>{side(human)}</strong></div>
      <b>VS</b>
      <div><small>對手</small><strong>Nightly AI</strong></div>
    </div>
    <div className="model-card">
      <span className="pulse" /><div><small>目前引擎</small><strong>{thinking ? `${engine} · 搜尋中` : engine}</strong></div>
    </div>
    <CapturedPieces game={game} />
    <ul className="rules"><li>第一顆翻出的顏色就是你的陣營</li><li>炮必須隔一顆棋才能吃子</li><li>小兵能吃將帥，將帥不能吃小兵</li><li>連續 60 步未吃子或三次重複局面判和</li></ul>
    <button className="new-game" onClick={onReset}>重新開局 <span>→</span></button>
    <p className="honesty">網頁會顯示實際載入的 NNUE 版本；若模型載入失敗，會明確降級為規則搜尋。</p>
  </aside>
}
