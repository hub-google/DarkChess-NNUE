import type { BrowserNNUE } from '../engine/nnue'
import { Board, Color, colorOf, Piece } from '../engine/board'
import { labels } from './Board'

const side = (color: Color) => color === Color.RED ? '红方' : color === Color.BLACK ? '黑方' : '尚未决定'

function CapturedPieces({ game }: { game: Board }) {
  const red = game.captured.filter(piece => colorOf(piece) === Color.RED)
  const black = game.captured.filter(piece => colorOf(piece) === Color.BLACK)
  const row = (title: string, pieces: Piece[], className: string) => <div className="captured-row">
    <small>{title}</small><div>{pieces.length ? pieces.map((piece, index) => <span className={className} key={`${piece}-${index}`}>{labels[piece]}</span>) : <em>尚无</em>}</div>
  </div>
  return <section className="captured-card" aria-label="已被吃掉的棋子">
    <strong>已被吃掉的棋子</strong>
    {row('红方损失', red, 'red-piece')}
    {row('黑方损失', black, 'black-piece')}
  </section>
}

export function Sidebar({ game, human, thinking, model, modelError, onReset }: {
  game: Board; human: Color; thinking: boolean; model: BrowserNNUE | null; modelError: boolean; onReset: () => void
}) {
  const engine = model ? `NNUE v2 · ${model.version}` : modelError ? '规则搜索（模型载入失败）' : '正在载入 NNUE 模型…'
  return <aside className="panel">
    <div className="eyebrow">PLAY AGAINST THE LAB</div>
    <h1>来测试正在<br /><em>进化的暗棋 AI</em></h1>
    <p className="lede">完整支持翻牌、移动、吃子、炮隔子攻击；AI 会在你的每一步后立即应战。</p>
    <div className="versus">
      <div><small>你的阵营</small><strong className={human === Color.RED ? 'red-text' : ''}>{side(human)}</strong></div>
      <b>VS</b>
      <div><small>对手</small><strong>Nightly AI</strong></div>
    </div>
    <div className="model-card">
      <span className="pulse" /><div><small>目前引擎</small><strong>{thinking ? `${engine} · 搜索中` : engine}</strong></div>
    </div>
    <CapturedPieces game={game} />
    <ul className="rules"><li>第一颗翻出的颜色就是你的阵营</li><li>炮必须隔一颗棋才能吃子</li><li>小兵能吃将帅，将帅不能吃小兵</li><li>连续 60 步未吃子或三次重复局面判和</li></ul>
    <button className="new-game" onClick={onReset}>重新开局 <span>↻</span></button>
    <p className="honesty">网页会显示实际载入的 NNUE 版本；若模型载入失败，会明确降级为规则搜索。</p>
  </aside>
}
