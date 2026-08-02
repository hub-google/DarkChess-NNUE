import { Board, Color, colorOf, Piece } from '../engine/board'

export const labels: Record<number, string> = {
  1: '帅', 2: '仕', 3: '相', 4: '俥', 5: '傌', 6: '炮', 7: '兵',
  8: '将', 9: '士', 10: '象', 11: '车', 12: '马', 13: '包', 14: '卒',
}

export function BoardView({ game, human, thinking, onSquare }: {
  game: Board; human: Color; thinking: boolean; onSquare: (index: number) => void
}) {
  const status = game.draw ? '和棋'
    : game.winner !== Color.NONE ? (game.winner === human ? '你赢了' : 'AI 获胜')
    : human === Color.NONE ? '翻开第一颗棋，决定你的阵营'
    : thinking ? 'AI 正在思考…' : game.turn === human ? '轮到你走棋' : '等待 AI'
  return (
    <section className="board-card" aria-label="暗棋棋盘">
      <div className="board-heading"><div><span>对局 #{String(game.ply + 1).padStart(3, '0')}</span><h2>{status}</h2></div><div className="turn-dot" /></div>
      <div className="board-grid">
        {game.grid.map((piece, index) => {
          const hidden = piece === Piece.HIDDEN, empty = piece === Piece.EMPTY
          const selected = game.selected === index
          const target = game.selected !== null && game.canMove(game.selected, index)
          return <button key={index} aria-label={hidden ? `翻开第 ${index + 1} 格` : labels[piece] || '空格'}
            className={`square ${hidden ? 'hidden' : ''} ${empty ? 'empty' : ''} ${colorOf(piece) === Color.RED ? 'red' : ''} ${colorOf(piece) === Color.BLACK ? 'black' : ''} ${selected ? 'selected' : ''} ${target ? 'target' : ''}`}
            onClick={() => onSquare(index)} disabled={thinking || game.isOver}><span>{hidden ? '暗' : labels[piece]}</span></button>
        })}
      </div>
      <div className="board-note"><span>● 你的棋</span><span>{thinking ? '模型正在评估合法走法' : '点选棋子，再点选目标位置'}</span></div>
    </section>
  )
}
