import { useCallback, useEffect, useState } from 'react'
import { BoardView } from './components/Board'
import { Sidebar } from './components/Sidebar'
import { Board, chooseAiMove, Color, colorOf, Piece } from './engine/board'
import './App.css'

function App() {
  const [game, setGame] = useState(() => new Board())
  const [human, setHuman] = useState<Color>(Color.NONE)
  const [thinking, setThinking] = useState(false)

  const refresh = useCallback(() => setGame(g => g.clone()), [])
  const reset = () => { setGame(new Board()); setHuman(Color.NONE); setThinking(false) }

  const clickSquare = (index: number) => {
    if (thinking || game.winner !== Color.NONE || (human !== Color.NONE && game.turn !== human)) return
    const piece = game.grid[index]
    if (piece === Piece.HIDDEN) {
      const revealedColor = colorOf(game.hidden[index])
      if (game.play({ from: index, to: index, flip: true })) {
        if (human === Color.NONE) setHuman(revealedColor)
        refresh()
      }
      return
    }
    if (game.selected === null) {
      if (colorOf(piece) === human) { game.selected = index; refresh() }
      return
    }
    if (colorOf(piece) === human) { game.selected = index; refresh(); return }
    game.play({ from: game.selected, to: index }); refresh()
  }

  useEffect(() => {
    if (human === Color.NONE || game.winner !== Color.NONE || game.turn === human || thinking) return
    setThinking(true)
    const timer = window.setTimeout(() => {
      const move = chooseAiMove(game)
      if (move) game.play(move)
      setThinking(false); refresh()
    }, 420)
    return () => window.clearTimeout(timer)
  }, [game, human, thinking, refresh])

  return (
    <main className="app-shell">
      <header className="topbar">
        <div><span className="brand-mark">暗</span><strong>DarkChess Lab</strong></div>
        <span className="build-badge"><i /> AI 對弈測試版</span>
      </header>
      <section className="game-layout">
        <Sidebar game={game} human={human} thinking={thinking} onReset={reset} />
        <BoardView game={game} human={human} thinking={thinking} onSquare={clickSquare} />
      </section>
      <footer>台灣暗棋・公開自我對弈研究計畫</footer>
    </main>
  )
}

export default App
