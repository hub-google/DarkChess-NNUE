import { useCallback, useEffect, useRef, useState } from 'react'
import { BoardView } from './components/Board'
import { Sidebar } from './components/Sidebar'
import { Board, Color, colorOf, Piece } from './engine/board'
import { chooseSearchedMove } from './engine/search'
import { BrowserNNUE, loadChampion } from './engine/nnue'
import './App.css'

function App() {
  const [game, setGame] = useState(() => new Board())
  const [human, setHuman] = useState<Color>(Color.NONE)
  const [thinking, setThinking] = useState(false)
  const [model, setModel] = useState<BrowserNNUE | null>(null)
  const [modelError, setModelError] = useState(false)
  const gameRef = useRef(game)
  const searchWorkerRef = useRef<Worker | null>(null)
  const searchSerialRef = useRef(0)

  const newSearchWorker = () => new Worker(new URL('./engine/search.worker.ts', import.meta.url), {type: 'module'})

  useEffect(() => {
    searchWorkerRef.current = newSearchWorker()
    return () => { searchWorkerRef.current?.terminate(); searchWorkerRef.current = null }
  }, [])

  useEffect(() => { gameRef.current = game }, [game])
  useEffect(() => {
    let active = true
    loadChampion().then(value => { if (active) setModel(value) }).catch(() => { if (active) setModelError(true) })
    return () => { active = false }
  }, [])

  const refresh = useCallback(() => setGame(gameRef.current.clone()), [])
  const reset = () => {
    searchSerialRef.current++
    searchWorkerRef.current?.terminate()
    searchWorkerRef.current = newSearchWorker()
    const next = new Board(); gameRef.current = next; setGame(next)
    setHuman(Color.NONE); setThinking(false)
  }

  const clickSquare = (index: number) => {
    const current = gameRef.current
    if (thinking || current.isOver || (human !== Color.NONE && current.turn !== human)) return
    const piece = current.grid[index]
    if (piece === Piece.HIDDEN) {
      const revealedColor = colorOf(current.hidden[index])
      if (current.play({ from: index, to: index, flip: true })) {
        if (human === Color.NONE) setHuman(revealedColor)
        refresh()
      }
      return
    }
    if (current.selected === null) {
      if (colorOf(piece) === human) { current.selected = index; refresh() }
      return
    }
    if (colorOf(piece) === human) { current.selected = index; refresh(); return }
    current.play({ from: current.selected, to: index }); refresh()
  }

  useEffect(() => {
    if (human === Color.NONE || game.isOver || game.turn === human) return
    let cancelled = false
    setThinking(true)
    const worker = searchWorkerRef.current
    const id = ++searchSerialRef.current
    if (!worker) {
      const current = gameRef.current
      const move = chooseSearchedMove(current, model ? state => model.evaluate(state) : undefined)
      if (move) current.play(move)
      setThinking(false); refresh()
      return
    }
    const onReply = (event: MessageEvent<{id: number; move?: {from: number; to: number; flip?: boolean}; error?: string}>) => {
      if (cancelled || event.data.id !== id) return
      worker.removeEventListener('message', onReply)
      const current = gameRef.current
      if (current.isOver || current.turn === human) return
      let move = event.data.move
      if (!move || event.data.error) {
        // Same search core as the worker, with a bounded emergency budget.
        move = chooseSearchedMove(current, model ? state => model.evaluate(state) : undefined,
          {maxDepth: 3, nodeBudget: 2500, maxMs: 150})
      }
      if (move) current.play(move)
      setThinking(false)
      refresh()
    }
    worker.addEventListener('message', onReply)
    worker.postMessage({id, board: gameRef.current, limits: {
      maxDepth: 12, nodeBudget: 25000, maxMs: 2500,
    }})
    return () => { cancelled = true; worker.removeEventListener('message', onReply) }
  }, [game, human, model, refresh])

  return (
    <main className="app-shell">
      <header className="topbar">
        <div><span className="brand-mark">暗</span><strong>DarkChess Lab</strong></div>
        <span className="build-badge"><i /> AI 對弈公開測試</span>
      </header>
      <section className="game-layout">
        <Sidebar game={game} human={human} thinking={thinking} model={model} modelError={modelError} onReset={reset} />
        <BoardView game={game} human={human} thinking={thinking} onSquare={clickSquare} />
      </section>
      <footer>台灣暗棋 · 公開自我對弈研究計畫</footer>
    </main>
  )
}

export default App
