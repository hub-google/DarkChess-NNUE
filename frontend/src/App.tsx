import { useCallback, useEffect, useRef, useState } from 'react'
import { BoardView } from './components/Board'
import { Sidebar } from './components/Sidebar'
import { Board, Color, colorOf, Piece, type Move } from './engine/board'
import { DarkChessEngine } from './engine/search'
import { BrowserNNUE, loadChampion } from './engine/nnue'
import { applyPublicMove, type PublicMove } from './engine/public-history'
import './App.css'

function App() {
  const [game, setGame] = useState(() => new Board())
  const [human, setHuman] = useState<Color>(Color.NONE)
  const [thinking, setThinking] = useState(false)
  const [model, setModel] = useState<BrowserNNUE | null>(null)
  const [modelError, setModelError] = useState(false)
  const [lastSearch, setLastSearch] = useState<{depth: number; nodes: number; elapsedMs: number; backend?: 'ts' | 'wasm'} | null>(null)
  const gameRef = useRef(game)
  const modelRef = useRef<BrowserNNUE | null>(null)
  const publicHistoryRef = useRef<PublicMove[]>([])
  const applyMove = useCallback((move: Move) => applyPublicMove(gameRef.current, move, publicHistoryRef.current), [])
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
    loadChampion().then(value => { if (active) { modelRef.current = value; setModel(value) } }).catch(() => { if (active) setModelError(true) })
    return () => { active = false }
  }, [])

  const refresh = useCallback(() => setGame(gameRef.current.clone()), [])
  const reset = () => {
    searchSerialRef.current++
    searchWorkerRef.current?.terminate()
    searchWorkerRef.current = newSearchWorker()
    publicHistoryRef.current = []
    const next = new Board(); gameRef.current = next; setGame(next)
    setHuman(Color.NONE); setThinking(false); setLastSearch(null)
  }

  const clickSquare = (index: number) => {
    const current = gameRef.current
    if (thinking || current.isOver || (human !== Color.NONE && current.turn !== human)) return
    const piece = current.grid[index]
    if (piece === Piece.HIDDEN) {
      const revealedColor = colorOf(current.hidden[index])
      if (applyMove({ from: index, to: index, flip: true })) {
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
    applyMove({ from: current.selected, to: index }); refresh()
  }

  useEffect(() => {
    if (human === Color.NONE || game.isOver || game.turn === human) return
    let cancelled = false
    setThinking(true)
    const worker = searchWorkerRef.current
    const id = ++searchSerialRef.current
    if (!worker) {
      const current = gameRef.current
      const analysis = new DarkChessEngine(modelRef.current ? state => modelRef.current!.evaluate(state) : undefined)
        .analyze(current, {maxDepth: 3, nodeBudget: 2500, maxMs: 150})
      if (analysis.move) applyMove(analysis.move)
      setLastSearch({...analysis,backend:'ts'})
      setThinking(false); refresh()
      return
    }
    const onReply = (event: MessageEvent<{id: number; move?: {from: number; to: number; flip?: boolean}; error?: string; depth?: number; nodes?: number; elapsedMs?: number; backend?: 'ts' | 'wasm'}>) => {
      if (cancelled || event.data.id !== id) return
      worker.removeEventListener('message', onReply)
      const current = gameRef.current
      if (current.isOver || current.turn === human) return
      let move = event.data.move
      if (!move || event.data.error) {
        // Same search core as the worker, with a bounded emergency budget.
        const fallback = new DarkChessEngine(modelRef.current ? state => modelRef.current!.evaluate(state) : undefined)
          .analyze(current, {maxDepth: 3, nodeBudget: 2500, maxMs: 150})
        move = fallback.move
        setLastSearch({...fallback,backend:'ts'})
      } else {
        setLastSearch({depth: event.data.depth ?? 0, nodes: event.data.nodes ?? 0,
          elapsedMs: event.data.elapsedMs ?? 0, backend: event.data.backend ?? 'ts'})
      }
      if (move) applyMove(move)
      setThinking(false)
      refresh()
    }
    worker.addEventListener('message', onReply)
    worker.postMessage({id, board: gameRef.current.publicClone(), history: publicHistoryRef.current.slice(),
      backend: new URLSearchParams(location.search).get('engine') === 'wasm' ? 'wasm' : 'ts', limits: {
      maxDepth: 12, nodeBudget: 25000, maxMs: 2500,
    }})
    return () => { cancelled = true; worker.removeEventListener('message', onReply) }
  }, [game, human, refresh, applyMove])

  return (
    <main className="app-shell" data-search-engine={lastSearch?.backend ?? "none"}>
      <header className="topbar">
        <div><span className="brand-mark">暗</span><strong>DarkChess Lab</strong></div>
        <span className="build-badge"><i /> AI 對弈公開測試</span>
      </header>
      <section className="game-layout">
        <Sidebar game={game} human={human} thinking={thinking} model={model} modelError={modelError} lastSearch={lastSearch} onReset={reset} />
        <BoardView game={game} human={human} thinking={thinking} onSquare={clickSquare} />
      </section>
      <footer>台灣暗棋 · 公開自我對弈研究計畫</footer>
    </main>
  )
}

export default App
