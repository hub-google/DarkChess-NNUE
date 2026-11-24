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
    type Reply = {id: number; move?: Move; error?: string; depth?: number; nodes?: number; elapsedMs?: number; backend?: 'ts' | 'wasm'}
    let finished = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const cleanup = () => {
      if (timer !== undefined) clearTimeout(timer)
      worker?.removeEventListener('message', onReply)
      worker?.removeEventListener('error', onError)
      worker?.removeEventListener('messageerror', onError)
    }
    const finish = (reply?: Reply) => {
      if (finished || cancelled || searchSerialRef.current !== id) return
      finished = true
      cleanup()
      const current = gameRef.current
      if (current.isOver || current.turn === human) { setThinking(false); return }
      const legal = current.legalMoves()
      const valid = (m?: Move) => m && legal.some(l => l.from === m.from && l.to === m.to && Boolean(l.flip) === Boolean(m.flip))
      let move = reply?.move
      if (!valid(move) || reply?.error) {
        // Restart a broken or stuck worker; never leave the board disabled.
        if (worker && searchWorkerRef.current === worker) {
          worker.terminate()
          searchWorkerRef.current = newSearchWorker()
        }
        try {
          const fallback = new DarkChessEngine(modelRef.current ? state => modelRef.current!.evaluate(state) : undefined)
            .analyze(current, {maxDepth: 3, nodeBudget: 2500, maxMs: 150})
          move = fallback.move
          setLastSearch({...fallback,backend:'ts'})
        } catch {
          move = legal[0]
          setLastSearch({depth:0,nodes:0,elapsedMs:0,backend:'ts'})
        }
      } else {
        setLastSearch({depth:reply?.depth ?? 0,nodes:reply?.nodes ?? 0,
          elapsedMs:reply?.elapsedMs ?? 0,backend:reply?.backend ?? 'ts'})
      }
      if (!valid(move)) {
        move = legal[0]
        setLastSearch({depth:0,nodes:0,elapsedMs:0,backend:'ts'})
      }
      if (move) applyMove(move)
      setThinking(false)
      refresh()
    }
    const onReply = (event: MessageEvent<Reply>) => {
      if (event.data.id === id) finish(event.data)
    }
    const onError = (event: Event) => { event.preventDefault(); finish() }
    if (!worker) { finish(); return }
    worker.addEventListener('message', onReply)
    worker.addEventListener('error', onError)
    worker.addEventListener('messageerror', onError)
    timer = setTimeout(() => finish(), 15000)
    try {
      worker.postMessage({id, board:gameRef.current.publicClone(),history:publicHistoryRef.current.slice(),
        backend:(new URLSearchParams(location.search).get('engine') ?? import.meta.env.VITE_DEFAULT_SEARCH_ENGINE) === 'wasm' ? 'wasm' : 'ts',
        limits:{maxDepth:12,nodeBudget:100000000,maxMs:2500}})
    } catch { finish() }
    return () => { cancelled=true; cleanup() }
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
