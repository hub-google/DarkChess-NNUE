import { useState } from 'react'
import { Board } from './components/Board'
import { Sidebar } from './components/Sidebar'
import { Board as EngineBoard } from './engine/board'

function App() {
  const [engine] = useState(() => new EngineBoard())
  // Force a re-render when the board changes
  const [, setTick] = useState(0)

  return (
    <div className="min-h-screen bg-slate-900 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-800 via-slate-900 to-black text-slate-100 flex items-center justify-center p-8 font-sans">
      <div className="flex gap-12 items-center max-w-6xl w-full mx-auto relative">
        {/* Decorative background glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-emerald-500/10 rounded-full blur-[100px] pointer-events-none"></div>
        <div className="absolute top-1/3 right-1/4 w-64 h-64 bg-cyan-500/10 rounded-full blur-[80px] pointer-events-none"></div>

        <Sidebar />
        
        <div className="flex-1 flex justify-center z-10">
          <Board engine={engine} onMove={() => setTick(t => t + 1)} />
        </div>
      </div>
    </div>
  )
}

export default App
