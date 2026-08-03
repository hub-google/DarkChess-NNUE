import { useEffect, useState } from 'react'

type WorkflowRun = {
  html_url: string
  status: 'queued' | 'in_progress' | 'completed'
  conclusion: string | null
  run_started_at: string
  updated_at: string
}

type TrainingSummary = {
  replayGames: number
  stagingGames: number
  totalGames: number
  updatedAt: string
}

type ModelCommit = {
  html_url: string
  commit: { committer: { date: string } }
}

const REPO_API = 'https://api.github.com/repos/hub-google/DarkChess-NNUE'
const REPO_URL = 'https://github.com/hub-google/DarkChess-NNUE'

const number = new Intl.NumberFormat('zh-TW')

function relativeTime(value?: string) {
  if (!value) return '更新中'
  const seconds = Math.round((new Date(value).getTime() - Date.now()) / 1000)
  const formatter = new Intl.RelativeTimeFormat('zh-TW', { numeric: 'auto' })
  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['year', 31_536_000], ['month', 2_592_000], ['day', 86_400],
    ['hour', 3_600], ['minute', 60],
  ]
  for (const [unit, size] of units) {
    if (Math.abs(seconds) >= size) return formatter.format(Math.round(seconds / size), unit)
  }
  return '剛剛'
}

function runLabel(run?: WorkflowRun) {
  if (!run) return '讀取中'
  if (run.status === 'in_progress') return '執行中'
  if (run.status === 'queued') return '排隊中'
  return ({ success: '已完成', failure: '失敗', cancelled: '已取消', skipped: '已略過' } as Record<string, string>)[run.conclusion ?? ''] ?? '已結束'
}

function runTone(run?: WorkflowRun) {
  if (!run) return 'muted'
  if (run.status !== 'completed') return 'live'
  return run.conclusion === 'success' ? 'success' : 'warning'
}

async function json<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: { Accept: 'application/vnd.github+json' } })
  if (!response.ok) throw new Error(`Request failed: ${response.status}`)
  return response.json() as Promise<T>
}

export function TrainingStatus() {
  const [summary, setSummary] = useState<TrainingSummary | null>(null)
  const [model, setModel] = useState<ModelCommit | null>(null)
  const [selfPlay, setSelfPlay] = useState<WorkflowRun | undefined>()
  const [train, setTrain] = useState<WorkflowRun | undefined>()

  useEffect(() => {
    let active = true
    const load = async () => {
      const results = await Promise.allSettled([
        json<TrainingSummary>(`${import.meta.env.BASE_URL}training-status.json?ts=${Date.now()}`),
        json<ModelCommit[]>(`${REPO_API}/commits?path=models/champion.nnue&per_page=1`),
        json<{ workflow_runs: WorkflowRun[] }>(`${REPO_API}/actions/workflows/self_play.yml/runs?per_page=1`),
        json<{ workflow_runs: WorkflowRun[] }>(`${REPO_API}/actions/workflows/train.yml/runs?per_page=1`),
      ])
      if (!active) return
      if (results[0].status === 'fulfilled') setSummary(results[0].value)
      if (results[1].status === 'fulfilled') setModel(results[1].value[0] ?? null)
      if (results[2].status === 'fulfilled') setSelfPlay(results[2].value.workflow_runs[0])
      if (results[3].status === 'fulfilled') setTrain(results[3].value.workflow_runs[0])
    }
    void load()
    const timer = window.setInterval(load, 120_000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  const modelDate = model?.commit.committer.date
  return <section className="training-card" aria-label="AI 訓練即時動態">
    <div className="training-heading">
      <div><span className="live-dot" /><strong>LAB LIVE</strong></div>
      <small>{summary ? relativeTime(summary.updatedAt) + '同步' : '安全連線中'}</small>
    </div>

    <a className="metric-row" href="https://huggingface.co/datasets/hub-google/DarkChess-NNUE-Data" target="_blank" rel="noreferrer">
      <span><small>訓練資料</small><strong>{summary ? number.format(summary.totalGames) : '—'} <em>局</em></strong></span>
      <span className="metric-detail">{summary ? `${number.format(summary.replayGames)} 已整合 · ${number.format(summary.stagingGames)} 待整合` : '等待首次統計'}</span>
    </a>

    <a className="metric-row" href={model?.html_url ?? `${REPO_URL}/commits/master/models/champion.nnue`} target="_blank" rel="noreferrer">
      <span><small>AI 模型更新</small><strong>{relativeTime(modelDate)}</strong></span>
      <span className="row-arrow" aria-hidden="true">↗</span>
    </a>

    <div className="workflow-grid">
      <a href={selfPlay?.html_url ?? `${REPO_URL}/actions/workflows/self_play.yml`} target="_blank" rel="noreferrer">
        <small>SELF-PLAY</small><strong><i className={runTone(selfPlay)} />{runLabel(selfPlay)}</strong><span>{relativeTime(selfPlay?.updated_at)}</span>
      </a>
      <a href={train?.html_url ?? `${REPO_URL}/actions/workflows/train.yml`} target="_blank" rel="noreferrer">
        <small>TRAIN</small><strong><i className={runTone(train)} />{runLabel(train)}</strong><span>{relativeTime(train?.updated_at)}</span>
      </a>
    </div>
  </section>
}
