/** Paired SPRT using the SAME DarkChessEngine as web and self-play.
 * Only the referee owns the hidden shuffled bag. Search sees public counts.
 * No incomplete games are ever silently counted as draws.
 */
import {writeFileSync, appendFileSync, readFileSync} from 'node:fs'
import {Board, Color, Piece} from '../../frontend/src/engine/board'
import {DarkChessEngine} from '../../frontend/src/engine/search'
import {loadNNUE} from './model_runtime'

type Config = {pairs: number; depth: number; nodes: number; ms: number; elo0: number; elo1: number;
  alpha: number; beta: number; seed: number; maxRetries: number; output: string; resume: string; champion: string; challenger: string}
const arg = (name: string, fallback: string): string => {
  const i = process.argv.indexOf('--' + name)
  return i >= 0 ? process.argv[i + 1] : fallback
}
const number = (name: string, fallback: number): number => {
  const val = Number(arg(name, String(fallback)))
  if (!Number.isFinite(val)) throw new Error('Invalid number: ' + name)
  return val
}
const config: Config = {
  pairs: number('pairs', 200), depth: number('depth', 6),
  nodes: number('node-budget', 1200), ms: 1000 * number('max-seconds', 0.1),
  elo0: number('elo0', -20), elo1: number('elo1', 20),
  alpha: number('alpha', 0.05), beta: number('beta', 0.05),
  seed: number('seed', 20260731), maxRetries: number('max-retries', 8),
  output: arg('output', 'sprt_results.json'), resume: arg('resume', ''),
  champion: arg('champion', 'validation_models/champion'),
  challenger: arg('challenger', 'validation_models/challenger')
}
if (config.pairs < 1 || config.depth < 1 || config.nodes < 1 || config.ms <= 0 ||
    config.elo1 <= config.elo0 || config.alpha <= 0 || config.alpha >= 1 ||
    config.beta <= 0 || config.beta >= 1) throw new Error('Invalid SPRT configuration')
let seed = config.seed >>> 0
function random() {
  seed += 0x6D2B79F5
  let x = seed
  x = Math.imul(x ^ (x >>> 15), x | 1)
  x ^= x + Math.imul(x ^ (x >>> 7), x | 61)
  return ((x ^ (x >>> 14)) >>> 0) / 4294967296
}
const inventory: Piece[] = [1,2,2,3,3,4,4,5,5,6,6,7,7,7,7,7,
  8,9,9,10,10,11,11,12,12,13,13,14,14,14,14,14] as Piece[]
function deal(): Piece[] {
  const bag = [...inventory]
  for (let i = 31; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[bag[i], bag[j]] = [bag[j], bag[i]]
  }
  return bag
}
function play(first: DarkChessEngine, second: DarkChessEngine, bag: Piece[], opening: number): number {
  const board = new Board(false)
  board.hidden = [...bag]
  const firstColor = bag[opening] <= Piece.RED_PAWN ? Color.RED : Color.BLACK
  if (!board.play({from: opening, to: opening, flip: true})) throw new Error('Invalid first reveal')
  for (let ply = 1; ply < 512 && !board.isOver; ply++) {
    const engine = board.turn === firstColor ? first : second
    const analysis = engine.analyze(board, {maxDepth: config.depth, nodeBudget: config.nodes, maxMs: config.ms})
    if (!analysis.move) throw new Error('Non-terminal has no legal moves')
    if (!board.play(analysis.move)) throw new Error('Illegal move by shared engine')
  }
  if (!board.isOver) throw new Error('Incomplete game after 512 plies')
  if (board.draw) return 0.5
  return board.winner === firstColor ? 1 : 0
}
function probability(elo: number) {return 1 / (1 + Math.pow(10, -elo / 400))}
function llrForPair(score: number): number {
  const p0 = probability(config.elo0), p1 = probability(config.elo1)
  return score * Math.log(p1 / p0) + (2 - score) * Math.log((1 - p1) / (1 - p0))
}
function publish(passed: boolean) {
  const value = passed ? 'true' : 'false'
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, 'passed=' + value + '\n')
  if (process.env.GITHUB_ENV) appendFileSync(process.env.GITHUB_ENV, 'PASSED=' + value + '\n')
}
function main() {
  const championModel = loadNNUE(config.champion)
  const challengerModel = loadNNUE(config.challenger)
  const champion = new DarkChessEngine(board => championModel.model.evaluate(board))
  const challenger = new DarkChessEngine(board => challengerModel.model.evaluate(board))
  const upper = Math.log((1 - config.beta) / config.alpha)
  const lower = Math.log(config.beta / (1 - config.alpha))
  let score = 0, llr = 0, played = 0, incomplete = 0
  let decision = 'UNDECIDED'
  let pair_scores: number[] = []
  if (config.resume) {
    const saved = JSON.parse(readFileSync(config.resume, 'utf8')) as Record<string, unknown>
    const same = (key: string, expected: unknown) => saved[key] === expected
    if (!same('decision', 'UNDECIDED') ||
        !same('champion_model', championModel.version) ||
        !same('challenger_model', challengerModel.version) ||
        !same('elo0', config.elo0) || !same('elo1', config.elo1) ||
        !same('alpha', config.alpha) || !same('beta', config.beta) ||
        !same('depth', config.depth) || !same('node_budget', config.nodes) ||
        !same('max_seconds', config.ms / 1000) || !same('seed', config.seed) ||
        !same('max_retries', config.maxRetries) ||
        !Number.isInteger(Number(saved.rng_state)) ||
        !Number.isInteger(Number(saved.pairs_played)) ||
        !Array.isArray(saved.pair_scores) ||
        saved.pair_scores.length !== saved.pairs_played ||
        Number(saved.pairs_played) < 1 || Number(saved.pairs_played) >= config.pairs) {
      throw new Error('SPRT resume mismatch: model, test configuration or saved progress changed')
    }
    pair_scores = saved.pair_scores as number[]
    if (!pair_scores.every(x => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= 2)) {
      throw new Error('SPRT resume contains invalid paired scores')
    }
    seed = Number(saved.rng_state) >>> 0
    played = pair_scores.length
    score = pair_scores.reduce((a, b) => a + b, 0)
    llr = pair_scores.reduce((a, b) => a + llrForPair(b), 0)
    incomplete = Number(saved.incomplete_deals_discarded)
    if (!Number.isSafeInteger(incomplete) || incomplete < 0 ||
        Math.abs(llr - Number(saved.log_likelihood_ratio)) > 1e-8) {
      throw new Error('SPRT resume statistical checkpoint is invalid')
    }
    console.log('[SPRT] Resuming existing paired experiment: ' + played + ' pairs; previous scores preserved')
  }
  for (let i = played; i < config.pairs; i++) {
    let completed = false
    for (let retry = 0; retry < config.maxRetries; retry++) {
      const bag = deal()
      const opening = Math.floor(random() * 32)
      try {
        const first = play(challenger, champion, bag, opening)
        const second = play(champion, challenger, bag, opening)
        const points = first + 1 - second
        pair_scores.push(points)
        score += points
        llr += llrForPair(points)
        played++
        completed = true
        break
      } catch (error) {
        if (!String(error).includes('Incomplete game')) throw error
        incomplete++
        console.warn('[SPRT] Quarantined incomplete paired deal ' + (i + 1))
      }
    }
    if (!completed) throw new Error('SPRT cannot finish matched pairs; refuse promotion')
    if (played % 10 === 0) console.log('[SPRT] pairs=' + played + ' rate=' + (score / (2 * played)).toFixed(4) + ' LLR=' + llr.toFixed(3))
    if (llr >= upper) {decision = 'ACCEPT_H1';break}
    if (llr <= lower) {decision = 'ACCEPT_H0';break}
  }
  const passed = decision === 'ACCEPT_H1'
  const report = {decision, passed, pairs_played: played, games_played: 2 * played,
    score_rate: score / (2 * played), log_likelihood_ratio: llr,
    elo0: config.elo0, elo1: config.elo1,
    alpha: config.alpha, beta: config.beta, depth: config.depth,
    node_budget: config.nodes, max_seconds: config.ms / 1000, seed: config.seed,
    max_retries: config.maxRetries, rng_state: seed >>> 0,
    pair_scores, incomplete_deals_discarded: incomplete,
    champion_model: championModel.version, challenger_model: challengerModel.version,
    note: 'One DarkChessEngine across web, self-play and SPRT. Undecided never promotes.'}
  writeFileSync(config.output, JSON.stringify(report, null, 2))
  publish(passed)
  console.log('[SPRT] decision=' + decision + ' pairs=' + played + ' challenger score=' + (100 * report.score_rate).toFixed(2) + '% LLR=' + llr.toFixed(3) + ' PASSED=' + passed)
}
main()
