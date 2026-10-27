export enum Color { RED, BLACK, NONE }

export enum Piece {
  EMPTY = 0, RED_KING, RED_GUARD, RED_MINISTER, RED_ROOK, RED_KNIGHT,
  RED_CANNON, RED_PAWN, BLK_KING, BLK_GUARD, BLK_MINISTER, BLK_ROOK,
  BLK_KNIGHT, BLK_CANNON, BLK_PAWN, HIDDEN,
}

export type Move = { from: number; to: number; flip?: boolean }
type ChaseRecord = [attacker: number, target: number, count: number, chaser: Color, route: number[]]

const inventory = [
  Piece.RED_KING, Piece.RED_GUARD, Piece.RED_GUARD,
  Piece.RED_MINISTER, Piece.RED_MINISTER, Piece.RED_ROOK, Piece.RED_ROOK,
  Piece.RED_KNIGHT, Piece.RED_KNIGHT, Piece.RED_CANNON, Piece.RED_CANNON,
  ...Array(5).fill(Piece.RED_PAWN), Piece.BLK_KING,
  Piece.BLK_GUARD, Piece.BLK_GUARD, Piece.BLK_MINISTER, Piece.BLK_MINISTER,
  Piece.BLK_ROOK, Piece.BLK_ROOK, Piece.BLK_KNIGHT, Piece.BLK_KNIGHT,
  Piece.BLK_CANNON, Piece.BLK_CANNON, ...Array(5).fill(Piece.BLK_PAWN),
] as Piece[]

export const INITIAL_COUNTS = [1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5]

export const colorOf = (p: Piece): Color => p >= 1 && p <= 7
  ? Color.RED : p >= 8 && p <= 14 ? Color.BLACK : Color.NONE
export const typeOf = (p: Piece) => p > 7 ? p - 8 : p - 1
const adjacent = (a: number, b: number) =>
  (Math.floor(a / 8) === Math.floor(b / 8) && Math.abs(a - b) === 1) || Math.abs(a - b) === 8

// Precomputed destination geometry avoids constructing/sorting candidate lists
// at millions of speculative nodes during exact-depth CPU self-play.
const NEIGHBOR_TARGETS: number[][] = []
const CANNON_TARGETS: number[][] = []
for (let from = 0; from < 32; from++) {
  const row = Math.floor(from / 8), col = from % 8
  const near: number[] = []
  if (row > 0) near.push(from - 8)
  if (row < 3) near.push(from + 8)
  if (col > 0) near.push(from - 1)
  if (col < 7) near.push(from + 1)
  NEIGHBOR_TARGETS.push(near.sort((a, b) => a - b))
  const all = new Set(near)
  for (let c = 0; c < 8; c++) if (c !== col) all.add(row * 8 + c)
  for (let r = 0; r < 4; r++) if (r !== row) all.add(r * 8 + col)
  CANNON_TARGETS.push([...all].sort((a, b) => a - b))
}

export class Board {
  public grid: Piece[]
  public hidden: Piece[]
  public remainingCounts = [...INITIAL_COUNTS]
  public captured: Piece[] = []
  public turn = Color.NONE
  public winner = Color.NONE
  public draw = false
  public selected: number | null = null
  public ply = 0
  public halfMoveClock = 0
  public history: string[] = []
  public tokenAtSquare = Array.from({ length: 32 }, (_, square) => square)
  public chaseThreats: ChaseRecord[] = []
  public pendingChase: ChaseRecord | null = null

  constructor(shuffle = true) {
    this.hidden = [...inventory]
    if (shuffle) {
      for (let i = 31; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[this.hidden[i], this.hidden[j]] = [this.hidden[j], this.hidden[i]]
      }
    }
    this.grid = Array(32).fill(Piece.HIDDEN)
  }

  get hiddenPieces() { return this.hidden }
  get isOver() { return this.winner !== Color.NONE || this.draw }
  static getColor(piece: Piece) { return colorOf(piece) }

  clone() {
    // Avoid rebuilding the initial inventory for every speculative node.
    const b = Object.create(Board.prototype) as Board
    b.grid = [...this.grid]; b.hidden = [...this.hidden]
    b.remainingCounts = [...this.remainingCounts]; b.captured = [...this.captured]
    b.turn = this.turn; b.winner = this.winner; b.draw = this.draw
    b.selected = this.selected; b.ply = this.ply; b.halfMoveClock = this.halfMoveClock
    b.history = [...this.history]
    b.tokenAtSquare = [...this.tokenAtSquare]
    b.chaseThreats = this.chaseThreats.map(record => [record[0], record[1], record[2], record[3], [...record[4]]])
    b.pendingChase = this.pendingChase ? [this.pendingChase[0], this.pendingChase[1], this.pendingChase[2], this.pendingChase[3], [...this.pendingChase[4]]] : null
    return b
  }

  getPiece(r: number, c: number) { return this.grid[r * 8 + c] }

  flipPiece(r: number, c: number) {
    const index = r * 8 + c
    if (!this.play({ from: index, to: index, flip: true })) throw new Error('Piece is not hidden')
    return this.grid[index]
  }

  private snapshot() {
    return `${this.grid.join(',')}|${this.remainingCounts.join(',')}|${this.turn}`
  }

  repetitionCount() {
    // A flip permanently reduces the hidden inventory; a capture permanently
    // removes a board piece. Either resets halfMoveClock to zero, so the
    // resulting position cannot equal any earlier legal position.
    if (this.halfMoveClock === 0) return 1
    const current = this.snapshot()
    let count = 1
    // Positions before the latest irreversible capture/flip cannot recur.
    // halfMoveClock is the exact number of quiet moves since that event.
    const earliest = Math.max(0, this.history.length - this.halfMoveClock)
    for (let i = earliest; i < this.history.length; i++) {
      if (this.history[i] === current) count++
    }
    return count
  }

  legalMoves(color = this.turn): Move[] {
    if (this.isOver) return []
    const moves: Move[] = []
    for (let square = 0; square < 32; square++) {
      if (this.grid[square] === Piece.HIDDEN) moves.push({from: square, to: square, flip: true})
    }
    if (color === Color.NONE) return moves
    for (let from = 0; from < 32; from++) {
      if (colorOf(this.grid[from]) !== color) continue
      const candidates = typeOf(this.grid[from]) === 5 ? CANNON_TARGETS[from] : NEIGHBOR_TARGETS[from]
      for (const to of candidates) if (this.canMove(from, to)) moves.push({from, to})
    }
    return moves
  }

  canMove(from: number, to: number) {
    return this.canMoveByBoardRules(from, to) && !this.continuesForbiddenChase(from, to)
  }

  private canMoveByBoardRules(from: number, to: number) {
    const attacker = this.grid[from], victim = this.grid[to]
    if (attacker === Piece.EMPTY || attacker === Piece.HIDDEN || colorOf(attacker) !== this.turn || colorOf(victim) === this.turn || victim === Piece.HIDDEN) return false
    if (typeOf(attacker) === 5 && victim !== Piece.EMPTY) {
      if (Math.floor(from / 8) !== Math.floor(to / 8) && from % 8 !== to % 8) return false
      const step = Math.floor(from / 8) === Math.floor(to / 8) ? (to > from ? 1 : -1) : (to > from ? 8 : -8)
      let screens = 0
      for (let at = from + step; at !== to; at += step) if (this.grid[at] !== Piece.EMPTY) screens++
      return screens === 1 && colorOf(victim) === 1 - this.turn
    }
    if (!adjacent(from, to)) return false
    if (victim === Piece.EMPTY) return true
    const attackerType = typeOf(attacker), victimType = typeOf(victim)
    if (attackerType === 0 && victimType === 6) return false
    if (attackerType === 6 && victimType === 0) return true
    return attackerType <= victimType
  }

  private squareOfToken(token: number) { return this.tokenAtSquare.indexOf(token) }

  private canAttack(attackerSquare: number, targetSquare: number, grid = this.grid) {
    const attacker = grid[attackerSquare], victim = grid[targetSquare]
    if (colorOf(attacker) === Color.NONE || colorOf(victim) === Color.NONE || colorOf(attacker) === colorOf(victim)) return false
    if (typeOf(attacker) !== 5) {
      if (!adjacent(attackerSquare, targetSquare)) return false
      const attackerType = typeOf(attacker), victimType = typeOf(victim)
      if (attackerType === 0 && victimType === 6) return false
      if (attackerType === 6 && victimType === 0) return true
      return attackerType <= victimType
    }
    if (Math.floor(attackerSquare / 8) !== Math.floor(targetSquare / 8) && attackerSquare % 8 !== targetSquare % 8) return false
    const step = Math.floor(attackerSquare / 8) === Math.floor(targetSquare / 8)
      ? (targetSquare > attackerSquare ? 1 : -1) : (targetSquare > attackerSquare ? 8 : -8)
    let screens = 0
    for (let at = attackerSquare + step; at !== targetSquare; at += step) if (grid[at] !== Piece.EMPTY) screens++
    return screens === 1
  }

  private continuesForbiddenChase(from: number, to: number) {
    if (!this.pendingChase) return false
    const [attackerToken, targetToken] = this.pendingChase
    if (this.tokenAtSquare[from] !== attackerToken || this.grid[to] !== Piece.EMPTY) return false
    const target = this.squareOfToken(targetToken)
    if (target < 0) return false
    const next = [...this.grid]
    next[to] = next[from]; next[from] = Piece.EMPTY
    if (!this.canAttack(to, target, next)) return false
    const route = this.pendingChase[4]
    return new Set(route).size !== route.length
  }

  private updateChase(mover: Color, movedToken: number, flip: boolean, capture: boolean) {
    const oldThreats = this.chaseThreats, oldPending = this.pendingChase
    this.chaseThreats = []; this.pendingChase = null
    if (flip || capture) return
    const movedSquare = this.squareOfToken(movedToken)
    if (movedSquare < 0) return
    for (const record of oldThreats) {
      const [attackerToken, targetToken, count, chaser, route] = record
      if (movedToken !== targetToken || mover === chaser) continue
      const attackerSquare = this.squareOfToken(attackerToken)
      if (attackerSquare >= 0 && !this.canAttack(attackerSquare, movedSquare)) {
        this.pendingChase = [attackerToken, targetToken, count, chaser, [...route, movedSquare]]
        return
      }
    }
    for (let target = 0; target < 32; target++) {
      if (!this.canAttack(movedSquare, target)) continue
      const targetToken = this.tokenAtSquare[target]
      const count = oldPending && oldPending[0] === movedToken && oldPending[1] === targetToken && oldPending[3] === mover
        ? oldPending[2] + 1 : 1
      const route = oldPending && oldPending[0] === movedToken && oldPending[1] === targetToken && oldPending[3] === mover
        ? oldPending[4] : [this.squareOfToken(targetToken)]
      this.chaseThreats.push([movedToken, targetToken, count, mover, route])
    }
  }

  play(move: Move) {
    if (this.isOver) return false
    this.history.push(this.snapshot())
    const mover = this.turn
    const movedToken = this.tokenAtSquare[move.from]
    let wasCapture = false
    if (move.flip) {
      if (this.grid[move.from] !== Piece.HIDDEN) { this.history.pop(); return false }
      const piece = this.hidden[move.from]
      this.grid[move.from] = piece
      this.remainingCounts[piece - 1]--
      this.halfMoveClock = 0
      if (this.turn === Color.NONE) this.turn = colorOf(piece)
    } else {
      if (!this.canMove(move.from, move.to)) { this.history.pop(); return false }
      const captured = this.grid[move.to]
      wasCapture = captured !== Piece.EMPTY
      if (wasCapture) this.tokenAtSquare[move.to] = -1
      this.grid[move.to] = this.grid[move.from]
      this.grid[move.from] = Piece.EMPTY
      this.tokenAtSquare[move.to] = movedToken
      this.tokenAtSquare[move.from] = -1
      if (captured !== Piece.EMPTY) { this.captured.push(captured); this.halfMoveClock = 0 }
      else this.halfMoveClock++
    }
    this.updateChase(mover, movedToken, Boolean(move.flip), wasCapture)
    this.turn = 1 - this.turn
    this.selected = null
    this.ply++
    const red = this.grid.some(p => colorOf(p) === Color.RED) || this.remainingCounts.slice(0, 7).some(Boolean)
    const black = this.grid.some(p => colorOf(p) === Color.BLACK) || this.remainingCounts.slice(7).some(Boolean)
    if (!red) this.winner = Color.BLACK
    else if (!black) this.winner = Color.RED
    else if (this.halfMoveClock >= 60) this.draw = true
    else if (!this.grid.includes(Piece.HIDDEN) && this.legalMoves(this.turn).length === 0) this.winner = 1 - this.turn
    return true
  }
}

export type Evaluator = (board: Board) => number
// All AI decisions are made through frontend/src/engine/search.ts:DarkChessEngine.
// No second heuristic-only board AI is maintained here.
