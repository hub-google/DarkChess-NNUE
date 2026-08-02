export enum Color { RED, BLACK, NONE }

export enum Piece {
  EMPTY = 0, RED_KING, RED_GUARD, RED_MINISTER, RED_ROOK, RED_KNIGHT,
  RED_CANNON, RED_PAWN, BLK_KING, BLK_GUARD, BLK_MINISTER, BLK_ROOK,
  BLK_KNIGHT, BLK_CANNON, BLK_PAWN, HIDDEN,
}

export type Move = { from: number; to: number; flip?: boolean }

const inventory = [
  Piece.RED_KING, Piece.RED_GUARD, Piece.RED_GUARD,
  Piece.RED_MINISTER, Piece.RED_MINISTER, Piece.RED_ROOK, Piece.RED_ROOK,
  Piece.RED_KNIGHT, Piece.RED_KNIGHT, Piece.RED_CANNON, Piece.RED_CANNON,
  ...Array(5).fill(Piece.RED_PAWN), Piece.BLK_KING,
  Piece.BLK_GUARD, Piece.BLK_GUARD, Piece.BLK_MINISTER, Piece.BLK_MINISTER,
  Piece.BLK_ROOK, Piece.BLK_ROOK, Piece.BLK_KNIGHT, Piece.BLK_KNIGHT,
  Piece.BLK_CANNON, Piece.BLK_CANNON, ...Array(5).fill(Piece.BLK_PAWN),
] as Piece[]

const rank = (p: Piece) => {
  const n = p > 7 ? p - 7 : p
  return [0, 7, 6, 5, 4, 3, 2, 1][n] ?? 0
}

export const colorOf = (p: Piece): Color => p >= 1 && p <= 7
  ? Color.RED : p >= 8 && p <= 14 ? Color.BLACK : Color.NONE

const typeOf = (p: Piece) => p > 7 ? p - 7 : p
const adjacent = (a: number, b: number) =>
  (Math.floor(a / 8) === Math.floor(b / 8) && Math.abs(a - b) === 1) || Math.abs(a - b) === 8

export class Board {
  public grid: Piece[]
  public hidden: Piece[]
  public turn = Color.NONE
  public winner = Color.NONE
  public selected: number | null = null
  public ply = 0

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
  static getColor(piece: Piece) { return colorOf(piece) }

  clone() {
    const b = new Board(false)
    b.grid = [...this.grid]; b.hidden = [...this.hidden]; b.turn = this.turn
    b.winner = this.winner; b.selected = this.selected; b.ply = this.ply
    return b
  }

  getPiece(r: number, c: number) { return this.grid[r * 8 + c] }

  flipPiece(r: number, c: number) {
    const index = r * 8 + c
    if (!this.play({ from: index, to: index, flip: true })) throw new Error('Piece is not hidden')
    return this.grid[index]
  }

  legalMoves(color = this.turn): Move[] {
    if (this.winner !== Color.NONE) return []
    const moves: Move[] = this.grid.flatMap((p, i) => p === Piece.HIDDEN ? [{ from: i, to: i, flip: true }] : [])
    if (color === Color.NONE) return moves
    for (let from = 0; from < 32; from++) {
      const piece = this.grid[from]
      if (colorOf(piece) !== color) continue
      for (let to = 0; to < 32; to++) {
        if (this.canMove(from, to)) moves.push({ from, to })
      }
    }
    return moves
  }

  canMove(from: number, to: number) {
    const a = this.grid[from], b = this.grid[to]
    if (a === Piece.EMPTY || a === Piece.HIDDEN || colorOf(a) !== this.turn || colorOf(b) === this.turn || b === Piece.HIDDEN) return false
    if (typeOf(a) === 6 && b !== Piece.EMPTY) {
      if (Math.floor(from / 8) !== Math.floor(to / 8) && from % 8 !== to % 8) return false
      const step = Math.floor(from / 8) === Math.floor(to / 8) ? (to > from ? 1 : -1) : (to > from ? 8 : -8)
      let screens = 0
      for (let at = from + step; at !== to; at += step) if (this.grid[at] !== Piece.EMPTY) screens++
      return screens === 1 && colorOf(b) === 1 - this.turn
    }
    if (!adjacent(from, to)) return false
    if (b === Piece.EMPTY) return true
    const at = typeOf(a), bt = typeOf(b)
    if (at === 1 && bt === 7) return false
    if (at === 7 && bt === 1) return true
    return rank(a) >= rank(b)
  }

  play(move: Move) {
    if (move.flip) {
      if (this.grid[move.from] !== Piece.HIDDEN) return false
      const piece = this.hidden[move.from]
      this.grid[move.from] = piece
      if (this.turn === Color.NONE) this.turn = 1 - colorOf(piece)
      else this.turn = 1 - this.turn
    } else {
      if (!this.canMove(move.from, move.to)) return false
      this.grid[move.to] = this.grid[move.from]; this.grid[move.from] = Piece.EMPTY
      this.turn = 1 - this.turn
    }
    this.selected = null; this.ply++
    const red = this.grid.some(p => colorOf(p) === Color.RED || p === Piece.HIDDEN)
    const black = this.grid.some(p => colorOf(p) === Color.BLACK || p === Piece.HIDDEN)
    if (!red) this.winner = Color.BLACK
    if (!black) this.winner = Color.RED
    if (this.turn !== Color.NONE && this.legalMoves(this.turn).length === 0) this.winner = 1 - this.turn
    return true
  }
}

const value = [0, 1200, 240, 120, 60, 30, 70, 15]

export function chooseAiMove(board: Board): Move {
  const moves = board.legalMoves()
  const tactical = moves.filter(m => !m.flip)
  const scored = tactical.map(move => {
    const captured = board.grid[move.to]
    let score = captured === Piece.EMPTY ? 0 : value[typeOf(captured)]
    const next = board.clone(); next.play(move)
    score += next.legalMoves().filter(reply => !reply.flip && next.grid[reply.to] !== Piece.EMPTY).length * -4
    score += Math.random() * 2
    return { move, score }
  })
  if (scored.length) return scored.sort((a, b) => b.score - a.score)[0].move
  const flips = moves.filter(m => m.flip)
  return flips[Math.floor(Math.random() * flips.length)]
}
