export enum Color {
  RED,
  BLACK,
  NONE
}

export enum Piece {
  EMPTY = 0,
  RED_KING = 1,
  RED_GUARD = 2,
  RED_MINISTER = 3,
  RED_ROOK = 4,
  RED_KNIGHT = 5,
  RED_CANNON = 6,
  RED_PAWN = 7,
  BLK_KING = 8,
  BLK_GUARD = 9,
  BLK_MINISTER = 10,
  BLK_ROOK = 11,
  BLK_KNIGHT = 12,
  BLK_CANNON = 13,
  BLK_PAWN = 14,
  HIDDEN = 15
}

export class Board {
  public grid: Piece[][];
  public hiddenPieces: Piece[];

  constructor() {
    this.grid = Array(4).fill(null).map(() => Array(8).fill(Piece.HIDDEN));
    this.hiddenPieces = this.initBag();
    this.shuffle(this.hiddenPieces);
  }

  private initBag(): Piece[] {
    const bag: Piece[] = [];
    // Red pieces
    bag.push(Piece.RED_KING);
    for(let i=0; i<2; i++) { bag.push(Piece.RED_GUARD, Piece.RED_MINISTER, Piece.RED_ROOK, Piece.RED_KNIGHT, Piece.RED_CANNON); }
    for(let i=0; i<5; i++) { bag.push(Piece.RED_PAWN); }
    // Black pieces
    bag.push(Piece.BLK_KING);
    for(let i=0; i<2; i++) { bag.push(Piece.BLK_GUARD, Piece.BLK_MINISTER, Piece.BLK_ROOK, Piece.BLK_KNIGHT, Piece.BLK_CANNON); }
    for(let i=0; i<5; i++) { bag.push(Piece.BLK_PAWN); }
    return bag;
  }

  private shuffle(array: Piece[]) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  public getPiece(r: number, c: number): Piece {
    return this.grid[r][c];
  }

  public flipPiece(r: number, c: number): Piece {
    if (this.grid[r][c] !== Piece.HIDDEN) {
      throw new Error("Piece is not hidden");
    }
    const p = this.hiddenPieces.pop();
    if (p === undefined) {
      throw new Error("No hidden pieces left");
    }
    this.grid[r][c] = p;
    return p;
  }

  public static getColor(p: Piece): Color {
    if (p >= Piece.RED_KING && p <= Piece.RED_PAWN) return Color.RED;
    if (p >= Piece.BLK_KING && p <= Piece.BLK_PAWN) return Color.BLACK;
    return Color.NONE;
  }
}
