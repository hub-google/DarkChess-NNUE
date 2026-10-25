import { Bitboard, setBit, clearBit, EMPTY_BOARD } from './Bitboard';
import { Color, Piece, PieceType, Square, getPieceColor, getPieceType } from './types';

export class Board {
  public pieces: Piece[]; // Size 32, contains actual pieces (omniscient)
  public occupied: Bitboard; // All pieces currently on the board
  public revealed: Bitboard; // Pieces that have been flipped
  public colorBB: Bitboard[]; // Revealed pieces by color (size 2)
  public typeBB: Bitboard[]; // Revealed pieces by type (size 8)
  public turn: Color;

  constructor() {
    this.pieces = new Array(32).fill(0);
    this.occupied = EMPTY_BOARD;
    this.revealed = EMPTY_BOARD;
    this.colorBB = [EMPTY_BOARD, EMPTY_BOARD];
    this.typeBB = new Array(8).fill(EMPTY_BOARD);
    this.turn = Color.None;
  }

  public clone(): Board {
    const b = new Board();
    b.pieces = [...this.pieces];
    b.occupied = this.occupied;
    b.revealed = this.revealed;
    b.colorBB = [...this.colorBB];
    b.typeBB = [...this.typeBB];
    b.turn = this.turn;
    return b;
  }

  public setPiece(square: Square, piece: Piece, isRevealed: boolean) {
    this.assertSquare(square);
    if (piece === PieceType.Empty) {
      throw new Error('Cannot place an empty piece');
    }
    if ((this.occupied & (1 << square)) !== 0) {
      this.removePiece(square);
    }
    this.pieces[square] = piece;
    this.occupied = setBit(this.occupied, square);
    
    if (isRevealed) {
      this.revealSquare(square);
    }
  }

  public revealSquare(square: Square) {
    this.assertSquare(square);
    if ((this.occupied & (1 << square)) === 0) {
      throw new Error(`Cannot reveal empty square ${square}`);
    }
    if ((this.revealed & (1 << square)) !== 0) return;

    const piece = this.pieces[square];
    const color = getPieceColor(piece);
    const type = getPieceType(piece);

    this.revealed = setBit(this.revealed, square);
    this.colorBB[color] = setBit(this.colorBB[color], square);
    this.typeBB[type] = setBit(this.typeBB[type], square);
  }

  public removePiece(square: Square) {
    this.assertSquare(square);
    if ((this.occupied & (1 << square)) === 0) return;

    const piece = this.pieces[square];
    const color = getPieceColor(piece);
    const type = getPieceType(piece);

    this.occupied = clearBit(this.occupied, square);
    if ((this.revealed & (1 << square)) !== 0) {
      this.revealed = clearBit(this.revealed, square);
      this.colorBB[color] = clearBit(this.colorBB[color], square);
      this.typeBB[type] = clearBit(this.typeBB[type], square);
    }
    this.pieces[square] = 0;
  }

  private assertSquare(square: Square) {
    if (!Number.isInteger(square) || square < 0 || square >= 32) {
      throw new RangeError(`Square must be an integer from 0 to 31: ${square}`);
    }
  }
}
