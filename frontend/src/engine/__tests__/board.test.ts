import { describe, it, expect, beforeEach } from 'vitest';
import { Board, Piece, Color } from '../board';

describe('DarkChess Board Engine', () => {
  let board: Board;

  beforeEach(() => {
    board = new Board();
  });

  it('should initialize with 32 hidden pieces', () => {
    expect(board.hiddenPieces.length).toBe(32);
    // 32 pieces should be on the board (hidden state)
    let count = 0;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 8; c++) {
        if (board.getPiece(r, c) === Piece.HIDDEN) {
          count++;
        }
      }
    }
    expect(count).toBe(32);
  });

  it('should allow flipping a piece', () => {
    const p = board.flipPiece(0, 0);
    expect(p).not.toBe(Piece.HIDDEN);
    expect(p).not.toBe(Piece.EMPTY);
    expect(board.getPiece(0, 0)).toBe(p);
  });

  it('should identify piece color correctly', () => {
    expect(Board.getColor(Piece.RED_KING)).toBe(Color.RED);
    expect(Board.getColor(Piece.BLK_KING)).toBe(Color.BLACK);
  });
});
