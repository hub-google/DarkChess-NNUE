import { describe, it, expect } from 'vitest';
import { Board } from './Board';
import { Game } from './Game';
import { MoveGenerator, canEat } from './MoveGenerator';
import { Color, PieceType, makePiece, MoveType, getFrom, getTo, getMoveType } from './types';

describe('MoveGenerator rules', () => {
  it('canEat logic should follow rank correctly', () => {
    // Cannon can eat anything
    expect(canEat(PieceType.Cannon, PieceType.King)).toBe(true);
    expect(canEat(PieceType.Cannon, PieceType.Pawn)).toBe(true);

    // King cannot eat pawn
    expect(canEat(PieceType.King, PieceType.Pawn)).toBe(false);

    // Pawn can eat King
    expect(canEat(PieceType.Pawn, PieceType.King)).toBe(true);

    // Normal ranks
    expect(canEat(PieceType.Rook, PieceType.Knight)).toBe(true); // 4 >= 3
    expect(canEat(PieceType.Knight, PieceType.Rook)).toBe(false); // 3 >= 4 -> false
    expect(canEat(PieceType.Elephant, PieceType.Elephant)).toBe(true); // 5 >= 5
  });

  it('should generate flip moves for unrevealed pieces', () => {
    const board = new Board();
    // Place a piece but do not reveal it
    board.setPiece(0, makePiece(Color.Red, PieceType.Rook), false);
    board.turn = Color.None; // Game start

    const moves = MoveGenerator.generateMoves(board);
    expect(moves.length).toBe(1);
    expect(getFrom(moves[0])).toBe(0);
    expect(getTo(moves[0])).toBe(0);
    expect(getMoveType(moves[0])).toBe(MoveType.Flip);
  });

  it('should generate correct cannon hops', () => {
    const board = new Board();
    // 0: Red Cannon
    board.setPiece(0, makePiece(Color.Red, PieceType.Cannon), true);
    // 1: Unrevealed piece (Mount)
    board.setPiece(1, makePiece(Color.Black, PieceType.Pawn), false);
    // 2: Empty
    // 3: Black Rook (Target)
    board.setPiece(3, makePiece(Color.Black, PieceType.Rook), true);

    board.turn = Color.Red;
    const moves = MoveGenerator.generateMoves(board);

    // The cannon at 0 can:
    // 1. Move to adjacent empty square (which is 8, down)
    // 2. Hop over 1 and capture 3
    
    // We expect one capture move to square 3
    const captures = moves.filter(m => getMoveType(m) === MoveType.Capture);
    expect(captures.length).toBe(1);
    expect(getTo(captures[0])).toBe(3);
  });
});
