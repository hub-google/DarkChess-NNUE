import { Board } from './Board';
import { Color, PieceType, makePiece } from './types';

export class Game {
  public board: Board;

  constructor() {
    this.board = new Board();
  }

  public static generateRandomBoard(seed?: number): Board {
    const board = new Board();
    const pieces: number[] = [];

    const addPieces = (color: Color) => {
      pieces.push(makePiece(color, PieceType.King));
      for (let i = 0; i < 2; i++) pieces.push(makePiece(color, PieceType.Advisor));
      for (let i = 0; i < 2; i++) pieces.push(makePiece(color, PieceType.Elephant));
      for (let i = 0; i < 2; i++) pieces.push(makePiece(color, PieceType.Rook));
      for (let i = 0; i < 2; i++) pieces.push(makePiece(color, PieceType.Knight));
      for (let i = 0; i < 2; i++) pieces.push(makePiece(color, PieceType.Cannon));
      for (let i = 0; i < 5; i++) pieces.push(makePiece(color, PieceType.Pawn));
    };

    addPieces(Color.Red);
    addPieces(Color.Black);

    // Fisher-Yates shuffle
    // In a real implementation we would use a seeded random number generator (PRNG) for reproducibility
    for (let i = pieces.length - 1; i > 0; i--) {
      // Basic Math.random() for now. A seeded PRNG should be used later for AB Testing
      const j = Math.floor(Math.random() * (i + 1));
      [pieces[i], pieces[j]] = [pieces[j], pieces[i]];
    }

    // Place on board
    for (let i = 0; i < 32; i++) {
      board.setPiece(i, pieces[i], false);
    }

    return board;
  }
}
