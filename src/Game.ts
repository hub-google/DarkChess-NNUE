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

    // Mulberry32 gives reproducible shuffles when a seed is supplied.
    let state = seed === undefined ? 0 : seed >>> 0;
    const random = seed === undefined
      ? Math.random
      : () => {
          state = (state + 0x6D2B79F5) >>> 0;
          let value = state;
          value = Math.imul(value ^ (value >>> 15), value | 1);
          value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
          return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
        };

    // Fisher-Yates shuffle
    for (let i = pieces.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [pieces[i], pieces[j]] = [pieces[j], pieces[i]];
    }

    // Place on board
    for (let i = 0; i < 32; i++) {
      board.setPiece(i, pieces[i], false);
    }

    return board;
  }
}
