import { expect, test } from 'vitest';
import { DarkChessBoard } from '../src/engine/board';
import { Color, Piece, encodeMove } from '../src/engine/types';
import { popcount, setBit } from '../src/engine/bitboard';

test('DarkChessBoard Initialization', () => {
    const board = new DarkChessBoard();
    expect(popcount(board.hiddenBitboard)).toBe(32);
    expect(board.occupiedBitboard).toBe(0);
    expect(board.sideToMove).toBe(Color.NONE);
});

test('DarkChessBoard Legal Moves Initial State', () => {
    const board = new DarkChessBoard();
    const moves = board.generateLegalMoves();
    expect(moves.length).toBe(32); // Can only flip the 32 hidden pieces
});

test('a trapped Black side is a Red win', () => {
    const board = new DarkChessBoard();
    board.hiddenBitboard = 0;
    board.pieceBitboards.fill(0);
    board.pieceBitboards[Piece.B_KING] = setBit(0, 0);
    board.pieceBitboards[Piece.R_PAWN] = setBit(setBit(0, 1), 8);
    board.occupiedBitboard = setBit(setBit(setBit(0, 0), 1), 8);
    board.sideToMove = Color.BLACK;

    expect(board.generateLegalMoves()).toHaveLength(0);
    expect(board.isGameOver()).toEqual({ over: true, result: 1.0 });
});

test('makeMove rejects moves outside the legal move list', () => {
    const board = new DarkChessBoard();
    expect(() => board.makeMove(encodeMove(0, 1))).toThrow('Illegal move');
});

test('flipping updates the public remaining-piece probability', () => {
    const board = new DarkChessBoard();
    board.hiddenPieces[0] = Piece.R_KING;
    board.makeMove(encodeMove(0, 0));

    expect(board.remainingCounts[Piece.R_KING]).toBe(0);
    expect(board.hiddenProbability(Piece.R_KING)).toBe(0);
    expect(board.sideToMove).toBe(Color.BLACK);
});

test('a color with no visible or hidden pieces loses immediately', () => {
    const board = new DarkChessBoard();
    board.pieceBitboards.fill(0);
    board.occupiedBitboard = 0;
    board.hiddenBitboard = setBit(0, 0);
    board.hiddenPieces.fill(Piece.B_PAWN);
    board.remainingCounts.fill(0);
    board.remainingCounts[Piece.B_PAWN] = 1;
    board.sideToMove = Color.RED;

    expect(board.isGameOver()).toEqual({ over: true, result: -1.0 });
});
