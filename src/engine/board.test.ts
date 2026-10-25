import { describe, expect, it } from 'vitest';
import { DarkChessBoard } from './board';
import { Color, Piece, encodeMove } from './types';

describe('DarkChessBoard perpetual-chase rule', () => {
    it('allows advancing pursuit but removes a third repetition of its route', () => {
        const board = new DarkChessBoard();
        board.pieceBitboards.fill(0);
        board.hiddenBitboard = 0;
        board.remainingCounts.fill(0);
        board.sideToMove = Color.RED;
        board.pieceBitboards[Piece.R_ADVISOR] = (1 << 8) >>> 0;
        board.pieceBitboards[Piece.B_CHARIOT] = (1 << 1) >>> 0;
        board.occupiedBitboard = ((1 << 8) | (1 << 1)) >>> 0;

        for (const [from, to] of [
            [8, 0],
            [1, 9], [0, 1], [9, 8], [1, 9], [8, 0], [9, 8], [0, 1],
        ]) {
            board.makeMove(encodeMove(from, to));
        }

        expect(board.generateLegalMoves()).not.toContain(encodeMove(8, 0));
        expect(board.generateLegalMoves()).not.toContain(encodeMove(8, 9));
        expect(board.generateLegalMoves()).toContain(encodeMove(8, 16));
        expect(() => board.makeMove(encodeMove(8, 0))).toThrow(/Illegal move/);
    });

    it('does not end an ordinary third repetition as a draw', () => {
        const board = new DarkChessBoard();
        board.pieceBitboards.fill(0);
        board.hiddenBitboard = 0;
        board.remainingCounts.fill(0);
        board.sideToMove = Color.RED;
        board.pieceBitboards[Piece.R_CHARIOT] = 1;
        board.pieceBitboards[Piece.B_CHARIOT] = (1 << 31) >>> 0;
        board.occupiedBitboard = (1 | (1 << 31)) >>> 0;

        for (const [from, to] of [
            [0, 1], [31, 30], [1, 0], [30, 31],
            [0, 1], [31, 30], [1, 0], [30, 31],
        ]) board.makeMove(encodeMove(from, to));

        expect(board.isGameOver()).toEqual({ over: false, result: 0 });
    });
});
