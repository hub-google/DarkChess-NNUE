import { expect, test } from 'vitest';
import { DarkChessBoard } from '../src/engine/board';
import { Color } from '../src/engine/types';
import { popcount } from '../src/engine/bitboard';

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
