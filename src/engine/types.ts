/**
 * Core types and enums for DarkChess (Banqi) engine.
 */

export enum Color {
    RED = 0,
    BLACK = 1,
    NONE = 2
}

export enum PieceType {
    KING = 0,    // 帥/將
    ADVISOR = 1, // 仕/士
    ELEPHANT = 2,// 相/象
    CHARIOT = 3, // 俥/車
    HORSE = 4,   // 傌/馬
    CANNON = 5,  // 炮/包
    PAWN = 6,    // 兵/卒
    EMPTY = 7
}

export enum Piece {
    R_KING = 0, R_ADVISOR = 1, R_ELEPHANT = 2, R_CHARIOT = 3, R_HORSE = 4, R_CANNON = 5, R_PAWN = 6,
    B_KING = 7, B_ADVISOR = 8, B_ELEPHANT = 9, B_CHARIOT = 10, B_HORSE = 11, B_CANNON = 12, B_PAWN = 13,
    EMPTY = 14,
    HIDDEN = 15 // A piece that hasn't been flipped yet
}

export const PIECE_COLOR = [
    Color.RED, Color.RED, Color.RED, Color.RED, Color.RED, Color.RED, Color.RED,
    Color.BLACK, Color.BLACK, Color.BLACK, Color.BLACK, Color.BLACK, Color.BLACK, Color.BLACK,
    Color.NONE, Color.NONE
];

export const PIECE_TYPE = [
    PieceType.KING, PieceType.ADVISOR, PieceType.ELEPHANT, PieceType.CHARIOT, PieceType.HORSE, PieceType.CANNON, PieceType.PAWN,
    PieceType.KING, PieceType.ADVISOR, PieceType.ELEPHANT, PieceType.CHARIOT, PieceType.HORSE, PieceType.CANNON, PieceType.PAWN,
    PieceType.EMPTY, PieceType.EMPTY
];

/**
 * 32-bit Integer Bitboard representation for 4x8 board.
 * Bit 0 is top-left (0,0). Bit 31 is bottom-right (3,7).
 */
export type Bitboard = number;

/**
 * Move Encoding: (from << 5) | to
 * If from == to, it represents a FLIP action on that square.
 */
export type MoveId = number;

export function encodeMove(from: number, to: number): MoveId {
    return (from << 5) | to;
}

export function decodeMove(move: MoveId): { from: number, to: number, isFlip: boolean } {
    const from = (move >> 5) & 31;
    const to = move & 31;
    return { from, to, isFlip: from === to };
}
