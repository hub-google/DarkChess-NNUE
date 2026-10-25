export type Bitboard = number;

export const EMPTY_BOARD: Bitboard = 0;
export const FULL_BOARD: Bitboard = -1; // All 32 bits set to 1

export const setBit = (bb: Bitboard, square: number): Bitboard => {
  return bb | (1 << square);
};

export const clearBit = (bb: Bitboard, square: number): Bitboard => {
  return bb & ~(1 << square);
};

export const testBit = (bb: Bitboard, square: number): boolean => {
  return (bb & (1 << square)) !== 0;
};

// 4x8 board, indices:
// 0  1  2  3  4  5  6  7
// 8  9 10 11 12 13 14 15
// 16 17 18 19 20 21 22 23
// 24 25 26 27 28 29 30 31

// Masks for boundaries
export const MASK_LEFT_EDGE = 0x01010101;
export const MASK_RIGHT_EDGE = 0x80808080;
export const MASK_TOP_EDGE = 0x000000FF;
export const MASK_BOTTOM_EDGE = 0xFF000000;

export const shiftUp = (bb: Bitboard): Bitboard => {
  return (bb >>> 8);
};

export const shiftDown = (bb: Bitboard): Bitboard => {
  return (bb << 8);
};

export const shiftLeft = (bb: Bitboard): Bitboard => {
  return (bb & ~MASK_LEFT_EDGE) >>> 1;
};

export const shiftRight = (bb: Bitboard): Bitboard => {
  return (bb & ~MASK_RIGHT_EDGE) << 1;
};

export const getAdjacent = (square: number): Bitboard => {
  const bb = setBit(0, square);
  return shiftUp(bb) | shiftDown(bb) | shiftLeft(bb) | shiftRight(bb);
};
