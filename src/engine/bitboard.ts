/**
 * Bitboard utilities for 32-bit integer operations.
 * Board is 4x8. Index = y * 8 + x (0 to 31).
 */

import { Bitboard } from './types';

// Sets the bit at the given index
export function setBit(bb: Bitboard, index: number): Bitboard {
    return (bb | (1 << index)) >>> 0;
}

// Clears the bit at the given index
export function clearBit(bb: Bitboard, index: number): Bitboard {
    return (bb & ~(1 << index)) >>> 0;
}

// Checks if the bit at the given index is set
export function hasBit(bb: Bitboard, index: number): boolean {
    return ((bb >>> index) & 1) !== 0;
}

// Counts the number of set bits (Brian Kernighan's algorithm)
export function popcount(bb: Bitboard): number {
    let count = 0;
    let n = bb >>> 0;
    while (n !== 0) {
        n &= n - 1;
        count++;
    }
    return count;
}

// Gets the index of the least significant set bit (LSB)
export function lsb(bb: Bitboard): number {
    if (bb === 0) return -1;
    return popcount((bb & -bb) - 1);
}

// Iterator for extracting all set bits
export function* iterateBits(bb: Bitboard): Generator<number> {
    let n = bb >>> 0;
    while (n !== 0) {
        const isolated = n & -n;
        yield popcount(isolated - 1);
        n ^= isolated;
    }
}

// Precompute adjacency masks for fast move generation
export const ADJACENT_MASKS: Bitboard[] = new Array(32).fill(0);
export const ROW_MASKS: Bitboard[] = new Array(4).fill(0);
export const COL_MASKS: Bitboard[] = new Array(8).fill(0);

function initMasks() {
    for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 8; x++) {
            const sq = y * 8 + x;
            let mask = 0;
            if (y > 0) mask |= 1 << ((y - 1) * 8 + x); // UP
            if (y < 3) mask |= 1 << ((y + 1) * 8 + x); // DOWN
            if (x > 0) mask |= 1 << (y * 8 + (x - 1)); // LEFT
            if (x < 7) mask |= 1 << (y * 8 + (x + 1)); // RIGHT
            ADJACENT_MASKS[sq] = mask >>> 0;
            
            ROW_MASKS[y] |= (1 << sq);
            COL_MASKS[x] |= (1 << sq);
        }
    }
}

initMasks();
