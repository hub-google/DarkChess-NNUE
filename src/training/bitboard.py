import numpy as np
from numba import njit

# Masks
ADJACENT_MASKS = np.zeros(32, dtype=np.uint32)
ROW_MASKS = np.zeros(4, dtype=np.uint32)
COL_MASKS = np.zeros(8, dtype=np.uint32)

def init_masks():
    for y in range(4):
        for x in range(8):
            sq = y * 8 + x
            mask = np.uint32(0)
            if y > 0: mask |= np.uint32(1 << ((y - 1) * 8 + x)) # UP
            if y < 3: mask |= np.uint32(1 << ((y + 1) * 8 + x)) # DOWN
            if x > 0: mask |= np.uint32(1 << (y * 8 + (x - 1))) # LEFT
            if x < 7: mask |= np.uint32(1 << (y * 8 + (x + 1))) # RIGHT
            ADJACENT_MASKS[sq] = mask
            
            ROW_MASKS[y] |= np.uint32(1 << sq)
            COL_MASKS[x] |= np.uint32(1 << sq)

init_masks()

@njit
def set_bit(bb, index):
    return bb | np.uint32(1 << index)

@njit
def clear_bit(bb, index):
    return bb & np.uint32(0xFFFFFFFF ^ (1 << index))

@njit
def has_bit(bb, index):
    return ((bb >> index) & 1) != 0

@njit
def popcount(bb):
    # Brian Kernighan's algorithm
    count = 0
    n = bb
    while n != 0:
        n &= (n - 1)
        count += 1
    return count

@njit
def iterate_bits(bb):
    # Extracts all set bit indices. Returns a fixed size array and the count since numba doesn't support generators well.
    indices = np.zeros(32, dtype=np.int32)
    count = 0
    n = bb
    while n != 0:
        isolated = n & (np.uint32(~n) + np.uint32(1))
        indices[count] = popcount(isolated - 1)
        count += 1
        n ^= isolated
    return indices, count
