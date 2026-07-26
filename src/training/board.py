import numpy as np
from numba import njit
from bitboard import ADJACENT_MASKS, set_bit, clear_bit, has_bit, popcount, iterate_bits

# Enums
RED, BLACK, NONE = 0, 1, 2
KING, ADVISOR, ELEPHANT, CHARIOT, HORSE, CANNON, PAWN, EMPTY_TYPE = 0, 1, 2, 3, 4, 5, 6, 7

PIECE_COLOR = np.array([
    RED, RED, RED, RED, RED, RED, RED,
    BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK,
    NONE, NONE
], dtype=np.int32)

PIECE_TYPE = np.array([
    KING, ADVISOR, ELEPHANT, CHARIOT, HORSE, CANNON, PAWN,
    KING, ADVISOR, ELEPHANT, CHARIOT, HORSE, CANNON, PAWN,
    EMPTY_TYPE, EMPTY_TYPE
], dtype=np.int32)

@njit
def encode_move(from_sq, to_sq):
    return (from_sq << 5) | to_sq

@njit
def decode_move(move):
    from_sq = (move >> 5) & 31
    to_sq = move & 31
    return from_sq, to_sq, from_sq == to_sq

@njit
def can_capture(attacker, victim):
    att_type = PIECE_TYPE[attacker]
    vic_type = PIECE_TYPE[victim]
    
    if att_type == CANNON:
        return True
        
    if att_type == KING and vic_type == PAWN: return False
    if att_type == PAWN and vic_type == KING: return True
    
    return att_type <= vic_type

class DarkChessBoardPy:
    def __init__(self, bag=None):
        self.piece_bitboards = np.zeros(14, dtype=np.uint32)
        self.hidden_bitboard = np.uint32(0xFFFFFFFF)
        self.occupied_bitboard = np.uint32(0)
        self.hidden_pieces = np.zeros(32, dtype=np.int32)
        self.side_to_move = NONE
        self.half_move_clock = 0
        self.history = []
        self._init_random_board(bag)
        
    def _init_random_board(self, bag=None):
        if bag is None:
            bag = np.array([
                0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6,
                7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13, 13
            ], dtype=np.int32)
            np.random.shuffle(bag)
        else:
            bag = np.array(bag, dtype=np.int32)
        self.hidden_pieces = bag
        
    def get_snapshot(self):
        # We use a tuple for history since it's hashable and fast
        return (tuple(self.piece_bitboards), self.hidden_bitboard, self.side_to_move)

    def generate_legal_moves(self):
        # This function acts as a wrapper. The heavy lifting should be jitted.
        return _generate_legal_moves(
            self.piece_bitboards, 
            self.hidden_bitboard, 
            self.occupied_bitboard, 
            self.side_to_move
        )

    def make_move(self, move):
        self.history.append(self.get_snapshot())
        from_sq, to_sq, is_flip = decode_move(move)
        
        if is_flip:
            flipped_piece = self.hidden_pieces[from_sq]
            self.hidden_bitboard = clear_bit(self.hidden_bitboard, from_sq)
            self.piece_bitboards[flipped_piece] = set_bit(self.piece_bitboards[flipped_piece], from_sq)
            self.occupied_bitboard = set_bit(self.occupied_bitboard, from_sq)
            self.half_move_clock = 0
            
            if self.side_to_move == NONE:
                self.side_to_move = PIECE_COLOR[flipped_piece]
        else:
            # Find piece
            piece = -1
            for i in range(14):
                if has_bit(self.piece_bitboards[i], from_sq):
                    piece = i
                    break
                    
            self.piece_bitboards[piece] = clear_bit(self.piece_bitboards[piece], from_sq)
            self.occupied_bitboard = clear_bit(self.occupied_bitboard, from_sq)
            
            is_capture = False
            if has_bit(self.occupied_bitboard, to_sq):
                is_capture = True
                for i in range(14):
                    if has_bit(self.piece_bitboards[i], to_sq):
                        self.piece_bitboards[i] = clear_bit(self.piece_bitboards[i], to_sq)
                        break
                        
            self.piece_bitboards[piece] = set_bit(self.piece_bitboards[piece], to_sq)
            self.occupied_bitboard = set_bit(self.occupied_bitboard, to_sq)
            
            if is_capture:
                self.half_move_clock = 0
            else:
                self.half_move_clock += 1
                
        if self.side_to_move != NONE:
            self.side_to_move = 1 - self.side_to_move

@njit
def _generate_legal_moves(piece_bitboards, hidden_bitboard, occupied_bitboard, side_to_move):
    moves = np.zeros(1024, dtype=np.int32)
    move_count = 0
    
    hidden_sqs, hidden_count = iterate_bits(hidden_bitboard)
    for i in range(hidden_count):
        sq = hidden_sqs[i]
        moves[move_count] = encode_move(sq, sq)
        move_count += 1
        
    if side_to_move != NONE:
        ally_color = side_to_move
        enemy_color = 1 - ally_color
        
        ally_mask = np.uint32(0)
        enemy_mask = np.uint32(0)
        
        for i in range(14):
            if PIECE_COLOR[i] == ally_color: ally_mask |= piece_bitboards[i]
            elif PIECE_COLOR[i] == enemy_color: enemy_mask |= piece_bitboards[i]
            
        empty_mask = ~(occupied_bitboard | hidden_bitboard)
        
        for p in range(14):
            if PIECE_COLOR[p] != ally_color: continue
            
            p_type = PIECE_TYPE[p]
            sqs, sqs_count = iterate_bits(piece_bitboards[p])
            
            for i in range(sqs_count):
                sq = sqs[i]
                adj = ADJACENT_MASKS[sq]
                
                # Normal moves
                move_targets = adj & empty_mask
                t_sqs, t_count = iterate_bits(move_targets)
                for j in range(t_count):
                    moves[move_count] = encode_move(sq, t_sqs[j])
                    move_count += 1
                    
                # Captures
                if p_type != CANNON:
                    cap_targets = adj & enemy_mask
                    c_sqs, c_count = iterate_bits(cap_targets)
                    for j in range(c_count):
                        target = c_sqs[j]
                        # Find enemy piece
                        for e in range(14):
                            if PIECE_COLOR[e] == enemy_color and has_bit(piece_bitboards[e], target):
                                if can_capture(p, e):
                                    moves[move_count] = encode_move(sq, target)
                                    move_count += 1
                                break
                else:
                    # Cannon logic
                    all_blockers = occupied_bitboard | hidden_bitboard
                    dirs = np.array([-1, 1, -8, 8], dtype=np.int32)
                    for d in dirs:
                        current = sq + d
                        jumped = False
                        while current >= 0 and current < 32:
                            if d == -1 and current % 8 == 7: break
                            if d == 1 and current % 8 == 0: break
                            
                            if has_bit(all_blockers, current):
                                if not jumped:
                                    jumped = True
                                else:
                                    for e in range(14):
                                        if PIECE_COLOR[e] == enemy_color and has_bit(piece_bitboards[e], current):
                                            moves[move_count] = encode_move(sq, current)
                                            move_count += 1
                                            break
                                    break
                            current += d
                            
    return moves[:move_count]
