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

INITIAL_COUNTS = np.array(
    [1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5],
    dtype=np.int32,
)

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
        # Public belief state: how many pieces of each type are still face-down.
        # Search and evaluation must use this array instead of inspecting
        # hidden_pieces, which is private referee state.
        self.remaining_counts = INITIAL_COUNTS.copy()
        self.side_to_move = NONE
        self.half_move_clock = 0
        self.history = []
        # Stable public identities (the square where a piece started).  These
        # let the referee distinguish two pieces of the same type when applying
        # the perpetual-chase rule.
        self.token_at_square = np.arange(32, dtype=np.int32)
        self.chase_threats = []
        self.pending_chase = None
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
        if bag.shape != (32,):
            raise ValueError(f"bag must contain exactly 32 pieces, got {bag.shape}")
        if np.any(bag < 0) or np.any(bag >= 14):
            raise ValueError("bag contains an invalid piece id")
        counts = np.bincount(bag, minlength=14).astype(np.int32)
        if not np.array_equal(counts, INITIAL_COUNTS):
            raise ValueError(f"bag has invalid inventory: {counts.tolist()}")
        self.hidden_pieces = bag.copy()
        self.remaining_counts = INITIAL_COUNTS.copy()

    def clone(self):
        board = object.__new__(DarkChessBoardPy)
        board.piece_bitboards = self.piece_bitboards.copy()
        board.hidden_bitboard = np.uint32(self.hidden_bitboard)
        board.occupied_bitboard = np.uint32(self.occupied_bitboard)
        board.hidden_pieces = self.hidden_pieces.copy()
        board.remaining_counts = self.remaining_counts.copy()
        board.side_to_move = int(self.side_to_move)
        board.half_move_clock = int(self.half_move_clock)
        board.history = list(self.history)
        board.token_at_square = self.token_at_square.copy()
        board.chase_threats = list(self.chase_threats)
        board.pending_chase = self.pending_chase
        return board
        
    def get_snapshot(self):
        # We use a tuple for history since it's hashable and fast
        return (
            tuple(int(value) for value in self.piece_bitboards),
            int(self.hidden_bitboard),
            tuple(int(value) for value in self.remaining_counts),
            int(self.side_to_move),
        )

    def generate_legal_moves(self):
        # This function acts as a wrapper. The heavy lifting should be jitted.
        moves = _generate_legal_moves(
            self.piece_bitboards, 
            self.hidden_bitboard, 
            self.occupied_bitboard, 
            self.side_to_move
        )
        if self.pending_chase is None:
            return moves
        return np.array(
            [int(move) for move in moves if not self._continues_forbidden_chase(int(move))],
            dtype=np.int32,
        )

    def _piece_at(self, square):
        for piece in range(14):
            if has_bit(self.piece_bitboards[piece], square):
                return piece
        return -1

    def _square_of_token(self, token):
        matches = np.flatnonzero(self.token_at_square == int(token))
        return int(matches[0]) if len(matches) else -1

    def _can_attack(self, attacker_sq, target_sq, occupied=None):
        attacker = self._piece_at(attacker_sq)
        victim = self._piece_at(target_sq)
        if attacker < 0 or victim < 0 or PIECE_COLOR[attacker] == PIECE_COLOR[victim]:
            return False
        if PIECE_TYPE[attacker] != CANNON:
            return bool(has_bit(ADJACENT_MASKS[attacker_sq], target_sq) and can_capture(attacker, victim))
        if attacker_sq // 8 != target_sq // 8 and attacker_sq % 8 != target_sq % 8:
            return False
        step = (1 if target_sq > attacker_sq else -1) if attacker_sq // 8 == target_sq // 8 else (8 if target_sq > attacker_sq else -8)
        blockers = int(self.occupied_bitboard | self.hidden_bitboard) if occupied is None else int(occupied)
        screens = 0
        square = attacker_sq + step
        while square != target_sq:
            if (blockers >> square) & 1:
                screens += 1
            square += step
        return screens == 1

    def _threatened_tokens(self, attacker_sq):
        result = []
        for target_sq in range(32):
            if self._can_attack(attacker_sq, target_sq):
                result.append(int(self.token_at_square[target_sq]))
        return result

    def _continues_forbidden_chase(self, move):
        attacker_token, target_token, count, _, _route = self.pending_chase
        from_sq, to_sq, is_flip = decode_move(move)
        if is_flip or int(self.token_at_square[from_sq]) != attacker_token:
            return False
        # Capturing the chased piece ends the chase; it is never a forbidden chase move.
        if has_bit(self.occupied_bitboard, to_sq):
            return False
        target_sq = self._square_of_token(target_token)
        if target_sq < 0:
            return False
        attacker = self._piece_at(from_sq)
        victim = self._piece_at(target_sq)
        if attacker < 0 or victim < 0:
            return False
        continues = False
        if PIECE_TYPE[attacker] != CANNON:
            continues = bool(has_bit(ADJACENT_MASKS[to_sq], target_sq) and can_capture(attacker, victim))
        else:
            occupied = int(self.occupied_bitboard | self.hidden_bitboard)
            occupied &= ~(1 << from_sq)
            occupied |= 1 << to_sq
            if to_sq // 8 == target_sq // 8 or to_sq % 8 == target_sq % 8:
                step = (1 if target_sq > to_sq else -1) if to_sq // 8 == target_sq // 8 else (8 if target_sq > to_sq else -8)
                screens = 0
                square = to_sq + step
                while square != target_sq:
                    screens += (occupied >> square) & 1
                    square += step
                continues = screens == 1
        if not continues:
            return False
        route = self.pending_chase[4]
        return len(route) != len(set(route))

    def _update_chase(self, mover, moved_token, is_flip, is_capture):
        old_threats = self.chase_threats
        old_pending = self.pending_chase
        self.chase_threats = []
        self.pending_chase = None
        if is_flip or is_capture:
            return
        moved_sq = self._square_of_token(moved_token)
        if moved_sq < 0:
            return
        if old_threats:
            # Only moving the specifically threatened piece to safety is an
            # escape response that permits the same chase sequence to continue.
            for attacker_token, target_token, count, chaser, route in old_threats:
                if moved_token != target_token or mover == chaser:
                    continue
                attacker_sq = self._square_of_token(attacker_token)
                if attacker_sq >= 0 and not self._can_attack(attacker_sq, moved_sq):
                    self.pending_chase = (attacker_token, target_token, count, chaser, route + (moved_sq,))
                    return
        threatened = self._threatened_tokens(moved_sq)
        for target_token in threatened:
            count = 1
            if old_pending is not None and old_pending[0] == moved_token and old_pending[1] == target_token and old_pending[3] == mover:
                count = old_pending[2] + 1
            route = (self._square_of_token(target_token),)
            if old_pending is not None and old_pending[0] == moved_token and old_pending[1] == target_token and old_pending[3] == mover:
                route = old_pending[4]
            self.chase_threats.append((moved_token, target_token, count, mover, route))

    def make_move(self, move, flip_piece=None, validate=True):
        move = int(move)
        if validate and move not in set(int(m) for m in self.generate_legal_moves()):
            raise ValueError(f"illegal move: {move}")

        from_sq, to_sq, is_flip = decode_move(move)
        mover = int(self.side_to_move)
        moved_token = int(self.token_at_square[from_sq])
        flipped_piece = None
        if is_flip:
            flipped_piece = (
                int(self.hidden_pieces[from_sq])
                if flip_piece is None
                else int(flip_piece)
            )
            if flipped_piece < 0 or flipped_piece >= 14:
                raise ValueError(f"invalid flipped piece: {flipped_piece}")
            if self.remaining_counts[flipped_piece] <= 0:
                raise ValueError(f"piece {flipped_piece} is not available in the bag")

        self.history.append(self.get_snapshot())

        if is_flip:
            self.hidden_bitboard = clear_bit(self.hidden_bitboard, from_sq)
            self.piece_bitboards[flipped_piece] = set_bit(self.piece_bitboards[flipped_piece], from_sq)
            self.occupied_bitboard = set_bit(self.occupied_bitboard, from_sq)
            self.remaining_counts[flipped_piece] -= 1
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
                self.token_at_square[to_sq] = -1
                        
            self.piece_bitboards[piece] = set_bit(self.piece_bitboards[piece], to_sq)
            self.occupied_bitboard = set_bit(self.occupied_bitboard, to_sq)
            self.token_at_square[to_sq] = moved_token
            self.token_at_square[from_sq] = -1
            
            if is_capture:
                self.half_move_clock = 0
            else:
                self.half_move_clock += 1

        self._update_chase(mover, moved_token, is_flip, False if is_flip else is_capture)
                
        if self.side_to_move != NONE:
            self.side_to_move = 1 - self.side_to_move

    def hidden_probability(self, piece):
        hidden_total = int(self.remaining_counts.sum())
        if hidden_total == 0:
            return 0.0
        return float(self.remaining_counts[int(piece)]) / hidden_total

    def repetition_count(self):
        current = self.get_snapshot()
        return 1 + sum(snapshot == current for snapshot in self.history)

    def _color_piece_count(self, color):
        visible = 0
        hidden = 0
        for piece in range(14):
            if PIECE_COLOR[piece] == color:
                visible += popcount(self.piece_bitboards[piece])
                hidden += int(self.remaining_counts[piece])
        return visible + hidden

    def is_game_over(self):
        red_count = self._color_piece_count(RED)
        black_count = self._color_piece_count(BLACK)
        if red_count == 0:
            return True, -1.0
        if black_count == 0:
            return True, 1.0

        if self.half_move_clock >= 60:
            return True, 0.0
        # A face-down square is always a legal flip, so the expensive move
        # generation is only needed once the bag is empty.
        if (
            self.side_to_move != NONE
            and int(self.hidden_bitboard) == 0
            and len(self.generate_legal_moves()) == 0
        ):
            return True, -1.0 if self.side_to_move == RED else 1.0
        return False, 0.0

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
