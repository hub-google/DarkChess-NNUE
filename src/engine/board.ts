import { Bitboard, Color, Piece, PieceType, MoveId, encodeMove, decodeMove, PIECE_COLOR, PIECE_TYPE } from './types';
import { setBit, clearBit, hasBit, popcount, iterateBits, ADJACENT_MASKS, ROW_MASKS, COL_MASKS } from './bitboard';

export class DarkChessBoard {
    pieceBitboards: Bitboard[];
    hiddenBitboard: Bitboard;
    occupiedBitboard: Bitboard; // Only face-up pieces
    
    // The actual hidden pieces for each square. Index is square (0-31).
    // If a square is flipped, its value here is not used anymore.
    hiddenPieces: Piece[];
    // Public belief state. The AI may read these counts, but it must never
    // inspect hiddenPieces when choosing a move.
    remainingCounts: number[];
    
    sideToMove: Color;
    halfMoveClock: number; // For 50-move rule
    history: string[]; // For 3-fold repetition
    tokenAtSquare: number[];
    chaseThreats: Array<[number, number, number, Color, number[]]>;
    pendingChase: [number, number, number, Color, number[]] | null;
    
    constructor() {
        this.pieceBitboards = new Array(14).fill(0);
        this.hiddenBitboard = 0xFFFFFFFF; // All 32 bits set
        this.occupiedBitboard = 0;
        this.hiddenPieces = new Array(32).fill(Piece.EMPTY);
        this.remainingCounts = [1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5];
        this.sideToMove = Color.NONE; // First move determines color
        this.halfMoveClock = 0;
        this.history = [];
        this.tokenAtSquare = Array.from({ length: 32 }, (_, square) => square);
        this.chaseThreats = [];
        this.pendingChase = null;
        this.initRandomBoard();
    }

    private initRandomBoard() {
        const bag = [
            Piece.R_KING, Piece.R_ADVISOR, Piece.R_ADVISOR, Piece.R_ELEPHANT, Piece.R_ELEPHANT, Piece.R_CHARIOT, Piece.R_CHARIOT, Piece.R_HORSE, Piece.R_HORSE, Piece.R_CANNON, Piece.R_CANNON, Piece.R_PAWN, Piece.R_PAWN, Piece.R_PAWN, Piece.R_PAWN, Piece.R_PAWN,
            Piece.B_KING, Piece.B_ADVISOR, Piece.B_ADVISOR, Piece.B_ELEPHANT, Piece.B_ELEPHANT, Piece.B_CHARIOT, Piece.B_CHARIOT, Piece.B_HORSE, Piece.B_HORSE, Piece.B_CANNON, Piece.B_CANNON, Piece.B_PAWN, Piece.B_PAWN, Piece.B_PAWN, Piece.B_PAWN, Piece.B_PAWN
        ];
        
        // Fisher-Yates shuffle
        for (let i = bag.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [bag[i], bag[j]] = [bag[j], bag[i]];
        }
        
        for (let i = 0; i < 32; i++) {
            this.hiddenPieces[i] = bag[i];
        }
    }

    public clone(): DarkChessBoard {
        const board = Object.create(DarkChessBoard.prototype) as DarkChessBoard;
        board.pieceBitboards = [...this.pieceBitboards];
        board.hiddenBitboard = this.hiddenBitboard >>> 0;
        board.occupiedBitboard = this.occupiedBitboard >>> 0;
        board.hiddenPieces = [...this.hiddenPieces];
        board.remainingCounts = [...this.remainingCounts];
        board.sideToMove = this.sideToMove;
        board.halfMoveClock = this.halfMoveClock;
        board.history = [...this.history];
        board.tokenAtSquare = [...this.tokenAtSquare];
        board.chaseThreats = this.chaseThreats.map(item => [item[0], item[1], item[2], item[3], [...item[4]]]);
        board.pendingChase = this.pendingChase ? [this.pendingChase[0], this.pendingChase[1], this.pendingChase[2], this.pendingChase[3], [...this.pendingChase[4]]] : null;
        return board;
    }

    private canCapture(attacker: Piece, victim: Piece): boolean {
        const attType = PIECE_TYPE[attacker];
        const vicType = PIECE_TYPE[victim];
        
        if (attType === PieceType.CANNON) {
            return true; // Cannon can capture anything if it jumps
        }
        
        if (attType === PieceType.KING && vicType === PieceType.PAWN) return false;
        if (attType === PieceType.PAWN && vicType === PieceType.KING) return true;
        
        return attType <= vicType; // Lower enum value means higher rank (0=KING)
    }

    public generateLegalMoves(): MoveId[] {
        const moves: MoveId[] = [];
        
        // 1. Flip Moves
        for (const sq of iterateBits(this.hiddenBitboard)) {
            moves.push(encodeMove(sq, sq));
        }

        // 2. Regular Moves & Captures (only if side is decided)
        if (this.sideToMove !== Color.NONE && this.hiddenBitboard === 0) {
            const allyColor = this.sideToMove;
            const enemyColor = 1 - allyColor; // 0->1, 1->0
            
            let allyMask = 0;
            let enemyMask = 0;
            for (let i = 0; i < 14; i++) {
                if (PIECE_COLOR[i] === allyColor) allyMask |= this.pieceBitboards[i];
                else if (PIECE_COLOR[i] === enemyColor) enemyMask |= this.pieceBitboards[i];
            }
            
            const emptyMask = ~(this.occupiedBitboard | this.hiddenBitboard) >>> 0;

            for (let p = 0; p < 14; p++) {
                if (PIECE_COLOR[p] !== allyColor) continue;
                
                const pType = PIECE_TYPE[p];
                for (const sq of iterateBits(this.pieceBitboards[p])) {
                    const adj = ADJACENT_MASKS[sq];
                    
                    // Normal Moves
                    const moveTargets = adj & emptyMask;
                    for (const target of iterateBits(moveTargets)) {
                        moves.push(encodeMove(sq, target));
                    }
                    
                    // Normal Captures (Non-Cannon)
                    if (pType !== PieceType.CANNON) {
                        const captureTargets = adj & enemyMask;
                        for (const target of iterateBits(captureTargets)) {
                            // Find which enemy piece is at target
                            for (let e = 0; e < 14; e++) {
                                if (PIECE_COLOR[e] === enemyColor && hasBit(this.pieceBitboards[e], target)) {
                                    if (this.canCapture(p, e)) {
                                        moves.push(encodeMove(sq, target));
                                    }
                                    break;
                                }
                            }
                        }
                    } 
                    // Cannon Captures
                    else {
                        const row = Math.floor(sq / 8);
                        const col = sq % 8;
                        
                        // Check jumping targets in row and col
                        this.generateCannonJumps(sq, row, col, enemyColor, moves);
                    }
                }
            }
        }
        
        if (!this.pendingChase) return moves;
        return moves.filter(move => !this.continuesForbiddenChase(move));
    }

    private pieceAt(square: number): Piece | undefined {
        for (let piece = 0; piece < 14; piece++) if (hasBit(this.pieceBitboards[piece], square)) return piece;
        return undefined;
    }

    private squareOfToken(token: number): number { return this.tokenAtSquare.indexOf(token); }

    private canAttack(attackerSquare: number, targetSquare: number, blockers?: number): boolean {
        const attacker = this.pieceAt(attackerSquare), victim = this.pieceAt(targetSquare);
        if (attacker === undefined || victim === undefined || PIECE_COLOR[attacker] === PIECE_COLOR[victim]) return false;
        if (PIECE_TYPE[attacker] !== PieceType.CANNON) {
            return hasBit(ADJACENT_MASKS[attackerSquare], targetSquare) && this.canCapture(attacker, victim);
        }
        if (Math.floor(attackerSquare / 8) !== Math.floor(targetSquare / 8) && attackerSquare % 8 !== targetSquare % 8) return false;
        const sameRow = Math.floor(attackerSquare / 8) === Math.floor(targetSquare / 8);
        const step = sameRow ? (targetSquare > attackerSquare ? 1 : -1) : (targetSquare > attackerSquare ? 8 : -8);
        const occupied = blockers === undefined ? (this.occupiedBitboard | this.hiddenBitboard) >>> 0 : blockers >>> 0;
        let screens = 0;
        for (let at = attackerSquare + step; at !== targetSquare; at += step) if (hasBit(occupied, at)) screens++;
        return screens === 1;
    }

    private threatenedTokens(attackerSquare: number): number[] {
        const targets: number[] = [];
        for (let square = 0; square < 32; square++) if (this.canAttack(attackerSquare, square)) targets.push(this.tokenAtSquare[square]);
        return targets;
    }

    private continuesForbiddenChase(move: MoveId): boolean {
        if (!this.pendingChase) return false;
        const [attackerToken, targetToken] = this.pendingChase;
        const { from, to, isFlip } = decodeMove(move);
        if (isFlip || this.tokenAtSquare[from] !== attackerToken || hasBit(this.occupiedBitboard, to)) return false;
        const target = this.squareOfToken(targetToken), attacker = this.pieceAt(from), victim = this.pieceAt(target);
        if (target < 0 || attacker === undefined || victim === undefined) return false;
        let continues = false;
        if (PIECE_TYPE[attacker] !== PieceType.CANNON) {
            continues = hasBit(ADJACENT_MASKS[to], target) && this.canCapture(attacker, victim);
        } else {
            let blockers = (this.occupiedBitboard | this.hiddenBitboard) >>> 0;
            blockers = clearBit(blockers, from); blockers = setBit(blockers, to);
            if (Math.floor(to / 8) === Math.floor(target / 8) || to % 8 === target % 8) {
                const sameRow = Math.floor(to / 8) === Math.floor(target / 8);
                const step = sameRow ? (target > to ? 1 : -1) : (target > to ? 8 : -8);
                let screens = 0;
                for (let at = to + step; at !== target; at += step) if (hasBit(blockers, at)) screens++;
                continues = screens === 1;
            }
        }
        if (!continues) return false;
        const route = this.pendingChase[4];
        return new Set(route).size !== route.length;
    }

    private updateChase(mover: Color, movedToken: number, isFlip: boolean, isCapture: boolean) {
        const oldThreats = this.chaseThreats, oldPending = this.pendingChase;
        this.chaseThreats = []; this.pendingChase = null;
        if (isFlip || isCapture) return;
        const movedSquare = this.squareOfToken(movedToken);
        if (movedSquare < 0) return;
        for (const record of oldThreats) {
            const [attackerToken, targetToken, count, chaser, route] = record;
            if (movedToken !== targetToken || mover === chaser) continue;
            const attackerSquare = this.squareOfToken(attackerToken);
            if (attackerSquare >= 0 && !this.canAttack(attackerSquare, movedSquare)) {
                this.pendingChase = [attackerToken, targetToken, count, chaser, [...route, movedSquare]];
                return;
            }
        }
        for (const targetToken of this.threatenedTokens(movedSquare)) {
            const count = oldPending && oldPending[0] === movedToken && oldPending[1] === targetToken && oldPending[3] === mover ? oldPending[2] + 1 : 1;
            const route = oldPending && oldPending[0] === movedToken && oldPending[1] === targetToken && oldPending[3] === mover
                ? oldPending[4] : [this.squareOfToken(targetToken)];
            this.chaseThreats.push([movedToken, targetToken, count, mover, route]);
        }
    }

    private generateCannonJumps(sq: number, row: number, col: number, enemyColor: Color, moves: MoveId[]) {
        const allBlockers = this.occupiedBitboard | this.hiddenBitboard;
        
        // Scan directions: -x, +x, -y, +y
        const dirs = [-1, 1, -8, 8];
        for (const d of dirs) {
            let current = sq + d;
            let jumped = false;
            
            while (current >= 0 && current < 32) {
                // Bounds check for left/right
                if (d === -1 && current % 8 === 7) break;
                if (d === 1 && current % 8 === 0) break;
                
                if (hasBit(allBlockers, current)) {
                    if (!jumped) {
                        jumped = true; // Found the mount (cannon platform)
                    } else {
                        // This is the second piece. Check if it's an enemy
                        for (let e = 0; e < 14; e++) {
                            if (PIECE_COLOR[e] === enemyColor && hasBit(this.pieceBitboards[e], current)) {
                                moves.push(encodeMove(sq, current));
                                break;
                            }
                        }
                        break; // Stop scanning this direction after hitting second piece
                    }
                }
                current += d;
            }
        }
    }

    public makeMove(move: MoveId, flippedPieceOverride?: Piece) {
        if (!this.generateLegalMoves().includes(move)) {
            throw new Error(`Illegal move: ${move}`);
        }

        const { from, to, isFlip } = decodeMove(move);
        const mover = this.sideToMove;
        const movedToken = this.tokenAtSquare[from];
        let isCapture = false;
        let flippedPiece: Piece | undefined;
        if (isFlip) {
            flippedPiece = flippedPieceOverride ?? this.hiddenPieces[from];
            if (flippedPiece < Piece.R_KING || flippedPiece > Piece.B_PAWN) {
                throw new Error(`Invalid flipped piece: ${flippedPiece}`);
            }
            if (this.remainingCounts[flippedPiece] <= 0) {
                throw new Error(`Piece ${flippedPiece} is not available in the bag`);
            }
        }

        // Save history only after all validation has passed.
        this.history.push(this.getSnapshot());

        if (isFlip) {
            this.hiddenBitboard = clearBit(this.hiddenBitboard, from);
            this.pieceBitboards[flippedPiece!] = setBit(this.pieceBitboards[flippedPiece!], from);
            this.occupiedBitboard = setBit(this.occupiedBitboard, from);
            this.remainingCounts[flippedPiece!]--;
            this.halfMoveClock = 0; // Reset 50-move rule
            
            // If first move, assign color
            if (this.sideToMove === Color.NONE) {
                this.sideToMove = PIECE_COLOR[flippedPiece!]; // First player controls the color they flipped
            }
        } else {
            // Find moving piece
            let piece = -1;
            for (let i = 0; i < 14; i++) {
                if (hasBit(this.pieceBitboards[i], from)) {
                    piece = i;
                    break;
                }
            }
            
            // Remove piece from source
            this.pieceBitboards[piece] = clearBit(this.pieceBitboards[piece], from);
            this.occupiedBitboard = clearBit(this.occupiedBitboard, from);
            
            // Check if target is a capture
            if (hasBit(this.occupiedBitboard, to)) {
                isCapture = true;
                // Remove victim
                for (let i = 0; i < 14; i++) {
                    if (hasBit(this.pieceBitboards[i], to)) {
                        this.pieceBitboards[i] = clearBit(this.pieceBitboards[i], to);
                        this.tokenAtSquare[to] = -1;
                        break;
                    }
                }
            }
            
            // Place piece at destination
            this.pieceBitboards[piece] = setBit(this.pieceBitboards[piece], to);
            this.occupiedBitboard = setBit(this.occupiedBitboard, to);
            this.tokenAtSquare[to] = movedToken;
            this.tokenAtSquare[from] = -1;
            
            if (isCapture) {
                this.halfMoveClock = 0;
            } else {
                this.halfMoveClock++;
            }
        }

        this.updateChase(mover, movedToken, isFlip, isFlip ? false : isCapture);
        
        // Switch turn (except if sideToMove is still NONE, which shouldn't happen after first flip)
        if (this.sideToMove !== Color.NONE) {
            this.sideToMove = 1 - this.sideToMove;
        }
    }

    private getSnapshot(): string {
        return `${this.pieceBitboards.join(',')}|${this.hiddenBitboard}|${this.remainingCounts.join(',')}|${this.sideToMove}`;
    }

    public hiddenProbability(piece: Piece): number {
        const total = this.remainingCounts.reduce((sum, count) => sum + count, 0);
        return total === 0 ? 0 : this.remainingCounts[piece] / total;
    }

    public isGameOver(): { over: boolean, result: number } {
        // A color loses as soon as it has no visible or face-down pieces left.
        // This is public information because remainingCounts is known.
        if (this.colorPieceCount(Color.RED) === 0) return { over: true, result: -1.0 };
        if (this.colorPieceCount(Color.BLACK) === 0) return { over: true, result: 1.0 };

        if (this.halfMoveClock >= 60) return { over: true, result: 0.0 };

        // Check if trapped (no legal moves for current player)
        if (this.sideToMove !== Color.NONE) {
            const moves = this.generateLegalMoves();
            if (moves.length === 0) {
                // Result is always stored from Red's perspective.
                return { over: true, result: this.sideToMove === Color.RED ? -1.0 : 1.0 };
            }
        }
        
        return { over: false, result: 0.0 };
    }

    private colorPieceCount(color: Color): number {
        let count = 0;
        for (let i = 0; i < 14; i++) {
            if (PIECE_COLOR[i] === color) {
                count += popcount(this.pieceBitboards[i]);
                count += this.remainingCounts[i];
            }
        }
        return count;
    }
}
