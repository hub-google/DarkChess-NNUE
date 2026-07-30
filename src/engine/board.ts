import { Bitboard, Color, Piece, PieceType, MoveId, encodeMove, decodeMove, PIECE_COLOR, PIECE_TYPE } from './types';
import { setBit, clearBit, hasBit, popcount, iterateBits, ADJACENT_MASKS, ROW_MASKS, COL_MASKS } from './bitboard';

export class DarkChessBoard {
    pieceBitboards: Bitboard[];
    hiddenBitboard: Bitboard;
    occupiedBitboard: Bitboard; // Only face-up pieces
    
    // The actual hidden pieces for each square. Index is square (0-31).
    // If a square is flipped, its value here is not used anymore.
    hiddenPieces: Piece[];
    
    sideToMove: Color;
    halfMoveClock: number; // For 50-move rule
    history: string[]; // For 3-fold repetition
    
    constructor() {
        this.pieceBitboards = new Array(14).fill(0);
        this.hiddenBitboard = 0xFFFFFFFF; // All 32 bits set
        this.occupiedBitboard = 0;
        this.hiddenPieces = new Array(32).fill(Piece.EMPTY);
        this.sideToMove = Color.NONE; // First move determines color
        this.halfMoveClock = 0;
        this.history = [];
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
        if (this.sideToMove !== Color.NONE) {
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
        
        return moves;
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

    public makeMove(move: MoveId) {
        if (!this.generateLegalMoves().includes(move)) {
            throw new Error(`Illegal move: ${move}`);
        }

        const { from, to, isFlip } = decodeMove(move);
        
        // Save history for repetition check
        this.history.push(this.getSnapshot());

        if (isFlip) {
            const flippedPiece = this.hiddenPieces[from];
            this.hiddenBitboard = clearBit(this.hiddenBitboard, from);
            this.pieceBitboards[flippedPiece] = setBit(this.pieceBitboards[flippedPiece], from);
            this.occupiedBitboard = setBit(this.occupiedBitboard, from);
            this.halfMoveClock = 0; // Reset 50-move rule
            
            // If first move, assign color
            if (this.sideToMove === Color.NONE) {
                this.sideToMove = PIECE_COLOR[flippedPiece]; // First player controls the color they flipped
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
            
            let isCapture = false;
            // Check if target is a capture
            if (hasBit(this.occupiedBitboard, to)) {
                isCapture = true;
                // Remove victim
                for (let i = 0; i < 14; i++) {
                    if (hasBit(this.pieceBitboards[i], to)) {
                        this.pieceBitboards[i] = clearBit(this.pieceBitboards[i], to);
                        break;
                    }
                }
            }
            
            // Place piece at destination
            this.pieceBitboards[piece] = setBit(this.pieceBitboards[piece], to);
            this.occupiedBitboard = setBit(this.occupiedBitboard, to);
            
            if (isCapture) {
                this.halfMoveClock = 0;
            } else {
                this.halfMoveClock++;
            }
        }
        
        // Switch turn (except if sideToMove is still NONE, which shouldn't happen after first flip)
        if (this.sideToMove !== Color.NONE) {
            this.sideToMove = 1 - this.sideToMove;
        }
    }

    private getSnapshot(): string {
        return `${this.pieceBitboards.join(',')}|${this.hiddenBitboard}|${this.sideToMove}`;
    }

    public isGameOver(): { over: boolean, result: number } {
        if (this.halfMoveClock >= 50) return { over: true, result: 0.0 }; // Draw
        
        // Check 3-fold repetition
        let repCount = 0;
        const currentSnapshot = this.getSnapshot();
        for (const snap of this.history) {
            if (snap === currentSnapshot) repCount++;
        }
        if (repCount >= 2) return { over: true, result: 0.0 }; // Draw (2 past + 1 current = 3)
        
        // Check if one side is wiped out
        const redAlive = this.hasAlivePieces(Color.RED);
        const blackAlive = this.hasAlivePieces(Color.BLACK);
        const hiddenCount = popcount(this.hiddenBitboard);
        
        if (hiddenCount === 0) {
            if (!redAlive) return { over: true, result: -1.0 }; // Assuming perspective is Red (loss)
            if (!blackAlive) return { over: true, result: 1.0 }; // Assuming perspective is Red (win)
        }
        
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

    private hasAlivePieces(color: Color): boolean {
        for (let i = 0; i < 14; i++) {
            if (PIECE_COLOR[i] === color && this.pieceBitboards[i] > 0) return true;
        }
        return false;
    }
}
