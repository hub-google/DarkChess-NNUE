import { MoveId, decodeMove, Piece, PieceType, PIECE_TYPE, PIECE_COLOR, Color } from './types';
import { DarkChessBoard } from './board';
import { hasBit } from './bitboard';

const PIECE_NAMES_RED = ["帥", "仕", "相", "俥", "傌", "炮", "兵"];
const PIECE_NAMES_BLACK = ["將", "士", "象", "車", "馬", "包", "卒"];

export function getPieceName(piece: Piece): string {
    if (piece === Piece.EMPTY || piece === Piece.HIDDEN) return "未知";
    const type = PIECE_TYPE[piece];
    if (PIECE_COLOR[piece] === Color.RED) return PIECE_NAMES_RED[type];
    return PIECE_NAMES_BLACK[type];
}

export function formatSquare(sq: number): string {
    const x = sq % 8;
    const y = Math.floor(sq / 8);
    return `${x},${y}`;
}

export function decodeMoveToHumanReadable(move: MoveId, boardBeforeMove: DarkChessBoard): string {
    const { from, to, isFlip } = decodeMove(move);
    
    if (isFlip) {
        // Find what was flipped by looking at the board AFTER the move (or we need the piece info)
        // Since we only have boardBeforeMove, we can't know the exact piece unless we simulate it or pass the flipped piece.
        // For the sake of standard output: "翻 [座標X,Y] > 出現 [顏色][棋子]"
        // We will just return the generic format. The caller should append " > 出現 [顏色][棋子]".
        return `翻 ${formatSquare(from)}`; 
    }

    // Find the moving piece
    let movingPiece = Piece.EMPTY;
    for (let i = 0; i < 14; i++) {
        if (hasBit(boardBeforeMove.pieceBitboards[i], from)) {
            movingPiece = i;
            break;
        }
    }
    
    // Find victim piece if capture
    let victimPiece = Piece.EMPTY;
    if (hasBit(boardBeforeMove.occupiedBitboard, to)) {
        for (let i = 0; i < 14; i++) {
            if (hasBit(boardBeforeMove.pieceBitboards[i], to)) {
                victimPiece = i;
                break;
            }
        }
    }
    
    const pName = getPieceName(movingPiece);
    const sqFrom = formatSquare(from);
    const sqTo = formatSquare(to);

    if (victimPiece !== Piece.EMPTY) {
        const vName = getPieceName(victimPiece);
        return `${pName} ${sqFrom} > ${sqTo} ${pName}吃${vName}`;
    } else {
        return `${pName} ${sqFrom} > ${sqTo} ${pName}移動`;
    }
}
