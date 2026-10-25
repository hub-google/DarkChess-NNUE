import { Board } from './Board';
import { Color, Move, MoveType, makeMove, PieceType, getPieceType } from './types';
import { testBit, getAdjacent } from './Bitboard';

/**
 * 判斷 attacker 是否可以吃掉 victim（不包含炮的隔山打牛邏輯，炮的跳吃會在 MoveGenerator 獨立處理）
 * 規則：
 * 1. 炮(2) 可以吃任何子（因為能進來這裡代表已經跳過了）
 * 2. 帥(7) 不能吃 兵(1)
 * 3. 兵(1) 可以吃 帥(7)
 * 4. 其他情況：階級大於等於對方即可吃
 */
export const canEat = (attacker: PieceType, victim: PieceType): boolean => {
  if (attacker === PieceType.Cannon) return true;
  if (attacker === PieceType.King && victim === PieceType.Pawn) return false;
  if (attacker === PieceType.Pawn && victim === PieceType.King) return true;
  return attacker >= victim;
};

export class MoveGenerator {
  /**
   * 根據當前盤面與輪到哪一方，生成所有合法的步數 (包含翻牌、移動、吃子)
   */
  public static generateMoves(board: Board): Move[] {
    const moves: Move[] = [];
    const turn = board.turn;

    // 1. 翻牌 (Flip moves)
    // 任何有棋子但在「未翻開」狀態的格子，都可以翻開。翻開視同消耗一回合。
    const unrevealed = board.occupied & ~board.revealed;
    for (let sq = 0; sq < 32; sq++) {
      if (testBit(unrevealed, sq)) {
        // 翻牌的 move 將 from 和 to 設為同一個位置，型態為 Flip
        moves.push(makeMove(sq, sq, MoveType.Flip));
      }
    }

    // 如果當前還沒決定輪到誰（例如開局第一步前），那就只能翻牌
    if (turn === Color.None) {
      return moves;
    }

    // 2. 移動與吃子 (Normal moves and Captures)
    const myPiecesBB = board.revealed & board.colorBB[turn];
    const enemyPiecesBB = board.revealed & board.colorBB[1 - turn];
    const emptyBB = ~board.occupied;

    for (let sq = 0; sq < 32; sq++) {
      if (testBit(myPiecesBB, sq)) {
        const piece = board.pieces[sq];
        const type = getPieceType(piece);

        if (type === PieceType.Cannon) {
          // 【炮】的特殊移動與吃子
          // a. 炮可以走一步到相鄰空格
          const adj = getAdjacent(sq);
          const emptyAdj = adj & emptyBB;
          for (let target = 0; target < 32; target++) {
            if (testBit(emptyAdj, target)) {
              moves.push(makeMove(sq, target, MoveType.Move));
            }
          }
          // b. 炮的跳吃 (隔山打牛)
          MoveGenerator.generateCannonCaptures(board, sq, enemyPiecesBB, moves);
        } else {
          // 【其他棋子】(帥仕相車馬卒)
          const adj = getAdjacent(sq);

          // a. 移動到相鄰空格
          const emptyAdj = adj & emptyBB;
          for (let target = 0; target < 32; target++) {
            if (testBit(emptyAdj, target)) {
              moves.push(makeMove(sq, target, MoveType.Move));
            }
          }

          // b. 吃掉相鄰的敵方棋子
          const enemyAdj = adj & enemyPiecesBB;
          for (let target = 0; target < 32; target++) {
            if (testBit(enemyAdj, target)) {
              const targetPiece = board.pieces[target];
              const targetType = getPieceType(targetPiece);
              if (canEat(type, targetType)) {
                moves.push(makeMove(sq, target, MoveType.Capture));
              }
            }
          }
        }
      }
    }

    return moves;
  }

  /**
   * 專門處理炮的「隔山打牛」跳吃邏輯
   */
  private static generateCannonCaptures(board: Board, sq: number, enemyPiecesBB: number, moves: Move[]) {
    // 將一維 index 轉為 4x8 二維座標
    const r = Math.floor(sq / 8);
    const c = sq % 8;

    // 向右找
    let mountFound = false;
    for (let tc = c + 1; tc < 8; tc++) {
      const targetSq = r * 8 + tc;
      if (testBit(board.occupied, targetSq)) {
        if (!mountFound) {
          mountFound = true; // 找到炮座了
        } else {
          // 找到第二個棋子，若是翻開的敵軍即可吃
          if (testBit(enemyPiecesBB, targetSq)) {
            moves.push(makeMove(sq, targetSq, MoveType.Capture));
          }
          break; // 不論是誰，都不能跳過兩個，方向中斷
        }
      }
    }

    // 向左找
    mountFound = false;
    for (let tc = c - 1; tc >= 0; tc--) {
      const targetSq = r * 8 + tc;
      if (testBit(board.occupied, targetSq)) {
        if (!mountFound) {
          mountFound = true;
        } else {
          if (testBit(enemyPiecesBB, targetSq)) {
            moves.push(makeMove(sq, targetSq, MoveType.Capture));
          }
          break;
        }
      }
    }

    // 向下找
    mountFound = false;
    for (let tr = r + 1; tr < 4; tr++) {
      const targetSq = tr * 8 + c;
      if (testBit(board.occupied, targetSq)) {
        if (!mountFound) {
          mountFound = true;
        } else {
          if (testBit(enemyPiecesBB, targetSq)) {
            moves.push(makeMove(sq, targetSq, MoveType.Capture));
          }
          break;
        }
      }
    }

    // 向上找
    mountFound = false;
    for (let tr = r - 1; tr >= 0; tr--) {
      const targetSq = tr * 8 + c;
      if (testBit(board.occupied, targetSq)) {
        if (!mountFound) {
          mountFound = true;
        } else {
          if (testBit(enemyPiecesBB, targetSq)) {
            moves.push(makeMove(sq, targetSq, MoveType.Capture));
          }
          break;
        }
      }
    }
  }
}
