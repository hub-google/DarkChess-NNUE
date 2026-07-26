import { DarkChessBoard } from '../engine/board';
import { Color, MoveId } from '../engine/types';

// Simple Greedy Heuristic for initial data generation
function getHeuristicMove(board: DarkChessBoard, moves: MoveId[]): MoveId {
    // Separate into captures, moves, and flips
    const captures: MoveId[] = [];
    const flips: MoveId[] = [];
    const normals: MoveId[] = [];

    for (const m of moves) {
        const isFlip = (m >> 5) === (m & 31);
        if (isFlip) {
            flips.push(m);
        } else {
            const to = m & 31;
            // Check if it's a capture
            const isCapture = ((board.occupiedBitboard >> to) & 1) !== 0;
            if (isCapture) captures.push(m);
            else normals.push(m);
        }
    }

    // 1. Prefer captures (greedy)
    if (captures.length > 0) {
        return captures[Math.floor(Math.random() * captures.length)];
    }

    // 2. Otherwise flip or move
    const remaining = [...flips, ...normals];
    return remaining[Math.floor(Math.random() * remaining.length)];
}

import { packGames, uploadToCloudflareR2 } from './uploader';

export async function runSelfPlayBatch(batchSize = 50) {
    const games = [];
    const batchId = crypto.randomUUID();

    for (let i = 0; i < batchSize; i++) {
        const board = new DarkChessBoard();
        const gameRecord: any = {
            id: crypto.randomUUID(),
            ver: "v1.0.0-heuristic",
            hid: [...board.hiddenPieces],
            mov: [],
            res: 0.0,
            ply: 0
        };

        while (true) {
            const { over, result } = board.isGameOver();
            if (over) {
                gameRecord.res = result; // 1.0 (Red win), -1.0 (Black win), 0.0 (Draw)
                break;
            }

            const moves = board.generateLegalMoves();
            const chosenMove = getHeuristicMove(board, moves);
            
            gameRecord.mov.push(chosenMove);
            board.makeMove(chosenMove);
            gameRecord.ply++;
        }
        
        games.push(gameRecord);
    }

    console.log(`[Self-Play] Batch ${batchId} generated. Compressing...`);
    const compressedData = packGames(games);
    
    console.log(`[Self-Play] Uploading to R2...`);
    const success = await uploadToCloudflareR2(compressedData, batchId);
    
    if (success) {
        console.log(`[Self-Play] Upload complete!`);
    } else {
        console.error(`[Self-Play] Upload failed! (Retries can be implemented here)`);
    }

    return success;
}
