import * as pako from 'pako';
import * as fs from 'fs';
import * as path from 'path';

export function packGames(games: any[]): Uint8Array {
    // 1. Convert array of game objects to JSONL string
    const jsonlString = games.map(g => JSON.stringify(g)).join('\n');
    
    // 2. Compress the string into GZIP
    return pako.gzip(jsonlString);
}

export function writeToLocalDisk(compressedData: Uint8Array, batchId: string, outputDir: string): boolean {
    try {
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        const filePath = path.join(outputDir, `data_${Date.now()}_${batchId}.jsonl.gz`);
        fs.writeFileSync(filePath, compressedData);
        console.log(`[Uploader] Wrote batch to ${filePath}`);
        return true;
    } catch (error) {
        console.error("[Uploader] Failed to write batch to disk:", error);
        return false;
    }
}
