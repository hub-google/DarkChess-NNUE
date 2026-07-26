import * as pako from 'pako';

// Replace this with the actual Cloudflare Worker URL or environment variable later
const CLOUDFLARE_WORKER_URL = "https://darkchess-nnue-worker.yourdomain.workers.dev"; 

export function packGames(games: any[]): Uint8Array {
    // 1. Convert array of game objects to JSONL string
    const jsonlString = games.map(g => JSON.stringify(g)).join('\n');
    
    // 2. Compress the string into GZIP
    return pako.gzip(jsonlString);
}

export async function uploadToCloudflareR2(compressedData: Uint8Array, batchId: string): Promise<boolean> {
    try {
        const response = await fetch(`${CLOUDFLARE_WORKER_URL}/upload/${batchId}.jsonl.gz`, {
            method: 'PUT',
            headers: {
                'Content-Encoding': 'gzip'
            },
            body: compressedData
        });
        
        return response.ok;
    } catch (error) {
        console.error("Failed to upload batch to R2:", error);
        return false;
    }
}
