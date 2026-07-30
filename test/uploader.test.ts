import { afterEach, expect, test } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { packGames, writeToLocalDisk } from '../src/workers/uploader';

const tempDirs: string[] = [];

afterEach(() => {
    for (const dir of tempDirs.splice(0)) {
        fs.rmSync(dir, { recursive: true, force: true });
    }
});

test('packGames should compress JSONL array into gzip format', () => {
    const mockGames = [
        { id: "1", ply: 10, res: 1.0 },
        { id: "2", ply: 20, res: -1.0 }
    ];

    const compressed = packGames(mockGames);
    
    // It should return a Uint8Array
    expect(compressed).toBeInstanceOf(Uint8Array);
    
    // Minimal gzip header check (magic number: 1F 8B)
    expect(compressed[0]).toBe(0x1F);
    expect(compressed[1]).toBe(0x8B);
});

test('writeToLocalDisk should save a gzip batch', () => {
    const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), 'darkchess-uploader-'));
    tempDirs.push(outputDir);
    const payload = new Uint8Array([0x1F, 0x8B, 0x08, 0x00]);
    const success = writeToLocalDisk(payload, 'batch-123', outputDir);
    expect(success).toBe(true);
    const files = fs.readdirSync(outputDir);
    expect(files).toHaveLength(1);
    expect(files[0]).toMatch(/^data_\d+_batch-123\.jsonl\.gz$/);
    expect(fs.readFileSync(path.join(outputDir, files[0]))).toEqual(Buffer.from(payload));
});

test('writeToLocalDisk should handle an invalid output path gracefully', () => {
    const parent = fs.mkdtempSync(path.join(os.tmpdir(), 'darkchess-uploader-'));
    tempDirs.push(parent);
    const filePath = path.join(parent, 'not-a-directory');
    fs.writeFileSync(filePath, 'occupied');
    const payload = new Uint8Array([0x1F, 0x8B]);
    const success = writeToLocalDisk(payload, 'batch-error', filePath);
    expect(success).toBe(false);
});
