import { expect, test, vi } from 'vitest';
import { packGames, uploadToCloudflareR2 } from '../src/workers/uploader';

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

test('uploadToCloudflareR2 should send PUT request with gzip payload', async () => {
    // Mock the global fetch
    global.fetch = vi.fn().mockResolvedValue({
        ok: true
    } as Response);

    const payload = new Uint8Array([0x1F, 0x8B, 0x08, 0x00]);
    const success = await uploadToCloudflareR2(payload, "batch-123");

    expect(success).toBe(true);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    
    const callArgs = vi.mocked(global.fetch).mock.calls[0];
    expect(callArgs[0]).toContain("batch-123.jsonl.gz");
    expect(callArgs[1]?.method).toBe("PUT");
    expect(callArgs[1]?.headers).toEqual({ 'Content-Encoding': 'gzip' });
    expect(callArgs[1]?.body).toBe(payload);
});

test('uploadToCloudflareR2 should handle errors gracefully', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("Network failure"));

    const payload = new Uint8Array([0x1F, 0x8B]);
    const success = await uploadToCloudflareR2(payload, "batch-error");

    expect(success).toBe(false);
});
