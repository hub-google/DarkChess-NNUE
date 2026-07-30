interface R2Object {
  key: string;
}

interface R2ObjectBody {
  body: ReadableStream<Uint8Array>;
}

interface R2Bucket {
  put(key: string, value: ReadableStream<Uint8Array> | null): Promise<unknown>;
  list(): Promise<{ objects: R2Object[] }>;
  get(key: string): Promise<R2ObjectBody | null>;
}

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}
