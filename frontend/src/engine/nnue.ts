import { BrowserNNUE, type TensorInfo } from './nnue-core'
export { BrowserNNUE, extractFeatures } from './nnue-core'

type Metadata = { format: number; inputSize: number; bytes: number; sha256: string; tensors: Record<string, TensorInfo> }
const tensor = (data: ArrayBuffer, info: TensorInfo) => new Float32Array(data, info.offset, info.length)

export async function loadChampion() {
  const base = `${import.meta.env.BASE_URL}models/`
  // Read the manifest first: the content-addressed binary URL cannot serve
  // a stale copy from an older Pages deployment or intermediate CDN cache.
  const metadataResponse = await fetch(`${base}champion.json`, { cache: 'no-store' })
  if (!metadataResponse.ok) throw new Error('Champion manifest is unavailable')
  const metadata = await metadataResponse.json() as Metadata
  if (metadata.format !== 1 || metadata.inputSize !== 498 ||
      !/^[0-9a-f]{64}$/.test(metadata.sha256)) {
    throw new Error('Unsupported champion manifest')
  }
  const binaryResponse = await fetch(`${base}champion.bin?v=${metadata.sha256}`, { cache: 'no-store' })
  if (!binaryResponse.ok) throw new Error('Champion model is unavailable')
  const data = await binaryResponse.arrayBuffer()
  if (data.byteLength !== metadata.bytes) throw new Error('Champion binary size mismatch')
  const digest = await crypto.subtle.digest('SHA-256', data)
  const actualHash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
  if (actualHash !== metadata.sha256) throw new Error('Champion model checksum mismatch')
  const t = metadata.tensors
  return new BrowserNNUE(
    metadata.sha256.slice(0, 8),
    tensor(data, t['fc1.weight']), tensor(data, t['fc1.bias']),
    tensor(data, t['fc2.weight']), tensor(data, t['fc2.bias']),
    tensor(data, t['fc3.weight']), tensor(data, t['fc3.bias']),
  )
}
