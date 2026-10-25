/** Portable binary NNUE loader, shared by match validation and self-play.
 * The binary is exported by src/training/export_nnue.py.
 */
import {createHash} from 'node:crypto'
import {readFileSync} from 'node:fs'
import {join} from 'node:path'
import {BrowserNNUE, type TensorInfo} from '../../frontend/src/engine/nnue-core'

type Manifest = {format: number; inputSize: number; bytes: number; sha256: string; tensors: Record<string, TensorInfo>}

export function loadNNUE(base = 'frontend/public/models'): {model: BrowserNNUE; version: string} {
  const metadata = JSON.parse(readFileSync(join(base, 'champion.json'), 'utf8')) as Manifest
  const bytes = readFileSync(join(base, 'champion.bin'))
  const hash = createHash('sha256').update(bytes).digest('hex')
  if (metadata.format !== 1 || metadata.inputSize !== 498 ||
      metadata.bytes !== bytes.length || metadata.sha256 !== hash) {
    throw new Error('Invalid/stale NNUE export at ' + base)
  }
  const tensor = (key: string): Float32Array => {
    const info = metadata.tensors[key]
    if (!info || info.offset < 0 || info.length <= 0 ||
        info.offset + 4 * info.length > bytes.length) throw new Error('Invalid tensor ' + key)
    const safe = new ArrayBuffer(4 * info.length)
    new Uint8Array(safe).set(bytes.subarray(info.offset, info.offset + info.length * 4))
    return new Float32Array(safe)
  }
  return {version: 'nnue-' + hash.slice(0, 12), model: new BrowserNNUE(hash.slice(0, 12),
    tensor('fc1.weight'), tensor('fc1.bias'),
    tensor('fc2.weight'), tensor('fc2.bias'),
    tensor('fc3.weight'), tensor('fc3.bias'))}
}
