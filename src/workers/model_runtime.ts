/** Node transport; validation and decoding are shared with the browser. */
import {createHash} from 'node:crypto'
import {readFileSync} from 'node:fs'
import {join} from 'node:path'
import {decodeNNUE, validateNNUEManifest, type BrowserNNUE} from '../../frontend/src/engine/nnue-core'

export function loadNNUE(base = 'frontend/public/models'): {model: BrowserNNUE; version: string} {
  const metadata = validateNNUEManifest(JSON.parse(readFileSync(join(base,'champion.json'),'utf8')))
  const bytes = readFileSync(join(base,'champion.bin'))
  const hash = createHash('sha256').update(bytes).digest('hex')
  return {version: 'nnue-'+hash.slice(0,12), model:decodeNNUE(metadata,bytes,hash)}
}
