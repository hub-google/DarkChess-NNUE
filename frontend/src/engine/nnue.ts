import { decodeNNUE, validateNNUEManifest } from './nnue-core'
export { BrowserNNUE, extractFeatures } from './nnue-core'

export async function loadChampionBytes() {
  const base = `${import.meta.env.BASE_URL}models/`
  const metadataResponse = await fetch(`${base}champion.json`, { cache: 'no-store' })
  if (!metadataResponse.ok) throw new Error('Champion manifest is unavailable')
  const metadata = validateNNUEManifest(await metadataResponse.json())
  const binaryResponse = await fetch(`${base}champion.bin?v=${metadata.sha256}`, { cache: 'no-store' })
  if (!binaryResponse.ok) throw new Error('Champion model is unavailable')
  const data = await binaryResponse.arrayBuffer()
  const digest = await crypto.subtle.digest('SHA-256', data)
  const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2,'0')).join('')
  if (hash !== metadata.sha256 || data.byteLength !== metadata.bytes) throw new Error('Champion binary checksum/size mismatch')
  return {metadata, bytes: new Uint8Array(data), hash}
}

export async function loadChampion() {
  const {metadata, bytes, hash} = await loadChampionBytes()
  return decodeNNUE(metadata,bytes,hash)
}
