/**
 * int8 weight serialisation for the model shipped with the site.
 *
 * The pretrained model exists so every visualisation is live before the visitor
 * has trained anything. It only needs to be good enough to draw, so full float32
 * precision is wasted bytes: per-tensor int8 cuts the payload 4x and the
 * resulting predictions are visually identical.
 */

import type { ModelConfig, Param } from './model'

export interface SerializedModel {
  version: 1
  config: ModelConfig
  vocab: string[]
  tensors: { name: string; scale: number; b64: string }[]
}

function toBase64(bytes: Int8Array): string {
  const u8 = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let binary = ''
  const CHUNK = 0x8000
  for (let i = 0; i < u8.length; i += CHUNK) {
    binary += String.fromCharCode(...u8.subarray(i, i + CHUNK))
  }
  return btoa(binary)
}

function fromBase64(b64: string): Int8Array {
  const binary = atob(b64)
  const u8 = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) u8[i] = binary.charCodeAt(i)
  return new Int8Array(u8.buffer)
}

export function serializeModel(
  cfg: ModelConfig,
  params: Param[],
  vocab: string[],
): SerializedModel {
  return {
    version: 1,
    config: cfg,
    vocab,
    tensors: params.map((p) => {
      let max = 0
      for (let i = 0; i < p.data.length; i++) {
        const a = Math.abs(p.data[i])
        if (a > max) max = a
      }
      // One scale per tensor. Layer-norm gains sit near 1 and attention
      // weights near 0.02, so a single global scale would crush the latter.
      const scale = max > 0 ? max / 127 : 1
      const q = new Int8Array(p.data.length)
      for (let i = 0; i < p.data.length; i++) {
        q[i] = Math.max(-127, Math.min(127, Math.round(p.data[i] / scale)))
      }
      return { name: p.name, scale, b64: toBase64(q) }
    }),
  }
}

/** Load serialised weights into an existing model with a matching config. */
export function loadInto(params: Param[], serialized: SerializedModel): void {
  const byName = new Map(serialized.tensors.map((t) => [t.name, t]))
  for (const p of params) {
    const t = byName.get(p.name)
    if (!t) throw new Error(`missing tensor in saved model: ${p.name}`)
    const q = fromBase64(t.b64)
    if (q.length !== p.data.length) {
      throw new Error(
        `size mismatch for ${p.name}: saved ${q.length}, expected ${p.data.length}`,
      )
    }
    for (let i = 0; i < q.length; i++) p.data[i] = q[i] * t.scale
  }
}
