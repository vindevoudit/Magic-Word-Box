/**
 * Multi-head causal self-attention.
 *
 * This file is the one the site puts on screen in the final stage, so it is
 * written to be read: explicit loops over batch, head and position, named
 * intermediates, and no clever flattening that would hide the mechanism.
 *
 * The whole operation, in one sentence: every token builds a query, compares it
 * against the key of each token at or before it, turns those comparisons into a
 * distribution that sums to 1, and takes that weighted average of the values of
 * the tokens it was allowed to see.
 */

import { addBias, matmul, matmulNT, matmulTN_acc, sumRows_acc } from './tensor'
import { softmaxInPlace } from './ops'

export interface AttentionWeights {
  /** Query/key/value/output projections, each [d, d] row-major. */
  wq: Float32Array
  wk: Float32Array
  wv: Float32Array
  wo: Float32Array
  bq: Float32Array
  bk: Float32Array
  bv: Float32Array
  bo: Float32Array
}

export interface AttentionCache {
  x: Float32Array
  q: Float32Array
  k: Float32Array
  v: Float32Array
  /** Post-softmax attention weights, [B, H, T, T]. Also what the heatmaps draw. */
  probs: Float32Array
  /** Concatenated head outputs before the output projection, [B*T, d]. */
  ctx: Float32Array
}

export function makeAttentionCache(B: number, T: number, d: number, H: number): AttentionCache {
  return {
    x: new Float32Array(B * T * d),
    q: new Float32Array(B * T * d),
    k: new Float32Array(B * T * d),
    v: new Float32Array(B * T * d),
    probs: new Float32Array(B * H * T * T),
    ctx: new Float32Array(B * T * d),
  }
}

export interface AttentionDims {
  B: number
  T: number
  d: number
  H: number
}

/**
 * Forward pass. x is [B*T, d], out is [B*T, d].
 *
 * cache is filled for the backward pass and for visualisation; cache.probs is
 * the tensor every attention picture on the site is drawn from.
 */
export function attentionForward(
  out: Float32Array,
  x: Float32Array,
  w: AttentionWeights,
  dims: AttentionDims,
  cache: AttentionCache,
  scratchScores: Float32Array,
): void {
  const { B, T, d, H } = dims
  const rows = B * T
  const headDim = d / H
  // Scaling by 1/sqrt(headDim) keeps the dot products from growing with
  // dimension. Without it, wide heads produce huge scores, the softmax
  // saturates into a one-hot, and the gradient through it vanishes.
  const scale = 1 / Math.sqrt(headDim)

  cache.x.set(x)
  matmul(cache.q, x, w.wq, rows, d, d)
  matmul(cache.k, x, w.wk, rows, d, d)
  matmul(cache.v, x, w.wv, rows, d, d)
  addBias(cache.q, w.bq, rows, d)
  addBias(cache.k, w.bk, rows, d)
  addBias(cache.v, w.bv, rows, d)

  cache.ctx.fill(0)

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const headOff = h * headDim
      const probsBase = b * H * T * T + h * T * T

      for (let t1 = 0; t1 < T; t1++) {
        const qOff = (b * T + t1) * d + headOff
        const scoreRow = t1 * T

        // Causal mask: only positions 0..t1 are ever scored. Rather than
        // computing the full square and adding -Infinity above the diagonal,
        // the loop simply stops at t1. Same result, half the work, and the
        // triangle is visible in the shape of the code itself.
        for (let t2 = 0; t2 <= t1; t2++) {
          const kOff = (b * T + t2) * d + headOff
          let dot = 0
          for (let i = 0; i < headDim; i++) dot += cache.q[qOff + i] * cache.k[kOff + i]
          scratchScores[scoreRow + t2] = dot * scale
        }

        softmaxInPlace(scratchScores, scoreRow, t1 + 1)

        // Positions after t1 keep weight exactly 0, which the tests assert.
        for (let t2 = 0; t2 <= t1; t2++) {
          cache.probs[probsBase + scoreRow + t2] = scratchScores[scoreRow + t2]
        }
        for (let t2 = t1 + 1; t2 < T; t2++) {
          cache.probs[probsBase + scoreRow + t2] = 0
        }

        // Weighted average of the values of every token we were allowed to see.
        const ctxOff = (b * T + t1) * d + headOff
        for (let t2 = 0; t2 <= t1; t2++) {
          const p = scratchScores[scoreRow + t2]
          if (p === 0) continue
          const vOff = (b * T + t2) * d + headOff
          for (let i = 0; i < headDim; i++) {
            cache.ctx[ctxOff + i] += p * cache.v[vOff + i]
          }
        }
      }
    }
  }

  matmul(out, cache.ctx, w.wo, rows, d, d)
  addBias(out, w.bo, rows, d)
}

export interface AttentionGrads {
  wq: Float32Array
  wk: Float32Array
  wv: Float32Array
  wo: Float32Array
  bq: Float32Array
  bk: Float32Array
  bv: Float32Array
  bo: Float32Array
}

export interface AttentionBackScratch {
  dq: Float32Array
  dk: Float32Array
  dv: Float32Array
  dctx: Float32Array
  dxPart: Float32Array
  dprobsRow: Float32Array
  dscoresRow: Float32Array
}

export function makeAttentionBackScratch(
  B: number,
  T: number,
  d: number,
): AttentionBackScratch {
  return {
    dq: new Float32Array(B * T * d),
    dk: new Float32Array(B * T * d),
    dv: new Float32Array(B * T * d),
    dctx: new Float32Array(B * T * d),
    dxPart: new Float32Array(B * T * d),
    dprobsRow: new Float32Array(T),
    dscoresRow: new Float32Array(T),
  }
}

/** Backward pass. dOut is [B*T, d]; accumulates into g and writes dx. */
export function attentionBackward(
  dx: Float32Array,
  dOut: Float32Array,
  w: AttentionWeights,
  g: AttentionGrads,
  dims: AttentionDims,
  cache: AttentionCache,
  s: AttentionBackScratch,
): void {
  const { B, T, d, H } = dims
  const rows = B * T
  const headDim = d / H
  const scale = 1 / Math.sqrt(headDim)

  // --- output projection ---
  matmulNT(s.dctx, dOut, w.wo, rows, d, d)
  matmulTN_acc(g.wo, cache.ctx, dOut, rows, d, d)
  sumRows_acc(g.bo, dOut, rows, d)

  s.dq.fill(0)
  s.dk.fill(0)
  s.dv.fill(0)

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const headOff = h * headDim
      const probsBase = b * H * T * T + h * T * T

      for (let t1 = 0; t1 < T; t1++) {
        const n = t1 + 1
        const scoreRow = t1 * T
        const dctxOff = (b * T + t1) * d + headOff

        // Two things fall out of the same weighted sum: how much each earlier
        // token was attended to (dprobs), and how much its value vector should
        // change (dv).
        for (let t2 = 0; t2 < n; t2++) {
          const vOff = (b * T + t2) * d + headOff
          let dot = 0
          for (let i = 0; i < headDim; i++) dot += s.dctx[dctxOff + i] * cache.v[vOff + i]
          s.dprobsRow[t2] = dot

          const p = cache.probs[probsBase + scoreRow + t2]
          if (p !== 0) {
            for (let i = 0; i < headDim; i++) {
              s.dv[vOff + i] += p * s.dctx[dctxOff + i]
            }
          }
        }

        // Softmax backward over just the unmasked prefix, folding in the scale.
        let dot = 0
        for (let t2 = 0; t2 < n; t2++) {
          dot += cache.probs[probsBase + scoreRow + t2] * s.dprobsRow[t2]
        }
        for (let t2 = 0; t2 < n; t2++) {
          const p = cache.probs[probsBase + scoreRow + t2]
          s.dscoresRow[t2] = p * (s.dprobsRow[t2] - dot) * scale
        }

        // A score is a dot product of two vectors, so each one picks up the
        // other as its gradient.
        const qOff = (b * T + t1) * d + headOff
        for (let t2 = 0; t2 < n; t2++) {
          const ds = s.dscoresRow[t2]
          if (ds === 0) continue
          const kOff = (b * T + t2) * d + headOff
          for (let i = 0; i < headDim; i++) {
            s.dq[qOff + i] += ds * cache.k[kOff + i]
            s.dk[kOff + i] += ds * cache.q[qOff + i]
          }
        }
      }
    }
  }

  // --- q/k/v projections ---
  matmulTN_acc(g.wq, cache.x, s.dq, rows, d, d)
  matmulTN_acc(g.wk, cache.x, s.dk, rows, d, d)
  matmulTN_acc(g.wv, cache.x, s.dv, rows, d, d)
  sumRows_acc(g.bq, s.dq, rows, d)
  sumRows_acc(g.bk, s.dk, rows, d)
  sumRows_acc(g.bv, s.dv, rows, d)

  // All three projections read the same x, so dx is the sum of three paths.
  matmulNT(dx, s.dq, w.wq, rows, d, d)
  matmulNT(s.dxPart, s.dk, w.wk, rows, d, d)
  for (let i = 0; i < dx.length; i++) dx[i] += s.dxPart[i]
  matmulNT(s.dxPart, s.dv, w.wv, rows, d, d)
  for (let i = 0; i < dx.length; i++) dx[i] += s.dxPart[i]
}
