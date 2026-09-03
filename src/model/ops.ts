/**
 * Differentiable ops, each written as an explicit forward/backward pair.
 *
 * There is no autograd tape here on purpose. Explicit backward functions are
 * what a reader can actually follow, they are faster than a tape for a model
 * this small, and they keep every intermediate reachable for the visuals. The
 * cost is that the gradients are hand-derived and therefore fallible, which is
 * exactly why `tests/gradcheck.test.ts` exists.
 */

const GELU_C = Math.sqrt(2 / Math.PI)

/** Numerically safe softmax over a contiguous span, in place. Returns nothing. */
export function softmaxInPlace(x: Float32Array, offset: number, n: number): void {
  let max = -Infinity
  for (let i = 0; i < n; i++) {
    const v = x[offset + i]
    if (v > max) max = v
  }
  // Subtracting the max before exponentiating is what stops `exp` overflowing
  // to Infinity on confident logits. It cancels out of the ratio exactly.
  let sum = 0
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[offset + i] - max)
    x[offset + i] = e
    sum += e
  }
  const inv = 1 / sum
  for (let i = 0; i < n; i++) x[offset + i] *= inv
}

/**
 * Backward through a softmax whose output was `p`.
 *
 * `dx_i = p_i * (dy_i - Σ_j p_j·dy_j)` — the subtracted term is what enforces
 * that the outputs must keep summing to 1, so no gradient can push the whole
 * distribution up or down at once.
 */
export function softmaxBackward(
  dx: Float32Array,
  dy: Float32Array,
  p: Float32Array,
  offset: number,
  n: number,
): void {
  let dot = 0
  for (let i = 0; i < n; i++) dot += p[offset + i] * dy[offset + i]
  for (let i = 0; i < n; i++) {
    dx[offset + i] = p[offset + i] * (dy[offset + i] - dot)
  }
}

/** GELU, tanh approximation — the same one GPT-2 shipped. */
export function gelu(x: number): number {
  const inner = GELU_C * (x + 0.044715 * x * x * x)
  return 0.5 * x * (1 + Math.tanh(inner))
}

export function geluForward(out: Float32Array, x: Float32Array): void {
  for (let i = 0; i < x.length; i++) out[i] = gelu(x[i])
}

/** d/dx of the tanh-approximation GELU, evaluated at the cached pre-activation. */
export function geluBackward(dx: Float32Array, dy: Float32Array, x: Float32Array): void {
  for (let i = 0; i < x.length; i++) {
    const v = x[i]
    const inner = GELU_C * (v + 0.044715 * v * v * v)
    const t = Math.tanh(inner)
    const dInner = GELU_C * (1 + 3 * 0.044715 * v * v)
    dx[i] = dy[i] * (0.5 * (1 + t) + 0.5 * v * (1 - t * t) * dInner)
  }
}

export interface LayerNormCache {
  /** Normalised activations, needed by both the input and the gamma gradient. */
  xhat: Float32Array
  /** Reciprocal standard deviation per row. */
  istd: Float32Array
}

export function makeLayerNormCache(rows: number, d: number): LayerNormCache {
  return { xhat: new Float32Array(rows * d), istd: new Float32Array(rows) }
}

const LN_EPS = 1e-5

/** Row-wise layer normalisation: `y = γ · (x - μ)/σ + β`. */
export function layerNormForward(
  out: Float32Array,
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  rows: number,
  d: number,
  cache: LayerNormCache,
): void {
  for (let r = 0; r < rows; r++) {
    const o = r * d
    let mean = 0
    for (let i = 0; i < d; i++) mean += x[o + i]
    mean /= d
    let varSum = 0
    for (let i = 0; i < d; i++) {
      const c = x[o + i] - mean
      varSum += c * c
    }
    const istd = 1 / Math.sqrt(varSum / d + LN_EPS)
    cache.istd[r] = istd
    for (let i = 0; i < d; i++) {
      const xh = (x[o + i] - mean) * istd
      cache.xhat[o + i] = xh
      out[o + i] = gamma[i] * xh + beta[i]
    }
  }
}

/**
 * Backward through layer norm.
 *
 * The collapsed form `dx = istd·(dxhat - mean(dxhat) - xhat·mean(dxhat·xhat))`
 * is used rather than differentiating mean and variance separately. The two
 * subtracted means are the gradient flowing back through mu and sigma, which is
 * the part that is easy to drop by accident and the reason layer norm is the
 * op most worth gradient-checking.
 */
export function layerNormBackward(
  dx: Float32Array,
  dgamma: Float32Array,
  dbeta: Float32Array,
  dy: Float32Array,
  gamma: Float32Array,
  rows: number,
  d: number,
  cache: LayerNormCache,
): void {
  for (let r = 0; r < rows; r++) {
    const o = r * d
    const istd = cache.istd[r]
    let meanDxhat = 0
    let meanDxhatXhat = 0
    for (let i = 0; i < d; i++) {
      const dxh = dy[o + i] * gamma[i]
      meanDxhat += dxh
      meanDxhatXhat += dxh * cache.xhat[o + i]
    }
    meanDxhat /= d
    meanDxhatXhat /= d
    for (let i = 0; i < d; i++) {
      const dxh = dy[o + i] * gamma[i]
      dx[o + i] = istd * (dxh - meanDxhat - cache.xhat[o + i] * meanDxhatXhat)
      dgamma[i] += dy[o + i] * cache.xhat[o + i]
      dbeta[i] += dy[o + i]
    }
  }
}

/**
 * Fused softmax + cross-entropy over `[rows, vocab]` logits.
 *
 * Fusing them is not just a speed trick: the composed gradient collapses to
 * `(p - onehot)/rows`, with no division by a probability anywhere, so a token
 * the model assigned near-zero probability produces a large but perfectly
 * finite gradient instead of a NaN.
 *
 * Writes probabilities into `probs` and the gradient into `dLogits`.
 * Returns mean loss in nats.
 */
export function softmaxCrossEntropy(
  probs: Float32Array,
  dLogits: Float32Array | null,
  logits: Float32Array,
  targets: Int32Array,
  rows: number,
  vocab: number,
): number {
  let loss = 0
  for (let r = 0; r < rows; r++) {
    const o = r * vocab
    let max = -Infinity
    for (let i = 0; i < vocab; i++) {
      const v = logits[o + i]
      if (v > max) max = v
    }
    let sum = 0
    for (let i = 0; i < vocab; i++) {
      const e = Math.exp(logits[o + i] - max)
      probs[o + i] = e
      sum += e
    }
    const inv = 1 / sum
    for (let i = 0; i < vocab; i++) probs[o + i] *= inv
    const t = targets[r]
    // log p_t recovered from the shifted logits directly, avoiding log(0).
    loss += -(logits[o + t] - max - Math.log(sum))
  }
  if (dLogits) {
    const scale = 1 / rows
    for (let r = 0; r < rows; r++) {
      const o = r * vocab
      for (let i = 0; i < vocab; i++) dLogits[o + i] = probs[o + i] * scale
      dLogits[o + targets[r]] -= scale
    }
  }
  return loss / rows
}
