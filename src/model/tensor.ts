/**
 * Flat Float32Array tensor primitives.
 *
 * Everything in this model is a flat, row-major Float32Array plus a shape. No
 * nested arrays, no objects per element. That is the single biggest reason a
 * transformer can train at interactive speed in a browser tab.
 *
 * The matmuls below use `i-k-j` loop order rather than the textbook `i-j-k`.
 * Both compute the same thing, but `i-k-j` walks both `b` and `out` forward
 * through memory in the innermost loop, so the CPU prefetcher stays useful and
 * the hot row of `a` collapses to a scalar. In practice that is worth roughly
 * 3-5x over the naive order, which is the difference between a 40-second train
 * and a three-minute one.
 */

/** Row-major matrix: `A[M,K] · B[K,N] → C[M,N]`. */
export function matmul(
  out: Float32Array,
  a: Float32Array,
  b: Float32Array,
  M: number,
  K: number,
  N: number,
): void {
  out.fill(0)
  for (let i = 0; i < M; i++) {
    const aRow = i * K
    const oRow = i * N
    for (let k = 0; k < K; k++) {
      const aik = a[aRow + k]
      if (aik === 0) continue
      const bRow = k * N
      for (let j = 0; j < N; j++) {
        out[oRow + j] += aik * b[bRow + j]
      }
    }
  }
}

/**
 * `A[M,K] · B[N,K]ᵀ → C[M,N]`, with B stored row-major as `[N,K]`.
 *
 * This is the shape a backward pass wants for `dx = dy · Wᵀ`: the weight stays
 * in its forward `[in,out]` layout and never needs an explicit transpose.
 */
export function matmulNT(
  out: Float32Array,
  a: Float32Array,
  b: Float32Array,
  M: number,
  K: number,
  N: number,
): void {
  for (let i = 0; i < M; i++) {
    const aRow = i * K
    const oRow = i * N
    for (let j = 0; j < N; j++) {
      const bRow = j * K
      let sum = 0
      for (let k = 0; k < K; k++) sum += a[aRow + k] * b[bRow + k]
      out[oRow + j] = sum
    }
  }
}

/**
 * `A[M,K]ᵀ · B[M,N] → C[K,N]`, accumulating into `out`.
 *
 * This is the weight-gradient shape (`dW = xᵀ · dy`). It accumulates rather
 * than overwrites because gradients sum over the batch.
 */
export function matmulTN_acc(
  out: Float32Array,
  a: Float32Array,
  b: Float32Array,
  M: number,
  K: number,
  N: number,
): void {
  for (let m = 0; m < M; m++) {
    const aRow = m * K
    const bRow = m * N
    for (let k = 0; k < K; k++) {
      const amk = a[aRow + k]
      if (amk === 0) continue
      const oRow = k * N
      for (let n = 0; n < N; n++) {
        out[oRow + n] += amk * b[bRow + n]
      }
    }
  }
}

/** Add a `[N]` bias to every row of a `[M,N]` matrix, in place. */
export function addBias(x: Float32Array, bias: Float32Array, M: number, N: number): void {
  for (let i = 0; i < M; i++) {
    const row = i * N
    for (let j = 0; j < N; j++) x[row + j] += bias[j]
  }
}

/** Sum a `[M,N]` gradient down its rows into a `[N]` bias gradient. */
export function sumRows_acc(out: Float32Array, dy: Float32Array, M: number, N: number): void {
  for (let i = 0; i < M; i++) {
    const row = i * N
    for (let j = 0; j < N; j++) out[j] += dy[row + j]
  }
}

/** Elementwise `a += b`. */
export function addInto(a: Float32Array, b: Float32Array): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i]
}

/** Box-Muller normal sampler with an injectable uniform source, so init is reproducible. */
export function randn(rng: () => number): number {
  let u = 0
  let v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

/** Deterministic, seedable PRNG (mulberry32) so a given seed always trains identically. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
