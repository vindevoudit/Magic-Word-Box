/**
 * Two-dimensional PCA of the token embedding matrix.
 *
 * This is what stage 4 animates: as training proceeds, words that predict
 * similar continuations drift together. Computed by power iteration with
 * deflation, which for a [600, 64] matrix is far cheaper than a full SVD and
 * plenty accurate for a scatter plot.
 */

export interface Basis {
  /** Two principal directions, each `d` long. */
  axes: [Float32Array, Float32Array]
  mean: Float32Array
}

function normalize(v: Float32Array): void {
  let n = 0
  for (let i = 0; i < v.length; i++) n += v[i] * v[i]
  n = Math.sqrt(n) || 1
  for (let i = 0; i < v.length; i++) v[i] /= n
}

/** One power-iteration pass against the covariance of the centred rows. */
function topDirection(
  rows: Float32Array,
  n: number,
  d: number,
  mean: Float32Array,
  deflate: Float32Array | null,
  seed: number,
): Float32Array {
  const v = new Float32Array(d)
  for (let i = 0; i < d; i++) v[i] = Math.sin(seed * (i + 1) * 12.9898) * 43758.5453 % 1
  normalize(v)

  const next = new Float32Array(d)
  const centred = new Float32Array(d)

  for (let iter = 0; iter < 40; iter++) {
    next.fill(0)
    for (let r = 0; r < n; r++) {
      const off = r * d
      let dot = 0
      for (let i = 0; i < d; i++) {
        const c = rows[off + i] - mean[i]
        centred[i] = c
        dot += c * v[i]
      }
      for (let i = 0; i < d; i++) next[i] += dot * centred[i]
    }
    if (deflate) {
      // Remove the component along the first axis so we converge to the second.
      let dot = 0
      for (let i = 0; i < d; i++) dot += next[i] * deflate[i]
      for (let i = 0; i < d; i++) next[i] -= dot * deflate[i]
    }
    normalize(next)
    v.set(next)
  }
  return v
}

/**
 * Project `rows` ([n, d]) onto two principal axes.
 *
 * `previous` keeps the picture stable across training frames: eigenvectors are
 * only defined up to sign, so without this the scatter mirrors itself at random
 * between updates and the drift becomes impossible to read.
 */
export function pca2d(
  rows: Float32Array,
  n: number,
  d: number,
  previous?: Basis | null,
): { coords: Float32Array; basis: Basis } {
  const mean = new Float32Array(d)
  for (let r = 0; r < n; r++) {
    const off = r * d
    for (let i = 0; i < d; i++) mean[i] += rows[off + i]
  }
  for (let i = 0; i < d; i++) mean[i] /= Math.max(1, n)

  const a1 = topDirection(rows, n, d, mean, null, 1)
  const a2 = topDirection(rows, n, d, mean, a1, 2)

  if (previous) {
    const align = (axis: Float32Array, prev: Float32Array) => {
      let dot = 0
      for (let i = 0; i < axis.length && i < prev.length; i++) dot += axis[i] * prev[i]
      if (dot < 0) for (let i = 0; i < axis.length; i++) axis[i] = -axis[i]
    }
    align(a1, previous.axes[0])
    align(a2, previous.axes[1])
  }

  const coords = new Float32Array(n * 2)
  for (let r = 0; r < n; r++) {
    const off = r * d
    let x = 0
    let y = 0
    for (let i = 0; i < d; i++) {
      const c = rows[off + i] - mean[i]
      x += c * a1[i]
      y += c * a2[i]
    }
    coords[r * 2] = x
    coords[r * 2 + 1] = y
  }

  return { coords, basis: { axes: [a1, a2], mean } }
}
