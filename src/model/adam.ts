/**
 * Adam, with bias correction and global-norm gradient clipping.
 *
 * Plain SGD would also work here, but Adam's per-parameter step size matters a
 * lot for a model whose embedding rows are updated at wildly different rates:
 * a word appearing 200 times and a word appearing twice need very different
 * effective learning rates, and Adam supplies that automatically.
 */

import type { Param } from './model'

export interface AdamConfig {
  lr: number
  beta1: number
  beta2: number
  eps: number
  /** Max global gradient norm; 0 disables clipping. */
  clipNorm: number
  weightDecay: number
}

export const DEFAULT_ADAM: AdamConfig = {
  lr: 3e-3,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  clipNorm: 1,
  weightDecay: 0,
}

export class Adam {
  private m: Float32Array[]
  private v: Float32Array[]
  private t = 0
  readonly cfg: AdamConfig

  constructor(
    private params: Param[],
    cfg: Partial<AdamConfig> = {},
  ) {
    this.cfg = { ...DEFAULT_ADAM, ...cfg }
    this.m = params.map((p) => new Float32Array(p.data.length))
    this.v = params.map((p) => new Float32Array(p.data.length))
  }

  /** Global L2 norm across every gradient, as one vector. */
  gradNorm(): number {
    let total = 0
    for (const p of this.params) {
      for (let i = 0; i < p.grad.length; i++) total += p.grad[i] * p.grad[i]
    }
    return Math.sqrt(total)
  }

  /**
   * One optimiser step. `lr` overrides the configured rate, which is how the
   * warmup/decay schedule is applied without mutating config.
   */
  step(lrOverride?: number): { gradNorm: number; clipped: boolean } {
    const { beta1, beta2, eps, clipNorm, weightDecay } = this.cfg
    const lr = lrOverride ?? this.cfg.lr
    this.t += 1

    // Clip the whole gradient as a single vector, not per parameter. Clipping
    // per tensor would silently change the *direction* of the update; scaling
    // everything by one factor preserves it and only shortens the step.
    const norm = this.gradNorm()
    const clipped = clipNorm > 0 && norm > clipNorm
    const scale = clipped ? clipNorm / (norm + 1e-6) : 1

    // Bias correction: m and v start at zero, so early estimates are biased
    // toward zero. Without these divisors the first few steps are far too
    // small and training looks stalled exactly when the reader is watching.
    const bc1 = 1 - Math.pow(beta1, this.t)
    const bc2 = 1 - Math.pow(beta2, this.t)

    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi]
      const m = this.m[pi]
      const v = this.v[pi]
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i] * scale
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g * g
        const mHat = m[i] / bc1
        const vHat = v[i] / bc2
        let update = (lr * mHat) / (Math.sqrt(vHat) + eps)
        if (weightDecay > 0) update += lr * weightDecay * p.data[i]
        p.data[i] -= update
      }
    }
    return { gradNorm: norm, clipped }
  }

  get stepCount(): number {
    return this.t
  }
}

/**
 * Learning rate schedule: short linear warmup, then cosine decay to 10%.
 *
 * The warmup exists because Adam's second-moment estimate is noisy for the
 * first few dozen steps; taking full-size steps then tends to spike the loss,
 * which reads as "the model got worse" on a live chart.
 */
export function lrAt(step: number, totalSteps: number, peakLr: number): number {
  const warmup = Math.max(5, Math.floor(totalSteps * 0.05))
  if (step < warmup) return (peakLr * (step + 1)) / warmup
  const progress = (step - warmup) / Math.max(1, totalSteps - warmup)
  const cosine = 0.5 * (1 + Math.cos(Math.PI * Math.min(1, progress)))
  return peakLr * (0.1 + 0.9 * cosine)
}
