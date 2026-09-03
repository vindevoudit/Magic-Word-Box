/**
 * Finite-difference gradient checks.
 *
 * The backward pass in this project is hand-derived, so it is guilty until
 * proven innocent: a subtly wrong gradient still trains *something*, just to a
 * worse model, and the site would then be teaching a lie with a straight face.
 *
 * Method: for each parameter tensor separately, pick a random direction u with
 * entries in {+1,-1}, and compare the analytic directional derivative
 * `<grad, u>` against the central difference `(L(t+hu) - L(t-hu)) / 2h`.
 *
 * Perturbing one whole tensor at a time is what makes this both sensitive and
 * numerically stable. A single-entry check on float32 weights has a terrible
 * signal-to-noise ratio, because the loss moves by ~1e-8 while float32 forward
 * noise is ~1e-6. Summing over a tensor lifts the signal by sqrt(n) while the
 * noise stays put, and it still catches wrong signs, wrong scale factors,
 * missing terms, transposed indices and permuted rows.
 */

import { describe, expect, it } from 'vitest'
import { TinyTransformer, type ModelConfig } from '../src/model/model'
import { mulberry32 } from '../src/model/tensor'

const H_STEP = 1e-3
const TOL = 5e-3

function makeFixture(cfg: Partial<ModelConfig> = {}, seed = 7) {
  const full: ModelConfig = {
    vocabSize: 17,
    dModel: 8,
    nLayers: 2,
    nHeads: 2,
    contextLength: 5,
    seed: 1234,
    ...cfg,
  }
  const model = new TinyTransformer(full)
  const B = 3
  const T = full.contextLength
  const rng = mulberry32(seed)
  const inputs = new Int32Array(B * T)
  const targets = new Int32Array(B * T)
  for (let i = 0; i < B * T; i++) {
    inputs[i] = Math.floor(rng() * full.vocabSize)
    targets[i] = Math.floor(rng() * full.vocabSize)
  }
  return { model, inputs, targets, B, T }
}

function relError(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1, Math.abs(a), Math.abs(b))
}

describe('gradient checks', () => {
  it('every parameter tensor matches its finite-difference derivative', () => {
    const { model, inputs, targets, B, T } = makeFixture()

    model.zeroGrad()
    const baseLoss = model.lossAndBackward(inputs, targets, B, T)
    expect(Number.isFinite(baseLoss)).toBe(true)

    const dirRng = mulberry32(99)
    const failures: string[] = []

    for (const p of model.params) {
      const n = p.data.length
      const u = new Float32Array(n)
      for (let i = 0; i < n; i++) u[i] = dirRng() < 0.5 ? -1 : 1

      let analytic = 0
      for (let i = 0; i < n; i++) analytic += p.grad[i] * u[i]

      const original = Float32Array.from(p.data)

      for (let i = 0; i < n; i++) p.data[i] = original[i] + H_STEP * u[i]
      const lossPlus = model.evalLoss(inputs, targets, B, T)

      for (let i = 0; i < n; i++) p.data[i] = original[i] - H_STEP * u[i]
      const lossMinus = model.evalLoss(inputs, targets, B, T)

      p.data.set(original)

      const numeric = (lossPlus - lossMinus) / (2 * H_STEP)
      const err = relError(analytic, numeric)

      if (!(err < TOL)) {
        failures.push(
          `${p.name}: analytic=${analytic.toFixed(6)} numeric=${numeric.toFixed(6)} relErr=${err.toExponential(2)}`,
        )
      }
    }

    expect(failures, `gradient mismatch:\n${failures.join('\n')}`).toEqual([])
  })

  it('holds for a single-head, single-layer model', () => {
    const { model, inputs, targets, B, T } = makeFixture({
      nLayers: 1,
      nHeads: 1,
      dModel: 6,
      vocabSize: 11,
    })
    model.zeroGrad()
    model.lossAndBackward(inputs, targets, B, T)

    const dirRng = mulberry32(5)
    for (const p of model.params) {
      const n = p.data.length
      const u = new Float32Array(n)
      for (let i = 0; i < n; i++) u[i] = dirRng() < 0.5 ? -1 : 1
      let analytic = 0
      for (let i = 0; i < n; i++) analytic += p.grad[i] * u[i]

      const original = Float32Array.from(p.data)
      for (let i = 0; i < n; i++) p.data[i] = original[i] + H_STEP * u[i]
      const lp = model.evalLoss(inputs, targets, B, T)
      for (let i = 0; i < n; i++) p.data[i] = original[i] - H_STEP * u[i]
      const lm = model.evalLoss(inputs, targets, B, T)
      p.data.set(original)

      const numeric = (lp - lm) / (2 * H_STEP)
      expect(relError(analytic, numeric), `${p.name}`).toBeLessThan(TOL)
    }
  })

  it('holds with more heads than the head dimension is wide', () => {
    const { model, inputs, targets, B, T } = makeFixture({
      dModel: 8,
      nHeads: 4,
      nLayers: 2,
    })
    model.zeroGrad()
    model.lossAndBackward(inputs, targets, B, T)

    const dirRng = mulberry32(41)
    for (const p of model.params) {
      const n = p.data.length
      const u = new Float32Array(n)
      for (let i = 0; i < n; i++) u[i] = dirRng() < 0.5 ? -1 : 1
      let analytic = 0
      for (let i = 0; i < n; i++) analytic += p.grad[i] * u[i]

      const original = Float32Array.from(p.data)
      for (let i = 0; i < n; i++) p.data[i] = original[i] + H_STEP * u[i]
      const lp = model.evalLoss(inputs, targets, B, T)
      for (let i = 0; i < n; i++) p.data[i] = original[i] - H_STEP * u[i]
      const lm = model.evalLoss(inputs, targets, B, T)
      p.data.set(original)

      expect(relError(analytic, (lp - lm) / (2 * H_STEP)), `${p.name}`).toBeLessThan(TOL)
    }
  })

  it('gradients are finite and not identically zero', () => {
    const { model, inputs, targets, B, T } = makeFixture()
    model.zeroGrad()
    model.lossAndBackward(inputs, targets, B, T)
    for (const p of model.params) {
      let sumAbs = 0
      for (let i = 0; i < p.grad.length; i++) {
        expect(Number.isFinite(p.grad[i]), `${p.name}[${i}] not finite`).toBe(true)
        sumAbs += Math.abs(p.grad[i])
      }
      // posEmbed rows beyond the batch would legitimately be zero, but with
      // T equal to the context length every row is touched.
      expect(sumAbs, `${p.name} has an all-zero gradient`).toBeGreaterThan(0)
    }
  })
})
