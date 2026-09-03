/**
 * Causality and normalisation of the attention weights.
 *
 * These two properties are the whole claim the site makes about attention, and
 * they are also what every heatmap and arc diagram is drawn from. If the
 * triangle leaked, the pictures would be wrong and so would the teaching.
 */

import { describe, expect, it } from 'vitest'
import { TinyTransformer } from '../src/model/model'
import { mulberry32 } from '../src/model/tensor'

function model(nHeads = 2, nLayers = 2, dModel = 8, contextLength = 6) {
  return new TinyTransformer({
    vocabSize: 13,
    dModel,
    nLayers,
    nHeads,
    contextLength,
    seed: 3,
  })
}

describe('attention weights', () => {
  it('never attends to a future token', () => {
    const m = model()
    const T = m.cfg.contextLength
    const H = m.cfg.nHeads
    const B = 2
    const rng = mulberry32(11)
    const inputs = new Int32Array(B * T)
    for (let i = 0; i < inputs.length; i++) inputs[i] = Math.floor(rng() * m.cfg.vocabSize)

    const act = m.forward(inputs, B, T)

    for (let l = 0; l < m.cfg.nLayers; l++) {
      const probs = act.layers[l].attn.probs
      for (let b = 0; b < B; b++) {
        for (let h = 0; h < H; h++) {
          const base = b * H * T * T + h * T * T
          for (let t1 = 0; t1 < T; t1++) {
            for (let t2 = t1 + 1; t2 < T; t2++) {
              expect(
                probs[base + t1 * T + t2],
                `layer ${l} head ${h}: position ${t1} peeked at ${t2}`,
              ).toBe(0)
            }
          }
        }
      }
    }
  })

  it('gives every row a distribution summing to one', () => {
    const m = model(4, 1, 8, 5)
    const T = m.cfg.contextLength
    const H = m.cfg.nHeads
    const B = 3
    const rng = mulberry32(22)
    const inputs = new Int32Array(B * T)
    for (let i = 0; i < inputs.length; i++) inputs[i] = Math.floor(rng() * m.cfg.vocabSize)

    const act = m.forward(inputs, B, T)
    const probs = act.layers[0].attn.probs

    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        const base = b * H * T * T + h * T * T
        for (let t1 = 0; t1 < T; t1++) {
          let sum = 0
          for (let t2 = 0; t2 < T; t2++) sum += probs[base + t1 * T + t2]
          expect(sum, `row ${t1} of head ${h}`).toBeCloseTo(1, 5)
        }
      }
    }
  })

  it('the first position can only attend to itself', () => {
    const m = model()
    const T = m.cfg.contextLength
    const inputs = new Int32Array(T)
    const act = m.forward(inputs, 1, T)
    const probs = act.layers[0].attn.probs
    expect(probs[0]).toBeCloseTo(1, 6)
  })

  it('a token prediction is unaffected by tokens after it', () => {
    // The practical consequence of causality: changing the tail of the window
    // must not move an earlier position's logits by even a float.
    const m = model()
    const T = m.cfg.contextLength
    const a = new Int32Array([1, 2, 3, 4, 5, 6])
    const b = new Int32Array([1, 2, 3, 9, 9, 9])

    const logitsA = Float32Array.from(m.forward(a, 1, T).logits)
    const logitsB = Float32Array.from(m.forward(b, 1, T).logits)

    const V = m.cfg.vocabSize
    for (let t = 0; t < 3; t++) {
      for (let i = 0; i < V; i++) {
        expect(logitsA[t * V + i]).toBe(logitsB[t * V + i])
      }
    }
    // ...and position 3 onward must differ, or the test proves nothing.
    let differs = false
    for (let i = 0; i < V; i++) {
      if (logitsA[3 * V + i] !== logitsB[3 * V + i]) differs = true
    }
    expect(differs).toBe(true)
  })

  it('a single attention layer cannot tell a reordered context apart', () => {
    // The exact claim stage 5 makes. Note two constraints, both load-bearing:
    //
    // 1. Only the PREFIX is shuffled. Attention is permutation-equivariant, not
    //    invariant, so the token doing the asking still matters and both
    //    sequences must end on the same word.
    // 2. It is measured on ONE layer. See the test below for why.
    const m = model(2, 1, 8, 4)
    const manBitesDog = [4, 5, 6, 7]
    const bitesManDog = [5, 4, 6, 7]

    const offA = m.predictNext(manBitesDog, false).probs
    const offB = m.predictNext(bitesManDog, false).probs
    for (let i = 0; i < offA.length; i++) {
      expect(offA[i], `token ${i}`).toBeCloseTo(offB[i], 6)
    }

    const onA = m.predictNext(manBitesDog, true).probs
    const onB = m.predictNext(bitesManDog, true).probs
    let maxDiff = 0
    for (let i = 0; i < onA.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(onA[i] - onB[i]))
    }
    expect(maxDiff).toBeGreaterThan(1e-6)
  })

  it('the first attention layer stays order-blind even in a deeper model', () => {
    const m = model(2, 3, 8, 4)
    const d = m.cfg.dModel
    const T = m.cfg.contextLength

    const firstLayerOut = (ids: number[]) => {
      const inputs = new Int32Array(T)
      for (let i = 0; i < ids.length; i++) inputs[i] = ids[i]
      const act = m.forward(inputs, 1, T, false)
      return act.layers[0].attnOut.slice((ids.length - 1) * d, ids.length * d)
    }

    const a = firstLayerOut([4, 5, 6, 7])
    const b = firstLayerOut([5, 4, 6, 7])
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i] - b[i]), `component ${i}`).toBeLessThan(1e-5)
    }
  })

  it('a stacked model is NOT order-blind, because masking leaks position', () => {
    // Counterintuitive but real, and the reason stage 5 measures one layer
    // rather than the final prediction. Each layer reads the layer below it at
    // every position, and those positions see different prefixes once the order
    // changes -- so causal masking alone carries some order information, with
    // no position embeddings involved at all.
    //
    // This test exists to stop someone "fixing" stage 5 to assert the stronger
    // claim, which would be false.
    const m = model(2, 2, 8, 4)
    const offA = m.predictNext([4, 5, 6, 7], false).probs
    const offB = m.predictNext([5, 4, 6, 7], false).probs
    let maxDiff = 0
    for (let i = 0; i < offA.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(offA[i] - offB[i]))
    }
    expect(maxDiff).toBeGreaterThan(1e-6)
  })
})
