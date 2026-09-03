/**
 * Can it learn at all?
 *
 * Gradient checks prove the derivatives are right; they do not prove the
 * optimiser, schedule, batching and tokenizer are wired together correctly.
 * The standard end-to-end proof is to make a model memorise one short text.
 * If loss does not collapse on data the model has ample capacity to store,
 * something between the data loader and the update step is broken.
 */

import { describe, expect, it } from 'vitest'
import { Adam, lrAt } from '../src/model/adam'
import { TinyTransformer, generate } from '../src/model/model'
import { buildDataset, decode, sampleBatch } from '../src/model/tokenizer'
import { mulberry32 } from '../src/model/tensor'

const SENTENCE = 'the quick brown fox jumps over the lazy dog and then the fox sleeps'

function trainOnce(text: string, opts: {
  T: number
  dModel: number
  nLayers: number
  nHeads: number
  B: number
  steps: number
  lr: number
  seed: number
}) {
  const data = buildDataset(text, 128, opts.T)
  const model = new TinyTransformer({
    vocabSize: data.vocab.tokens.length,
    dModel: opts.dModel,
    nLayers: opts.nLayers,
    nHeads: opts.nHeads,
    contextLength: opts.T,
    seed: opts.seed,
  })
  const opt = new Adam(model.params, { lr: opts.lr })
  const rng = mulberry32(7)
  const inputs = new Int32Array(opts.B * opts.T)
  const targets = new Int32Array(opts.B * opts.T)

  let firstLoss = 0
  let lastLoss = 0
  for (let step = 0; step < opts.steps; step++) {
    sampleBatch(inputs, targets, data.ids, opts.B, opts.T, rng)
    model.zeroGrad()
    const loss = model.lossAndBackward(inputs, targets, opts.B, opts.T)
    opt.step(lrAt(step, opts.steps, opts.lr))
    if (step === 0) firstLoss = loss
    lastLoss = loss
  }
  return { model, data, firstLoss, lastLoss }
}

describe('end-to-end learning', () => {
  it('memorises a single sentence', () => {
    const { firstLoss, lastLoss } = trainOnce(SENTENCE, {
      T: 8, dModel: 32, nLayers: 2, nHeads: 4, B: 8, steps: 400, lr: 5e-3, seed: 42,
    })
    expect(firstLoss).toBeGreaterThan(1.5)
    expect(
      lastLoss,
      `loss went ${firstLoss.toFixed(3)} -> ${lastLoss.toFixed(3)}`,
    ).toBeLessThan(0.1)
  })

  it('reproduces the text it memorised', () => {
    const { model, data } = trainOnce(SENTENCE, {
      T: 8, dModel: 32, nLayers: 2, nHeads: 4, B: 8, steps: 400, lr: 5e-3, seed: 42,
    })
    const prompt = [...data.ids.slice(0, 4)]
    const out = generate(model, prompt, {
      maxTokens: 5,
      temperature: 0,
      topK: 0,
      rng: mulberry32(1),
    })
    expect(decode(out, data.vocab)).toContain('the quick brown fox jumps')
  })

  it('beats the uniform baseline on a repetitive corpus', () => {
    const corpus = [
      'twinkle twinkle little star',
      'how i wonder what you are',
      'up above the world so high',
      'like a diamond in the sky',
      'twinkle twinkle little star',
      'how i wonder what you are',
    ].join('\n')

    const { data, lastLoss } = trainOnce(corpus, {
      T: 8, dModel: 32, nLayers: 1, nHeads: 2, B: 16, steps: 300, lr: 4e-3, seed: 5,
    })
    const uniform = Math.log(data.vocab.tokens.length)
    expect(
      lastLoss,
      `uniform baseline is ${uniform.toFixed(3)} nats, got ${lastLoss.toFixed(3)}`,
    ).toBeLessThan(uniform * 0.25)
  })
})
