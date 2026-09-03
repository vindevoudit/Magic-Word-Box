/**
 * Trains the model that ships with the site and writes public/pretrained.json.
 *
 * Run with `npm run pretrain` whenever the model code or the default corpus
 * changes. The output is committed, so the Render build stays a plain `vite
 * build` with no training step and no devDependency surprises on the server.
 */

import { writeFileSync } from 'node:fs'
import { Adam, lrAt } from '../src/model/adam'
import { TinyTransformer, generate } from '../src/model/model'
import { buildDataset, decode, sampleBatch } from '../src/model/tokenizer'
import { mulberry32 } from '../src/model/tensor'
import { serializeModel } from '../src/model/quantize'
import { DEFAULT_CORPUS } from '../src/data/corpora'
import { SHIPPED_PRESET } from '../src/model/presets'

const OUT = 'public/pretrained.json'

function main() {
  const { dModel, nLayers, nHeads, contextLength, batchSize, steps, lr, maxVocab } =
    SHIPPED_PRESET

  const data = buildDataset(DEFAULT_CORPUS.text, maxVocab, contextLength)
  const V = data.vocab.tokens.length
  console.log(
    `corpus "${DEFAULT_CORPUS.name}": ${data.ids.length} tokens, ${V} vocab, ${data.windowCount} windows`,
  )

  const model = new TinyTransformer({
    vocabSize: V,
    dModel,
    nLayers,
    nHeads,
    contextLength,
    seed: 1337,
  })
  console.log(`model: ${model.paramCount.toLocaleString()} parameters`)

  const opt = new Adam(model.params, { lr })
  const rng = mulberry32(2024)
  const inputs = new Int32Array(batchSize * contextLength)
  const targets = new Int32Array(batchSize * contextLength)

  const started = Date.now()
  let loss = 0
  for (let step = 0; step < steps; step++) {
    sampleBatch(inputs, targets, data.ids, batchSize, contextLength, rng)
    model.zeroGrad()
    loss = model.lossAndBackward(inputs, targets, batchSize, contextLength)
    opt.step(lrAt(step, steps, lr))
    if (step % 200 === 0 || step === steps - 1) {
      console.log(`  step ${String(step).padStart(4)}  loss ${loss.toFixed(4)}`)
    }
  }
  const elapsed = Date.now() - started
  console.log(
    `trained ${steps} steps in ${(elapsed / 1000).toFixed(1)}s  (${(elapsed / steps).toFixed(1)} ms/step)`,
  )
  console.log(`uniform baseline would be ${Math.log(V).toFixed(3)} nats`)

  const sample = generate(model, [...data.ids.slice(0, 3)], {
    maxTokens: 24,
    temperature: 0.7,
    topK: 10,
    rng: mulberry32(5),
  })
  console.log('\nsample generation:\n' + decode(sample, data.vocab) + '\n')

  const payload = serializeModel(model.cfg, model.params, data.vocab.tokens)
  const json = JSON.stringify(payload)
  writeFileSync(OUT, json)
  console.log(`wrote ${OUT}  (${(json.length / 1024).toFixed(0)} KB)`)
}

main()
