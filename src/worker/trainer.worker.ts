/// <reference lib="webworker" />
/**
 * Training worker.
 *
 * Training runs here rather than on the page for one reason: a transformer
 * step is hundreds of milliseconds of dense arithmetic, and on the main thread
 * that means a frozen scroll and a stuttering loss curve at exactly the moment
 * the visitor is watching most closely.
 */

import { Adam, lrAt } from '../model/adam'
import { TinyTransformer, generate } from '../model/model'
import { buildDataset, decode, sampleBatch } from '../model/tokenizer'
import { mulberry32 } from '../model/tensor'
import { serializeModel } from '../model/quantize'
import { pca2d, type Basis } from '../model/pca'
import type { ProgressMessage, WorkerRequest, WorkerResponse } from './protocol'

const ctx = self as unknown as DedicatedWorkerGlobalScope

let cancelled = false

/** How many tokens the embedding scatter plots. Beyond this it is a smear. */
const MAX_PLOTTED_TOKENS = 120

function post(msg: WorkerResponse, transfer: Transferable[] = []) {
  ctx.postMessage(msg, transfer)
}

ctx.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const req = e.data
  if (req.type === 'cancel') {
    cancelled = true
    return
  }
  if (req.type === 'train') {
    cancelled = false
    void train(req)
  }
}

async function train(req: Extract<WorkerRequest, { type: 'train' }>) {
  try {
    const { contextLength: T, batchSize: B, steps } = req
    const data = buildDataset(req.text, req.maxVocab, T)
    const V = data.vocab.tokens.length

    if (data.windowCount < 2) {
      post({
        type: 'error',
        message: 'That text is too short to make even one training window.',
      })
      return
    }

    const model = new TinyTransformer({
      vocabSize: V,
      dModel: req.dModel,
      nLayers: req.nLayers,
      nHeads: req.nHeads,
      contextLength: T,
      seed: req.seed,
    })

    post({
      type: 'ready',
      vocab: data.vocab.tokens,
      tokenCount: data.ids.length,
      windowCount: data.windowCount,
      distinctWords: data.vocab.distinctWords,
      unkOccurrences: data.vocab.unkOccurrences,
      paramCount: model.paramCount,
      uniformLoss: Math.log(V),
      config: model.cfg,
    })

    const opt = new Adam(model.params, { lr: req.lr })
    const rng = mulberry32(req.seed + 1)
    const inputs = new Int32Array(B * T)
    const targets = new Int32Array(B * T)
    const lossHistory = new Float32Array(steps)

    // Which token ids the scatter plot tracks: the most frequent ones, since
    // rare words barely move and would only add noise.
    const plotted = data.vocab.counts
      .map((count, id) => ({ count, id }))
      .filter((r) => r.id >= 3)
      .sort((a, b) => b.count - a.count)
      .slice(0, MAX_PLOTTED_TOKENS)
      .map((r) => r.id)
    const plottedIds = Int32Array.from(plotted)
    let basis: Basis | null = null

    const started = performance.now()
    let lastPostAt = started
    let stepsSinceLastPost = 0

    for (let step = 0; step < steps; step++) {
      if (cancelled) break

      sampleBatch(inputs, targets, data.ids, B, T, rng)
      model.zeroGrad()
      const loss = model.lossAndBackward(inputs, targets, B, T)
      const lr = lrAt(step, steps, req.lr)
      const { gradNorm } = opt.step(lr)
      lossHistory[step] = loss
      stepsSinceLastPost++

      const isSampleStep = step % 40 === 0 || step === steps - 1
      const isProgressStep = step % 5 === 0 || step === steps - 1

      if (isProgressStep) {
        const now = performance.now()
        const msg: ProgressMessage = {
          type: 'progress',
          step,
          totalSteps: steps,
          loss,
          lr,
          gradNorm,
          stepsPerSecond: (stepsSinceLastPost * 1000) / Math.max(1, now - lastPostAt),
        }
        lastPostAt = now
        stepsSinceLastPost = 0

        if (isSampleStep) {
          const out = generate(model, [...data.ids.slice(1, 4)], {
            maxTokens: 20,
            temperature: 0.8,
            topK: 12,
            rng: mulberry32(99),
          })
          msg.sample = decode(out, data.vocab)

          const d = model.cfg.dModel
          const rows = new Float32Array(plottedIds.length * d)
          for (let i = 0; i < plottedIds.length; i++) {
            rows.set(
              model.tokenEmbed.data.subarray(plottedIds[i] * d, plottedIds[i] * d + d),
              i * d,
            )
          }
          const { coords, basis: nextBasis } = pca2d(rows, plottedIds.length, d, basis)
          basis = nextBasis
          msg.embedding = { coords, ids: plottedIds }
        }

        post(msg)
        // Yield so a cancel message can actually be delivered. Without this
        // the worker is busy for the entire run and Stop does nothing.
        await new Promise((r) => setTimeout(r, 0))
      }
    }

    const payload = serializeModel(model.cfg, model.params, data.vocab.tokens)
    post({
      type: 'done',
      weights: payload.tensors,
      finalLoss: lossHistory[Math.max(0, Math.min(steps, opt.stepCount) - 1)],
      elapsedMs: performance.now() - started,
      lossHistory: lossHistory.slice(0, Math.max(1, opt.stepCount)),
    })
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) })
  }
}
