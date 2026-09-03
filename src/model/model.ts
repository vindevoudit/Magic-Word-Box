/**
 * TinyTransformer: a decoder-only transformer, small enough to train in a tab.
 *
 * Architecture is deliberately the real thing rather than a simplification —
 * pre-layer-norm blocks, residual connections, multi-head causal attention, a
 * 4x GELU feed-forward, a final norm, and a linear head onto the vocabulary.
 * Only the sizes are toy. Anyone who understands this file understands the
 * shape of GPT.
 */

import {
  attentionBackward,
  attentionForward,
  makeAttentionBackScratch,
  makeAttentionCache,
  type AttentionBackScratch,
  type AttentionCache,
  type AttentionDims,
} from './attention'
import {
  geluBackward,
  geluForward,
  layerNormBackward,
  layerNormForward,
  makeLayerNormCache,
  softmaxCrossEntropy,
  type LayerNormCache,
} from './ops'
import {
  addBias,
  matmul,
  matmulNT,
  matmulTN_acc,
  mulberry32,
  randn,
  sumRows_acc,
} from './tensor'
import { BOS } from './tokenizer'

export interface ModelConfig {
  vocabSize: number
  dModel: number
  nLayers: number
  nHeads: number
  contextLength: number
  seed: number
}

/** A named parameter and its gradient, kept side by side. */
export interface Param {
  name: string
  data: Float32Array
  grad: Float32Array
  shape: number[]
}

interface LayerParams {
  ln1g: Param
  ln1b: Param
  wq: Param
  wk: Param
  wv: Param
  wo: Param
  bq: Param
  bk: Param
  bv: Param
  bo: Param
  ln2g: Param
  ln2b: Param
  fc1: Param
  fc1b: Param
  fc2: Param
  fc2b: Param
}

interface LayerAct {
  ln1: LayerNormCache
  ln1Out: Float32Array
  attn: AttentionCache
  attnOut: Float32Array
  resid1: Float32Array
  ln2: LayerNormCache
  ln2Out: Float32Array
  fc1Out: Float32Array
  geluOut: Float32Array
  mlpOut: Float32Array
  resid2: Float32Array
}

export interface Activations {
  B: number
  T: number
  inputs: Int32Array
  embed: Float32Array
  layers: LayerAct[]
  lnF: LayerNormCache
  lnFOut: Float32Array
  logits: Float32Array
  probs: Float32Array
  scratchScores: Float32Array
}

interface BackBuffers {
  dLogits: Float32Array
  dx: Float32Array
  dxNext: Float32Array
  dHidden: Float32Array
  dGelu: Float32Array
  /** Reused per layer so the backward pass allocates nothing per step. */
  dResid: Float32Array
  dxIn: Float32Array
  attnScratch: AttentionBackScratch
}

export class TinyTransformer {
  readonly cfg: ModelConfig
  readonly params: Param[] = []

  tokenEmbed: Param
  posEmbed: Param
  lnFg: Param
  lnFb: Param
  head: Param
  headB: Param
  layers: LayerParams[] = []

  private act: Activations | null = null
  private back: BackBuffers | null = null

  constructor(cfg: ModelConfig) {
    if (cfg.dModel % cfg.nHeads !== 0) {
      throw new Error(
        `dModel (${cfg.dModel}) must divide evenly into nHeads (${cfg.nHeads})`,
      )
    }
    this.cfg = cfg
    const rng = mulberry32(cfg.seed)
    const { vocabSize: V, dModel: d, nLayers: L, contextLength: T } = cfg

    // GPT-2 initialisation: every weight from N(0, 0.02), biases at zero, and
    // layer-norm gains at one. Small enough that the residual stream does not
    // explode through L blocks, large enough that gradients are not asleep.
    const p = (name: string, shape: number[], init: 'normal' | 'zeros' | 'ones') => {
      const n = shape.reduce((a, b) => a * b, 1)
      const data = new Float32Array(n)
      if (init === 'normal') for (let i = 0; i < n; i++) data[i] = randn(rng) * 0.02
      else if (init === 'ones') data.fill(1)
      const param: Param = { name, data, grad: new Float32Array(n), shape }
      this.params.push(param)
      return param
    }

    this.tokenEmbed = p('tokenEmbed', [V, d], 'normal')
    this.posEmbed = p('posEmbed', [T, d], 'normal')

    for (let l = 0; l < L; l++) {
      this.layers.push({
        ln1g: p(`block${l}.ln1.gain`, [d], 'ones'),
        ln1b: p(`block${l}.ln1.bias`, [d], 'zeros'),
        wq: p(`block${l}.attn.wq`, [d, d], 'normal'),
        wk: p(`block${l}.attn.wk`, [d, d], 'normal'),
        wv: p(`block${l}.attn.wv`, [d, d], 'normal'),
        wo: p(`block${l}.attn.wo`, [d, d], 'normal'),
        bq: p(`block${l}.attn.bq`, [d], 'zeros'),
        bk: p(`block${l}.attn.bk`, [d], 'zeros'),
        bv: p(`block${l}.attn.bv`, [d], 'zeros'),
        bo: p(`block${l}.attn.bo`, [d], 'zeros'),
        ln2g: p(`block${l}.ln2.gain`, [d], 'ones'),
        ln2b: p(`block${l}.ln2.bias`, [d], 'zeros'),
        fc1: p(`block${l}.mlp.fc1`, [d, 4 * d], 'normal'),
        fc1b: p(`block${l}.mlp.fc1.bias`, [4 * d], 'zeros'),
        fc2: p(`block${l}.mlp.fc2`, [4 * d, d], 'normal'),
        fc2b: p(`block${l}.mlp.fc2.bias`, [d], 'zeros'),
      })
    }

    this.lnFg = p('lnFinal.gain', [d], 'ones')
    this.lnFb = p('lnFinal.bias', [d], 'zeros')
    this.head = p('head', [d, V], 'normal')
    this.headB = p('head.bias', [V], 'zeros')
  }

  get paramCount(): number {
    return this.params.reduce((n, p) => n + p.data.length, 0)
  }

  /** Per-component parameter counts, for the breakdown chart in stage 7. */
  paramBreakdown(): { label: string; count: number }[] {
    const group = (prefix: string) =>
      this.params
        .filter((p) => p.name.startsWith(prefix))
        .reduce((n, p) => n + p.data.length, 0)
    const attn = this.params
      .filter((p) => p.name.includes('.attn.'))
      .reduce((n, p) => n + p.data.length, 0)
    const mlp = this.params
      .filter((p) => p.name.includes('.mlp.'))
      .reduce((n, p) => n + p.data.length, 0)
    const norms = this.params
      .filter((p) => p.name.includes('ln'))
      .reduce((n, p) => n + p.data.length, 0)
    return [
      { label: 'Token embeddings', count: group('tokenEmbed') },
      { label: 'Position embeddings', count: group('posEmbed') },
      { label: 'Attention', count: attn },
      { label: 'Feed-forward', count: mlp },
      { label: 'Layer norms', count: norms },
      { label: 'Output head', count: group('head') },
    ].filter((r) => r.count > 0)
  }

  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0)
  }

  /** Allocate (or reuse) activation buffers for a given batch shape. */
  private ensureBuffers(B: number, T: number): { act: Activations; back: BackBuffers } {
    const { dModel: d, nLayers: L, nHeads: H, vocabSize: V } = this.cfg
    if (this.act && this.act.B === B && this.act.T === T && this.back) {
      return { act: this.act, back: this.back }
    }
    const rows = B * T
    const layers: LayerAct[] = []
    for (let l = 0; l < L; l++) {
      layers.push({
        ln1: makeLayerNormCache(rows, d),
        ln1Out: new Float32Array(rows * d),
        attn: makeAttentionCache(B, T, d, H),
        attnOut: new Float32Array(rows * d),
        resid1: new Float32Array(rows * d),
        ln2: makeLayerNormCache(rows, d),
        ln2Out: new Float32Array(rows * d),
        fc1Out: new Float32Array(rows * 4 * d),
        geluOut: new Float32Array(rows * 4 * d),
        mlpOut: new Float32Array(rows * d),
        resid2: new Float32Array(rows * d),
      })
    }
    this.act = {
      B,
      T,
      inputs: new Int32Array(rows),
      embed: new Float32Array(rows * d),
      layers,
      lnF: makeLayerNormCache(rows, d),
      lnFOut: new Float32Array(rows * d),
      logits: new Float32Array(rows * V),
      probs: new Float32Array(rows * V),
      scratchScores: new Float32Array(T * T),
    }
    this.back = {
      dLogits: new Float32Array(rows * V),
      dx: new Float32Array(rows * d),
      dxNext: new Float32Array(rows * d),
      dHidden: new Float32Array(rows * 4 * d),
      dGelu: new Float32Array(rows * 4 * d),
      dResid: new Float32Array(rows * d),
      dxIn: new Float32Array(rows * d),
      attnScratch: makeAttentionBackScratch(B, T, d),
    }
    return { act: this.act, back: this.back }
  }

  /**
   * Forward pass over `[B, T]` token ids. Returns the filled activations.
   *
   * `usePositional: false` zeroes out the position embeddings. That is not a
   * performance switch — it powers the demonstration in stage 5, where the
   * model becomes literally unable to tell word order apart.
   */
  forward(inputs: Int32Array, B: number, T: number, usePositional = true): Activations {
    const { dModel: d, nLayers: L, nHeads: H, vocabSize: V } = this.cfg
    const { act } = this.ensureBuffers(B, T)
    const rows = B * T
    act.inputs.set(inputs)

    // --- embed: look up each token, then add where it sits in the window ---
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const row = (b * T + t) * d
        const tokRow = inputs[b * T + t] * d
        const posRow = t * d
        for (let i = 0; i < d; i++) {
          act.embed[row + i] =
            this.tokenEmbed.data[tokRow + i] +
            (usePositional ? this.posEmbed.data[posRow + i] : 0)
        }
      }
    }

    const dims: AttentionDims = { B, T, d, H }
    let x = act.embed

    for (let l = 0; l < L; l++) {
      const lp = this.layers[l]
      const la = act.layers[l]

      // Pre-norm: normalise going *into* each sublayer, and leave the residual
      // stream itself untouched. This is what lets gradients reach layer 0
      // without being squeezed through a norm at every step.
      layerNormForward(la.ln1Out, x, lp.ln1g.data, lp.ln1b.data, rows, d, la.ln1)
      attentionForward(
        la.attnOut,
        la.ln1Out,
        {
          wq: lp.wq.data,
          wk: lp.wk.data,
          wv: lp.wv.data,
          wo: lp.wo.data,
          bq: lp.bq.data,
          bk: lp.bk.data,
          bv: lp.bv.data,
          bo: lp.bo.data,
        },
        dims,
        la.attn,
        act.scratchScores,
      )
      for (let i = 0; i < rows * d; i++) la.resid1[i] = x[i] + la.attnOut[i]

      layerNormForward(la.ln2Out, la.resid1, lp.ln2g.data, lp.ln2b.data, rows, d, la.ln2)
      matmul(la.fc1Out, la.ln2Out, lp.fc1.data, rows, d, 4 * d)
      addBias(la.fc1Out, lp.fc1b.data, rows, 4 * d)
      geluForward(la.geluOut, la.fc1Out)
      matmul(la.mlpOut, la.geluOut, lp.fc2.data, rows, 4 * d, d)
      addBias(la.mlpOut, lp.fc2b.data, rows, d)
      for (let i = 0; i < rows * d; i++) la.resid2[i] = la.resid1[i] + la.mlpOut[i]

      x = la.resid2
    }

    layerNormForward(act.lnFOut, x, this.lnFg.data, this.lnFb.data, rows, d, act.lnF)
    matmul(act.logits, act.lnFOut, this.head.data, rows, d, V)
    addBias(act.logits, this.headB.data, rows, V)

    return act
  }

  /**
   * Forward, loss, and backward in one call. Accumulates into `param.grad`.
   * Returns mean cross-entropy loss in nats.
   */
  lossAndBackward(inputs: Int32Array, targets: Int32Array, B: number, T: number): number {
    const { dModel: d, nLayers: L, nHeads: H, vocabSize: V } = this.cfg
    const act = this.forward(inputs, B, T)
    const back = this.back
    if (!back) throw new Error('buffers not allocated')
    const rows = B * T

    const loss = softmaxCrossEntropy(
      act.probs,
      back.dLogits,
      act.logits,
      targets,
      rows,
      V,
    )

    // --- head ---
    matmulTN_acc(this.head.grad, act.lnFOut, back.dLogits, rows, d, V)
    sumRows_acc(this.headB.grad, back.dLogits, rows, V)
    matmulNT(back.dx, back.dLogits, this.head.data, rows, V, d)

    // --- final norm ---
    back.dxNext.fill(0)
    layerNormBackward(
      back.dxNext,
      this.lnFg.grad,
      this.lnFb.grad,
      back.dx,
      this.lnFg.data,
      rows,
      d,
      act.lnF,
    )
    back.dx.set(back.dxNext)

    const dims: AttentionDims = { B, T, d, H }

    for (let l = L - 1; l >= 0; l--) {
      const lp = this.layers[l]
      const la = act.layers[l]
      // --- MLP sublayer. The residual add means dx arrives at *both* the
      // sublayer input and, unchanged, at the stream below it. That identity
      // path is why deep transformers train at all. ---
      matmulTN_acc(lp.fc2.grad, la.geluOut, back.dx, rows, 4 * d, d)
      sumRows_acc(lp.fc2b.grad, back.dx, rows, d)
      matmulNT(back.dGelu, back.dx, lp.fc2.data, rows, d, 4 * d)
      geluBackward(back.dHidden, back.dGelu, la.fc1Out)
      matmulTN_acc(lp.fc1.grad, la.ln2Out, back.dHidden, rows, d, 4 * d)
      sumRows_acc(lp.fc1b.grad, back.dHidden, rows, 4 * d)
      matmulNT(back.dxNext, back.dHidden, lp.fc1.data, rows, 4 * d, d)

      const dResid1 = back.dResid
      layerNormBackward(
        dResid1,
        lp.ln2g.grad,
        lp.ln2b.grad,
        back.dxNext,
        lp.ln2g.data,
        rows,
        d,
        la.ln2,
      )
      for (let i = 0; i < rows * d; i++) dResid1[i] += back.dx[i]

      // --- attention sublayer ---
      attentionBackward(
        back.dxNext,
        dResid1,
        {
          wq: lp.wq.data,
          wk: lp.wk.data,
          wv: lp.wv.data,
          wo: lp.wo.data,
          bq: lp.bq.data,
          bk: lp.bk.data,
          bv: lp.bv.data,
          bo: lp.bo.data,
        },
        {
          wq: lp.wq.grad,
          wk: lp.wk.grad,
          wv: lp.wv.grad,
          wo: lp.wo.grad,
          bq: lp.bq.grad,
          bk: lp.bk.grad,
          bv: lp.bv.grad,
          bo: lp.bo.grad,
        },
        dims,
        la.attn,
        back.attnScratch,
      )

      const dxIn = back.dxIn
      layerNormBackward(
        dxIn,
        lp.ln1g.grad,
        lp.ln1b.grad,
        back.dxNext,
        lp.ln1g.data,
        rows,
        d,
        la.ln1,
      )
      for (let i = 0; i < rows * d; i++) dxIn[i] += dResid1[i]
      back.dx.set(dxIn)
    }

    // --- embeddings: scatter-add, because one token id can appear many times
    // in a batch and every occurrence contributes to the same row. ---
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const row = (b * T + t) * d
        const tokRow = act.inputs[b * T + t] * d
        const posRow = t * d
        for (let i = 0; i < d; i++) {
          this.tokenEmbed.grad[tokRow + i] += back.dx[row + i]
          this.posEmbed.grad[posRow + i] += back.dx[row + i]
        }
      }
    }

    return loss
  }

  /**
   * Run the model on a prompt and return the distribution over the next token.
   *
   * The prompt is placed at positions 0..n-1 rather than right-aligned in the
   * window. Because attention is causal, whatever sits after position n-1
   * cannot influence it, so the unused tail costs nothing and the position
   * embeddings stay aligned with "start of sequence".
   */
  predictNext(
    promptIds: number[],
    usePositional = true,
  ): { probs: Float32Array; act: Activations; readRow: number; used: number[] } {
    const T = this.cfg.contextLength
    const V = this.cfg.vocabSize
    const used = promptIds.length === 0 ? [BOS] : promptIds.slice(-T)
    const n = used.length

    const inputs = new Int32Array(T)
    inputs.fill(BOS)
    for (let i = 0; i < n; i++) inputs[i] = used[i]

    const act = this.forward(inputs, 1, T, usePositional)
    const readRow = n - 1
    const probs = new Float32Array(V)
    const off = readRow * V

    let max = -Infinity
    for (let i = 0; i < V; i++) if (act.logits[off + i] > max) max = act.logits[off + i]
    let sum = 0
    for (let i = 0; i < V; i++) {
      const e = Math.exp(act.logits[off + i] - max)
      probs[i] = e
      sum += e
    }
    for (let i = 0; i < V; i++) probs[i] /= sum

    return { probs, act, readRow, used }
  }

  /** Cross-entropy on a batch without touching gradients. */
  evalLoss(inputs: Int32Array, targets: Int32Array, B: number, T: number): number {
    const act = this.forward(inputs, B, T)
    return softmaxCrossEntropy(
      act.probs,
      null,
      act.logits,
      targets,
      B * T,
      this.cfg.vocabSize,
    )
  }
}

/**
 * Reshape a probability distribution by temperature.
 *
 * Temperature divides the log-probabilities before renormalising. Below 1 the
 * peaks sharpen towards greedy; above 1 the distribution flattens towards
 * uniform. At exactly 0 it *is* greedy, handled as a special case because
 * dividing by zero is not a distribution.
 */
export function applyTemperature(probs: Float32Array, temperature: number): Float32Array {
  const V = probs.length
  const out = new Float32Array(V)
  if (temperature <= 1e-6) {
    let best = 0
    for (let i = 1; i < V; i++) if (probs[i] > probs[best]) best = i
    out[best] = 1
    return out
  }
  let max = -Infinity
  for (let i = 0; i < V; i++) {
    const l = Math.log(Math.max(probs[i], 1e-30)) / temperature
    out[i] = l
    if (l > max) max = l
  }
  let sum = 0
  for (let i = 0; i < V; i++) {
    const e = Math.exp(out[i] - max)
    out[i] = e
    sum += e
  }
  for (let i = 0; i < V; i++) out[i] /= sum
  return out
}

/** Indices of the k highest-probability tokens, descending. */
export function topK(probs: Float32Array, k: number): number[] {
  const idx = Array.from({ length: probs.length }, (_, i) => i)
  idx.sort((a, b) => probs[b] - probs[a])
  return idx.slice(0, k)
}

/** Sample one index from a distribution, optionally restricted to the top k. */
export function sampleFrom(
  probs: Float32Array,
  rng: () => number,
  k = 0,
): number {
  let pool: number[]
  if (k > 0 && k < probs.length) {
    pool = topK(probs, k)
  } else {
    pool = Array.from({ length: probs.length }, (_, i) => i)
  }
  let total = 0
  for (const i of pool) total += probs[i]
  let r = rng() * total
  for (const i of pool) {
    r -= probs[i]
    if (r <= 0) return i
  }
  return pool[pool.length - 1]
}

export interface GenerateOptions {
  maxTokens: number
  temperature: number
  topK: number
  rng: () => number
}

/** Autoregressive generation: predict, sample, append, repeat. */
export function generate(
  model: TinyTransformer,
  promptIds: number[],
  opts: GenerateOptions,
): number[] {
  const out = [...promptIds]
  for (let i = 0; i < opts.maxTokens; i++) {
    const { probs } = model.predictNext(out)
    const shaped = applyTemperature(probs, opts.temperature)
    const next = sampleFrom(shaped, opts.rng, opts.topK)
    out.push(next)
  }
  return out
}
