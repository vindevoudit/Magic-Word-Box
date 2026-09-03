import { useMemo } from 'react'
import { useStore } from './store'
import type { Activations } from '../model/model'

export interface InferenceView {
  probs: Float32Array | null
  act: Activations | null
  /** Which row of the window holds the prediction we care about. */
  readRow: number
  /** The token ids actually fed in, after truncation to the context window. */
  used: number[]
  labels: string[]
}

/**
 * Run the current model on the current prompt.
 *
 * Recomputed whenever the prompt or the weights change. A forward pass on a
 * single sequence is well under a millisecond at these sizes, so this can sit
 * directly in render without a worker or a debounce.
 */
export function useInference(promptIds: number[], usePositional = true): InferenceView {
  const { model, vocab, modelVersion } = useStore()

  return useMemo(() => {
    if (!model || !vocab) {
      return { probs: null, act: null, readRow: 0, used: [], labels: [] }
    }
    const { probs, act, readRow, used } = model.predictNext(promptIds, usePositional)
    return {
      probs,
      act,
      readRow,
      used,
      labels: used.map((id) => vocab.tokens[id] ?? '<unk>'),
    }
    // modelVersion is the signal that the mutable model object changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, vocab, modelVersion, promptIds.join(','), usePositional])
}

/** Extract the [n, n] attention block for one layer and head from a forward pass. */
export function attentionSlice(
  act: Activations | null,
  layer: number,
  head: number,
  nHeads: number,
  T: number,
): Float32Array | null {
  if (!act || !act.layers[layer]) return null
  const probs = act.layers[layer].attn.probs
  const base = head * T * T
  void nHeads
  return probs.subarray(base, base + T * T)
}
