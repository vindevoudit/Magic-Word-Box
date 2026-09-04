/**
 * Training presets.
 *
 * Sizes are chosen against a wall-clock budget, not a quality target: the whole
 * experience rests on the visitor watching training finish, so a preset that
 * takes three minutes is the wrong preset however good the loss curve looks.
 */

export interface Preset {
  id: string
  name: string
  blurb: string
  dModel: number
  nLayers: number
  nHeads: number
  contextLength: number
  batchSize: number
  steps: number
  lr: number
  maxVocab: number
}

/**
 * Largest vocabulary any preset will build.
 *
 * This is a ceiling, not a target: the built-in poems produce about 46 tokens
 * and are unaffected. It matters for pasted prose, where the previous limit of
 * 600 pushed roughly one word in nine into <unk> - enough to make <unk> the
 * second most frequent token in the corpus, and so the model's best guess
 * almost everywhere.
 *
 * The cost is real and roughly linear: the token embedding and the output head
 * are both [vocab, width], so a 2700-word vocabulary is around 20 seconds of
 * training rather than four. The training stage shows a live estimate rather
 * than hiding that.
 */
export const VOCAB_CEILING = 4000

export const PRESETS: Preset[] = [
  {
    id: 'quick',
    name: 'Quick',
    blurb: 'One block, two heads. Finishes in a few seconds.',
    dModel: 32,
    nLayers: 1,
    nHeads: 2,
    contextLength: 8,
    batchSize: 16,
    steps: 400,
    lr: 4e-3,
    maxVocab: VOCAB_CEILING,
  },
  {
    id: 'standard',
    name: 'Standard',
    blurb: 'Two blocks, four heads. Better text, and more to look at per layer.',
    dModel: 48,
    nLayers: 2,
    nHeads: 4,
    contextLength: 12,
    batchSize: 16,
    steps: 800,
    lr: 3e-3,
    maxVocab: VOCAB_CEILING,
  },
  {
    id: 'big',
    name: 'Patient',
    blurb: 'Three blocks, four heads, a longer window. Takes the longest.',
    dModel: 64,
    nLayers: 3,
    nHeads: 4,
    contextLength: 16,
    batchSize: 16,
    steps: 1200,
    lr: 2.5e-3,
    maxVocab: VOCAB_CEILING,
  },
]

export const DEFAULT_PRESET = PRESETS[0]

/** What ships in public/pretrained.json, so the site has a live model on load. */
export const SHIPPED_PRESET = { ...PRESETS[1], steps: 2500 }

export const LIMITS = {
  dModel: { min: 16, max: 64, step: 8 },
  nLayers: { min: 1, max: 3, step: 1 },
  nHeads: [1, 2, 4, 8],
  contextLength: { min: 4, max: 16, step: 2 },
  steps: { min: 100, max: 2000, step: 100 },
  maxCorpusChars: 200_000,
  minCorpusWords: 120,
}
