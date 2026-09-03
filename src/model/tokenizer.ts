/**
 * Word-level tokenizer.
 *
 * Real LLMs use subword schemes like BPE, which trade legibility for coverage.
 * This site keeps whole words as tokens because the entire experience rests on
 * the reader recognising what the model just predicted. "the cat sat on the
 * ___ -> mat" teaches something; a prediction of "##at" does not.
 *
 * The cost is an open vocabulary, handled the same way early neural LMs did:
 * keep the most frequent words and map the long tail to a single unknown token.
 */

export const BOS = 0
export const UNK = 1
export const NL = 2
export const SPECIAL_TOKENS = ['<start>', '<unk>', '<br>'] as const

/** Words, contractions kept whole, and punctuation as standalone tokens. */
const TOKEN_RE = /[a-z0-9]+(?:['\u2019][a-z]+)*|[.,!?;:"()\u2014-]/g

const PUNCT_NO_SPACE_BEFORE = new Set([',', '.', '!', '?', ';', ':', ')'])

export interface Vocab {
  /** Token string by id. */
  tokens: string[]
  /** id by token string. */
  ids: Map<string, number>
  /** Occurrence count in the corpus, by id. Drives the frequency grid in stage 3. */
  counts: number[]
  /** How many distinct words existed before the frequency cap. */
  distinctWords: number
  /** How many token occurrences got mapped to <unk>. */
  unkOccurrences: number
}

/** Split raw text into token strings, with <br> standing in for a line break. */
export function tokenize(text: string): string[] {
  const out: string[] = []
  const lines = text.toLowerCase().split(/\r?\n/)
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const matches = line.match(TOKEN_RE)
    if (matches) out.push(...matches)
    // Blank lines collapse: one <br> per line break, never a run of them.
    if (i < lines.length - 1 && out.length > 0 && out[out.length - 1] !== '<br>') {
      out.push('<br>')
    }
  }
  while (out.length > 0 && out[out.length - 1] === '<br>') out.pop()
  return out
}

/** Build a frequency-capped vocabulary from token strings. */
export function buildVocab(tokenStrings: string[], maxVocab: number): Vocab {
  const freq = new Map<string, number>()
  for (const t of tokenStrings) {
    if (t === '<br>') continue
    freq.set(t, (freq.get(t) ?? 0) + 1)
  }

  const sorted = [...freq.entries()].sort((a, b) => {
    // Frequency first, then alphabetical, so the vocabulary is deterministic
    // and the grid in stage 3 does not reshuffle between runs.
    if (b[1] !== a[1]) return b[1] - a[1]
    return a[0] < b[0] ? -1 : 1
  })

  const keep = sorted.slice(0, Math.max(1, maxVocab - SPECIAL_TOKENS.length))

  const tokens = [...SPECIAL_TOKENS] as string[]
  const ids = new Map<string, number>()
  SPECIAL_TOKENS.forEach((t, i) => ids.set(t, i))
  const counts: number[] = [0, 0, 0]

  for (const [word, count] of keep) {
    ids.set(word, tokens.length)
    tokens.push(word)
    counts.push(count)
  }

  let unkOccurrences = 0
  for (const [word, count] of sorted.slice(keep.length)) {
    void word
    unkOccurrences += count
  }
  counts[UNK] = unkOccurrences
  counts[NL] = tokenStrings.reduce((n, t) => (t === '<br>' ? n + 1 : n), 0)

  return { tokens, ids, counts, distinctWords: freq.size, unkOccurrences }
}

/** Map token strings to ids, with <start> prepended so position 0 has context. */
export function encode(tokenStrings: string[], vocab: Vocab): Int32Array {
  const out = new Int32Array(tokenStrings.length + 1)
  out[0] = BOS
  for (let i = 0; i < tokenStrings.length; i++) {
    out[i + 1] = vocab.ids.get(tokenStrings[i]) ?? UNK
  }
  return out
}

/** Render ids back to readable text, respecting punctuation spacing. */
export function decode(ids: ArrayLike<number>, vocab: Vocab): string {
  let out = ''
  for (let i = 0; i < ids.length; i++) {
    const tok = vocab.tokens[ids[i]] ?? '<unk>'
    if (tok === '<start>') continue
    if (tok === '<br>') {
      out += '\n'
      continue
    }
    const needsSpace =
      out.length > 0 && !out.endsWith('\n') && !PUNCT_NO_SPACE_BEFORE.has(tok)
    out += (needsSpace ? ' ' : '') + tok
  }
  return out
}

export interface Dataset {
  ids: Int32Array
  vocab: Vocab
  /** Number of training windows available at context length T. */
  windowCount: number
}

export function buildDataset(text: string, maxVocab: number, contextLength: number): Dataset {
  const strings = tokenize(text)
  const vocab = buildVocab(strings, maxVocab)
  const ids = encode(strings, vocab)
  return {
    ids,
    vocab,
    windowCount: Math.max(0, ids.length - contextLength),
  }
}

/**
 * Fill one training batch of `[B, T]` inputs and `[B, T]` targets.
 *
 * Targets are the inputs shifted one position left: predicting every position
 * in the window at once, rather than only the last, is what makes each forward
 * pass yield T training signals instead of one.
 */
export function sampleBatch(
  inputs: Int32Array,
  targets: Int32Array,
  data: Int32Array,
  B: number,
  T: number,
  rng: () => number,
): void {
  const maxStart = data.length - T - 1
  for (let b = 0; b < B; b++) {
    const start = maxStart > 0 ? Math.floor(rng() * (maxStart + 1)) : 0
    for (let t = 0; t < T; t++) {
      inputs[b * T + t] = data[start + t] ?? BOS
      targets[b * T + t] = data[start + t + 1] ?? BOS
    }
  }
}

/**
 * Rebuild a Vocab from just its token list.
 *
 * The worker and the shipped model both send tokens as a plain array, since a
 * Map does not survive structured cloning in a useful shape. Counts are lost,
 * which is fine: only the encode/decode direction is needed after training.
 */
export function vocabFromTokens(tokens: string[]): Vocab {
  const ids = new Map<string, number>()
  tokens.forEach((t, i) => ids.set(t, i))
  return {
    tokens,
    ids,
    counts: new Array(tokens.length).fill(0),
    distinctWords: tokens.length,
    unkOccurrences: 0,
  }
}

/** Encode free text typed into the playground, dropping unknown words to <unk>. */
export function encodePrompt(text: string, vocab: Vocab): number[] {
  return tokenize(text).map((t) => vocab.ids.get(t) ?? UNK)
}
