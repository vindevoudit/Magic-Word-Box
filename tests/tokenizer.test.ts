import { describe, expect, it } from 'vitest'
import {
  BOS,
  UNK,
  buildDataset,
  buildVocab,
  decode,
  encode,
  sampleBatch,
  tokenize,
} from '../src/model/tokenizer'
import { mulberry32 } from '../src/model/tensor'

describe('tokenize', () => {
  it('lowercases and splits punctuation into its own tokens', () => {
    expect(tokenize('The cat, it sat!')).toEqual(['the', 'cat', ',', 'it', 'sat', '!'])
  })

  it('keeps contractions whole', () => {
    expect(tokenize("don't stop")).toEqual(["don't", 'stop'])
  })

  it('marks line breaks and collapses blank lines', () => {
    expect(tokenize('a\nb')).toEqual(['a', '<br>', 'b'])
    expect(tokenize('a\n\n\nb')).toEqual(['a', '<br>', 'b'])
  })

  it('does not leave a trailing line break', () => {
    expect(tokenize('a\n')).toEqual(['a'])
  })
})

describe('vocabulary', () => {
  it('orders by frequency and is deterministic on ties', () => {
    const v = buildVocab(tokenize('b b b a a c'), 100)
    expect(v.tokens.slice(0, 3)).toEqual(['<start>', '<unk>', '<br>'])
    expect(v.tokens[3]).toBe('b')
    expect(v.tokens[4]).toBe('a')
    expect(v.tokens[5]).toBe('c')
  })

  it('maps the long tail to <unk> when capped', () => {
    const v = buildVocab(tokenize('aa aa bb cc dd'), 5)
    expect(v.tokens.length).toBe(5)
    expect(v.ids.has('aa')).toBe(true)
    const ids = encode(tokenize('aa zz'), v)
    expect(ids[0]).toBe(BOS)
    expect(ids[1]).toBe(v.ids.get('aa'))
    expect(ids[2]).toBe(UNK)
  })

  it('counts distinct words before the cap was applied', () => {
    expect(buildVocab(tokenize('a b c d e'), 4).distinctWords).toBe(5)
  })
})

describe('encode and decode', () => {
  it('prepends a start token', () => {
    const v = buildVocab(tokenize('hello world'), 50)
    expect(encode(tokenize('hello world'), v)[0]).toBe(BOS)
  })

  it('round-trips text, restoring punctuation spacing', () => {
    const text = 'the cat sat, and it slept.'
    const v = buildVocab(tokenize(text), 50)
    expect(decode(encode(tokenize(text), v), v)).toBe(text)
  })

  it('renders line breaks back as newlines', () => {
    const text = 'one two\nthree four'
    const v = buildVocab(tokenize(text), 50)
    expect(decode(encode(tokenize(text), v), v)).toBe(text)
  })
})

describe('batching', () => {
  it('produces targets shifted one position from inputs', () => {
    const data = buildDataset('a b c d e f g h i j k l', 50, 4)
    const B = 4
    const T = 4
    const inputs = new Int32Array(B * T)
    const targets = new Int32Array(B * T)
    sampleBatch(inputs, targets, data.ids, B, T, mulberry32(1))
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T - 1; t++) {
        expect(targets[b * T + t]).toBe(inputs[b * T + t + 1])
      }
    }
  })

  it('reports how many windows a corpus yields', () => {
    // 5 words + 1 start token = 6 ids, minus context 3 = 3 windows.
    expect(buildDataset('a b c d e', 50, 3).windowCount).toBe(3)
  })
})
