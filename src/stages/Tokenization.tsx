import { useMemo, useState } from 'react'
import { Panel, Stage } from './Stage'
import { useStore } from '../state/store'
import { buildVocab, tokenize } from '../model/tokenizer'
import { TokenChips } from '../viz/TokenChips'
import { VocabGrid } from '../viz/VocabGrid'
import { useInView } from '../viz/useInView'
import { useReveal } from '../viz/useTween'

/**
 * Stage 3. Text becomes numbers.
 *
 * Two ideas share this stage because they are the same idea: the model never
 * sees words, only ids, and it never sees the whole corpus, only fixed windows
 * cut out of it.
 */
export function Tokenization() {
  const { corpus, preset } = useStore()
  const [ref, seen] = useInView<HTMLDivElement>(0.3)
  const reveal = useReveal(seen, 1200)
  const [activeId, setActiveId] = useState<number | null>(null)
  const [windowStart, setWindowStart] = useState(0)

  const { tokens, vocab, ids } = useMemo(() => {
    const toks = tokenize(corpus)
    const v = buildVocab(toks, preset.maxVocab)
    return {
      tokens: toks,
      vocab: v,
      ids: toks.map((t) => v.ids.get(t) ?? 1),
    }
  }, [corpus, preset.maxVocab])

  const T = preset.contextLength
  const maxStart = Math.max(0, tokens.length - T - 1)
  const start = Math.min(windowStart, maxStart)

  return (
    <Stage
      id="tokens"
      index={3}
      kicker="text to ids"
      shape={`[${preset.batchSize}, ${T}]`}
      title="Words become numbers"
      lede={
        <>
          The model cannot read. Every word is replaced by its position in a
          vocabulary list, and from here on the text is just a stream of
          integers.
        </>
      }
    >
      <div ref={ref}>
        <Panel
          title="Your text, split into tokens"
          note="Punctuation becomes its own token, and a line break is a token too, which is how the model learns where lines end."
        >
          <TokenChips
            tokens={tokens}
            ids={ids}
            reveal={reveal}
            max={120}
            onHover={(i) => setActiveId(i == null ? null : ids[i])}
            activeIndex={null}
          />
        </Panel>
      </div>

      <Panel
        title={`The vocabulary: ${vocab.tokens.length} tokens`}
        note={
          vocab.unkOccurrences > 0
            ? `${vocab.unkOccurrences.toLocaleString()} rare word occurrences did not make the cut and became <unk>. The model can never say them.`
            : 'Every word in your text fit inside the vocabulary, so nothing was lost.'
        }
      >
        <VocabGrid
          tokens={vocab.tokens}
          counts={vocab.counts}
          activeId={activeId}
          onHover={setActiveId}
        />
      </Panel>

      <Panel
        title="Cutting the training examples"
        note={`Every window is one training example, and each of its ${T} positions is a separate guess. Drag to slide the window along the corpus.`}
      >
        <div className="window-strip mono">
          {tokens.slice(start, start + T + 1).map((t, i) => (
            <span
              key={i}
              className={`wtok${i === T ? ' wtok--target' : ' wtok--input'}`}
            >
              {t === '<br>' ? '\u21b5' : t}
            </span>
          ))}
        </div>
        <p className="window-legend">
          <span className="key key--input" /> what the model sees
          <span className="key key--target" /> what it must predict
        </p>
        <input
          className="slider"
          type="range"
          min={0}
          max={maxStart}
          value={start}
          aria-label="Slide the context window along the corpus"
          onChange={(e) => setWindowStart(Number(e.target.value))}
        />
        <p className="muted mono window-count">
          window {start + 1} of {maxStart + 1}
        </p>
      </Panel>
    </Stage>
  )
}
