import { useMemo } from 'react'
import { Panel, Stage, Stat } from './Stage'
import { useStore } from '../state/store'
import { CORPORA } from '../data/corpora'
import { buildVocab, tokenize } from '../model/tokenizer'
import { LIMITS, VOCAB_CEILING } from '../model/presets'

/**
 * Stage 2. Where the reader supplies the only thing the model will ever know.
 *
 * The warnings here matter more than they look: a word-level model this small
 * needs repetition, and a reader who pastes three sentences of prose and gets
 * mush will conclude the site is broken rather than that the model is tiny.
 */
export function Corpus() {
  const { corpus, setCorpus } = useStore()

  const stats = useMemo(() => {
    const toks = tokenize(corpus)
    const words = toks.filter((t) => t !== '<br>')
    const unique = new Set(words)
    const repetition = words.length / Math.max(1, unique.size)
    // Coverage is the number that actually predicts whether generation will be
    // readable. Anything outside the vocabulary becomes <unk>, and <unk> is a
    // word the model can never say back.
    const vocab = buildVocab(toks, VOCAB_CEILING)
    const unkPct = (vocab.unkOccurrences / Math.max(1, toks.length)) * 100
    return {
      vocabSize: vocab.tokens.length,
      unkPct,
      chars: corpus.length,
      words: words.length,
      unique: unique.size,
      lines: corpus.split(/\r?\n/).filter((l) => l.trim()).length,
      repetition,
    }
  }, [corpus])

  const tooShort = stats.words < LIMITS.minCorpusWords
  const lossy = !tooShort && stats.unkPct > 2
  const thin = !tooShort && !lossy && stats.repetition < 2.2
  const slow = stats.vocabSize > 1200

  return (
    <Stage
      id="corpus"
      index={2}
      kicker="the training data"
      shape={`${stats.words.toLocaleString()} words`}
      title="Give it something to read"
      lede={
        <>
          The model starts knowing nothing at all &mdash; not grammar, not
          spelling, not that words exist. This text is the only thing it will
          ever learn from.
        </>
      }
    >
      <Panel wide>
        <div className="corpus__picker">
          {CORPORA.map((c) => (
            <button
              key={c.id}
              type="button"
              className={`corpus__preset${corpus === c.text ? ' is-active' : ''}`}
              onClick={() => setCorpus(c.text)}
            >
              <b>{c.name}</b>
              <span>{c.blurb}</span>
            </button>
          ))}
        </div>

        <label className="visually-hidden" htmlFor="corpus-text">
          Training text
        </label>
        <textarea
          id="corpus-text"
          className="corpus__text mono"
          value={corpus}
          spellCheck={false}
          onChange={(e) => setCorpus(e.target.value.slice(0, LIMITS.maxCorpusChars))}
        />

        <div className="stats">
          <Stat label="words" value={stats.words} />
          <Stat label="different words" value={stats.unique} />
          <Stat label="lines" value={stats.lines} />
          <Stat
            label="repetition"
            value={`${stats.repetition.toFixed(1)}x`}
            sub="times the average word recurs"
          />
          <Stat
            label="vocabulary"
            value={stats.vocabSize}
            sub={`covers ${(100 - stats.unkPct).toFixed(1)}% of the text`}
          />
        </div>

        {tooShort ? (
          <p className="notice notice--warn">
            That is {stats.words} words. Below about {LIMITS.minCorpusWords} there is
            not enough signal for the model to find any pattern &mdash; try one of the
            presets above, or paste more.
          </p>
        ) : lossy ? (
          <p className="notice notice--warn">
            This text uses {stats.unique.toLocaleString()} different words, more than
            the {VOCAB_CEILING.toLocaleString()}-word limit, so{' '}
            <b>{stats.unkPct.toFixed(0)}% of it</b> becomes an unknown token the
            model can never say back. Trim it to a few thousand words, or pick a
            passage that reuses its vocabulary.
          </p>
        ) : thin ? (
          <p className="notice">
            This text has little repetition, so a model this small will struggle. It
            works best on writing with refrains, fixed phrasing or a template &mdash;
            song lyrics, recipes, logs.
          </p>
        ) : (
          <p className="notice notice--ok">
            Good shape for a small model: the average word appears{' '}
            {stats.repetition.toFixed(1)} times, so there are patterns to find.
          </p>
        )}

        {slow && !tooShort ? (
          <p className="notice">
            A {stats.vocabSize.toLocaleString()}-word vocabulary makes the model
            several times larger, since it has to score every one of those words at
            every position. Training will take noticeably longer than the few
            seconds the presets quote.
          </p>
        ) : null}
      </Panel>
    </Stage>
  )
}
