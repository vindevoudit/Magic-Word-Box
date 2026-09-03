import { useEffect, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { applyTemperature, sampleFrom, topK } from '../model/model'
import { mulberry32 } from '../model/tensor'
import { prefersReducedMotion } from '../viz/useTween'

interface Word {
  id: number
  text: string
  alternatives: { text: string; p: number }[]
}

/**
 * The thesis, running.
 *
 * Rather than describe next-word prediction, the hero performs it: the real
 * shipped model writes a line one word at a time, and the words it nearly
 * chose fade out beside each pick. Everything below is machinery for making
 * this guess better.
 */
export function Hero() {
  const { model, vocab, modelVersion } = useStore()
  const [words, setWords] = useState<Word[]>([])
  const timerRef = useRef<number | undefined>(undefined)

  useEffect(() => {
    if (!model || !vocab) return
    const rng = mulberry32(Date.now() % 100000)
    let ids: number[] = []
    let out: Word[] = []
    const still = prefersReducedMotion()

    const stepOnce = () => {
      const { probs } = model.predictNext(ids)
      const shaped = applyTemperature(probs, 0.75)
      const next = sampleFrom(shaped, rng, 8)
      const alts = topK(probs, 3)
        .filter((id) => id !== next)
        .slice(0, 2)
        .map((id) => ({ text: label(vocab.tokens[id]), p: probs[id] }))

      ids = [...ids, next]
      out = [...out, { id: next, text: label(vocab.tokens[next]), alternatives: alts }]

      if (ids.length > 18) {
        ids = []
        out = []
      }
      setWords(out)
    }

    if (still) {
      for (let i = 0; i < 12; i++) stepOnce()
      return
    }

    timerRef.current = window.setInterval(stepOnce, 620)
    return () => window.clearInterval(timerRef.current)
  }, [model, vocab, modelVersion])

  return (
    <header className="hero">
      <p className="eyebrow hero__eyebrow">A language model, from nothing</p>
      <h1 className="hero__title">
        Magic
        <br />
        Word Box
      </h1>
      <p className="hero__lede">
        A language model does one thing: guess the next word. Everything else is
        machinery for guessing better. Train one on your own text, and watch the
        machinery work.
      </p>

      <div className="hero__demo" aria-live="off">
        <span className="hero__demo-label eyebrow">Writing, one word at a time</span>
        <p className="hero__stream">
          {words.map((w, i) => (
            <span className={`hw${i === words.length - 1 ? ' hw--new' : ''}`} key={i}>
              <span className="hw__word">{w.text}</span>
              <span className="hw__alts" aria-hidden="true">
                {w.alternatives.map((a, j) => (
                  <span key={j} style={{ opacity: 0.15 + a.p * 0.6 }}>
                    {a.text}
                  </span>
                ))}
              </span>
            </span>
          ))}
          <span className="hw__caret" aria-hidden="true" />
        </p>
        {words.length === 0 ? (
          <p className="hero__loading mono">loading the starter model...</p>
        ) : null}
      </div>

      <a className="hero__scroll" href="#guess">
        Open the box
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path
            d="M12 4v15m0 0l-6-6m6 6l6-6"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
        </svg>
      </a>
    </header>
  )
}

function label(tok: string | undefined): string {
  if (!tok) return '?'
  if (tok === '<br>') return '/'
  if (tok === '<start>') return ''
  return tok
}
