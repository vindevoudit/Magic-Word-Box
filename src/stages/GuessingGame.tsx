import { useState } from 'react'
import { Panel, Stage } from './Stage'
import { useStore } from '../state/store'
import { useInference } from '../state/useInference'
import { ProbBars } from '../viz/ProbBars'
import { encodePrompt } from '../model/tokenizer'
import { suppressSpecials } from '../model/model'

// Chosen because the shipped model genuinely splits three ways here
// (bird / lamp / boat, all near a third). A prompt it answers with 99%
// confidence would undercut the whole point of the stage.
const PROMPT = 'the little'

/**
 * Stage 1. Establishes the frame the whole site depends on: prediction is a
 * distribution, not an answer. The reader guesses first, so that the model's
 * distribution lands as a comparison rather than as trivia.
 */
export function GuessingGame() {
  const { vocab } = useStore()
  const [guess, setGuess] = useState('')
  const [submitted, setSubmitted] = useState(false)

  const ids = vocab ? encodePrompt(PROMPT, vocab) : []
  const { probs: raw, labels } = useInference(ids)
  const probs = raw ? suppressSpecials(raw) : null
  void labels

  const guessId = vocab && guess ? (vocab.ids.get(guess.trim().toLowerCase()) ?? -1) : -1
  const guessProb = probs && guessId >= 0 ? probs[guessId] : null

  return (
    <Stage
      id="guess"
      index={1}
      kicker="next-word prediction"
      title="Everything is a guess"
      lede={
        <>
          The model on this page has read exactly one short poem, and nothing
          else in its life. Finish the line for it. There is no single right
          answer &mdash; and that is the point, because the model does not
          produce one either. It produces a spread.
        </>
      }
    >
      <Panel>
        <p className="cloze">
          <span>{PROMPT}</span>
          <input
            className="cloze__input mono"
            value={guess}
            placeholder="?"
            aria-label="Your guess for the next word"
            onChange={(e) => setGuess(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') setSubmitted(true)
            }}
          />
        </p>
        <button
          className="btn"
          type="button"
          onClick={() => setSubmitted(true)}
          disabled={!guess.trim()}
        >
          Compare with the model
        </button>
        <p className="muted footnote cloze__hint">
          The poem it read is about a lamp, a boat and a bird.
        </p>

        {submitted ? (
          <div className="cloze__result">
            {guessProb != null && guessProb > 0 ? (
              <p>
                The model gives <b className="mono">{guess.trim().toLowerCase()}</b> a
                probability of{' '}
                <b className="mono accent">{(guessProb * 100).toFixed(1)}%</b>. Here is
                its full shortlist.
              </p>
            ) : (
              <p>
                <b className="mono">{guess.trim().toLowerCase()}</b> is not in this
                model&rsquo;s vocabulary at all. It can only ever say words it has
                actually read &mdash; about fifty of them. Here is what it does
                consider.
              </p>
            )}
          </div>
        ) : null}
      </Panel>

      <Panel
        title="What the model says"
        note="Real numbers from the model running on this page. Notice that the top three are almost tied: the model is not confused, it is correctly reporting that three continuations are about equally good."
      >
        {probs && vocab ? (
          <ProbBars probs={probs} labels={vocab.tokens} k={6} />
        ) : (
          <p className="muted">Loading the starter model...</p>
        )}
      </Panel>
    </Stage>
  )
}
