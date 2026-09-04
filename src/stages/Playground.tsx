import { useState } from 'react'
import { Panel, Stage } from './Stage'
import { useStore } from '../state/store'
import { useInference } from '../state/useInference'
import { ProbBars } from '../viz/ProbBars'
import { applyTemperature, sampleFrom, suppressSpecials } from '../model/model'
import { decode } from '../model/tokenizer'
import { mulberry32 } from '../model/tensor'

/**
 * Stage 9. Hands the model over.
 *
 * The temperature slider is the important control here: it makes visible that
 * a model does not "have" an answer, it has a distribution, and that sampling
 * from it is a separate choice made after the model is done thinking.
 */
export function Playground() {
  const { model, vocab, promptText, setPromptText, promptIds, temperature, setTemperature } =
    useStore()
  const [committed, setCommitted] = useState<number[]>([])
  const [lastPick, setLastPick] = useState<number | null>(null)

  const ids = [...promptIds, ...committed]
  const { probs } = useInference(ids)
  const shaped = probs ? applyTemperature(suppressSpecials(probs), temperature) : null

  const commit = (id: number) => {
    setCommitted((c) => [...c, id])
    setLastPick(id)
  }

  const generate = () => {
    if (!model || !shaped) return
    const rng = mulberry32(Date.now() % 99991)
    let seq = [...ids]
    const added: number[] = []
    for (let i = 0; i < 12; i++) {
      const p = suppressSpecials(model.predictNext(seq).probs)
      const next = sampleFrom(applyTemperature(p, temperature), rng, 12)
      seq = [...seq, next]
      added.push(next)
    }
    setCommitted((c) => [...c, ...added])
  }

  const continuation = vocab && committed.length ? decode(committed, vocab) : ''

  return (
    <Stage
      id="playground"
      index={9}
      kicker="sampling"
      shape={`[${vocab?.tokens.length ?? 0}] probabilities`}
      title="Write with it"
      lede={
        <>
          The model produces a probability for every word it knows. Picking one
          is a separate decision &mdash; and temperature is the dial that
          decides how adventurous that pick is.
        </>
      }
    >
      <Panel wide>
        <label className="field">
          <span className="field__label eyebrow">Starting words</span>
          <input
            className="field__input mono"
            value={promptText}
            spellCheck={false}
            onChange={(e) => {
              setPromptText(e.target.value)
              setCommitted([])
              setLastPick(null)
            }}
          />
        </label>

        <div className="prompt-line mono">
          <span className="prompt-line__given">{promptText}</span>
          <span className="prompt-line__added">{continuation}</span>
          <span className="prompt-line__caret" />
        </div>

        <div className="controls">
          <label className="control">
            <span className="eyebrow">
              Temperature <b className="mono accent">{temperature.toFixed(2)}</b>
            </span>
            <input
              className="slider"
              type="range"
              min={0}
              max={1.6}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />
            <span className="control__hint muted">
              {temperature < 0.15
                ? 'Always takes the single most likely word. Same output every time.'
                : temperature < 0.9
                  ? 'Favours likely words, with room for surprises.'
                  : 'Flattens the odds. More variety, less sense.'}
            </span>
          </label>

          <div className="control__buttons">
            <button type="button" className="btn btn--primary" onClick={generate}>
              Write 12 words
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                setCommitted([])
                setLastPick(null)
              }}
              disabled={committed.length === 0}
            >
              Reset
            </button>
          </div>
        </div>
      </Panel>

      <Panel
        title="What comes next"
        note="Click any word to commit to it, and the model will re-rank everything that could follow."
      >
        {shaped && vocab ? (
          <ProbBars
            probs={shaped}
            labels={vocab.tokens}
            k={8}
            onPick={commit}
            chosen={lastPick}
          />
        ) : (
          <p className="muted">Loading...</p>
        )}
      </Panel>
    </Stage>
  )
}
