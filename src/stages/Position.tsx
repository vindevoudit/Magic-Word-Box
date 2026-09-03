import { useMemo, useState } from 'react'
import { Panel, Stage } from './Stage'
import { useStore } from '../state/store'
import { encodePrompt } from '../model/tokenizer'
import { rampCss } from '../viz/ramp'

const ORIGINAL = 'the little lamp is on the hill and'
const SHUFFLED = 'hill the on lamp little the is and'

/**
 * Stage 5. Why position has to be supplied separately.
 *
 * The demonstration is measured at the *first attention layer*, because that is
 * where the property is exactly true: one attention layer with no position
 * information is permutation-invariant over the tokens it can see.
 *
 * It is deliberately not measured on the final prediction. Stacked layers are
 * not order-blind even without position embeddings - each layer reads prefixes,
 * and prefixes change when the order changes, so causal masking leaks position
 * information all by itself. That subtlety is called out on the page rather
 * than hidden, because a reader who tests the claim on the full model would
 * otherwise catch the site being sloppy.
 *
 * Note also that only the *prefix* is shuffled. Attention is permutation-
 * equivariant, not invariant: the token doing the asking still matters, so both
 * phrases must end on the same word.
 */
export function Position() {
  const { model, vocab, modelVersion } = useStore()
  const [usePos, setUsePos] = useState(false)

  const result = useMemo(() => {
    if (!model || !vocab) return null
    const d = model.cfg.dModel
    const T = model.cfg.contextLength

    const attnOut = (ids: number[]) => {
      const n = Math.min(ids.length, T)
      const inputs = new Int32Array(T).fill(0)
      for (let i = 0; i < n; i++) inputs[i] = ids[i]
      const act = model.forward(inputs, 1, T, usePos)
      return act.layers[0].attnOut.slice((n - 1) * d, n * d)
    }

    const a = attnOut(encodePrompt(ORIGINAL, vocab))
    const b = attnOut(encodePrompt(SHUFFLED, vocab))
    let maxDiff = 0
    for (let i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]))
    const scale = Math.max(...Array.from(a, Math.abs), ...Array.from(b, Math.abs), 1e-6)
    // The difference is what makes the claim visible. Two similar-looking
    // strips prove nothing to the eye; a strip that is uniformly black does.
    const diff = new Float32Array(a.length)
    for (let i = 0; i < a.length; i++) diff[i] = Math.abs(a[i] - b[i])
    return { a, b, diff, maxDiff, scale }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, vocab, modelVersion, usePos])

  const identical = result != null && result.maxDiff < 1e-5

  return (
    <Stage
      id="position"
      index={5}
      kicker="order information"
      shape={`[${model?.cfg.contextLength ?? 0}, ${model?.cfg.dModel ?? 0}]`}
      title="Attention cannot tell word order"
      lede={
        <>
          Attention compares every word against every other word, but comparison
          has no sense of sequence. On its own it sees the earlier words as a
          bag: which words are present, not what order they arrived in. Order
          has to be handed to it separately.
        </>
      }
    >
      <Panel wide>
        <div className="toggle-row">
          <button
            type="button"
            className={`switch${usePos ? ' is-on' : ''}`}
            role="switch"
            aria-checked={usePos}
            onClick={() => setUsePos((v) => !v)}
          >
            <span className="switch__track">
              <span className="switch__thumb" />
            </span>
            Position embeddings {usePos ? 'on' : 'off'}
          </button>
          <p className="muted toggle-row__hint">
            {usePos
              ? 'Every slot now adds its own learned vector, so the same word in a different place is a different input.'
              : 'Both phrases are being read as an unordered bag of words.'}
          </p>
        </div>

        <div className="compare">
          <div className="compare__side">
            <p className="eyebrow">As written</p>
            <p className="compare__text mono">{ORIGINAL}</p>
          </div>
          <div className="compare__side">
            <p className="eyebrow">Same words, prefix shuffled</p>
            <p className="compare__text mono">{SHUFFLED}</p>
          </div>
        </div>

        <p className="eyebrow" style={{ marginTop: 'var(--gap-m)' }}>
          What attention produced for the final word
        </p>
        {result ? (
          <div className="posvectors">
            {[
              { label: 'as written', vec: result.a },
              { label: 'shuffled', vec: result.b },
            ].map((row) => (
              <div className="posvectors__row" key={row.label}>
                <span className="posvectors__label mono">{row.label}</span>
                <div className="vector">
                  {Array.from(row.vec, (v, j) => (
                    <span
                      key={j}
                      className="vector__cell"
                      style={{ background: rampCss((v / result.scale + 1) / 2) }}
                      title={v.toFixed(5)}
                    />
                  ))}
                </div>
              </div>
            ))}
            <div className="posvectors__row posvectors__row--diff">
              <span className="posvectors__label mono">difference</span>
              <div className="vector">
                {Array.from(result.diff, (v, j) => (
                  <span
                    key={j}
                    className="vector__cell"
                    style={{
                      background:
                        v < 1e-5 ? '#0d1220' : rampCss(Math.sqrt(v / result.scale)),
                    }}
                    title={v.toExponential(2)}
                  />
                ))}
              </div>
            </div>
          </div>
        ) : null}

        <p className={`notice ${identical ? 'notice--warn' : 'notice--ok'}`}>
          {identical ? (
            <>
              The two are identical. Largest difference anywhere in the vector:{' '}
              <b className="mono">{result?.maxDiff.toExponential(1)}</b> &mdash;
              which is just rounding. Attention produced exactly the same result
              for two sentences that mean different things.
            </>
          ) : (
            <>
              Now they differ, by up to{' '}
              <b className="mono">{result?.maxDiff.toFixed(3)}</b>. That
              difference is entirely the position vectors doing their job.
            </>
          )}
        </p>

        <p className="muted footnote">
          Two details this demonstration is careful about. Only the words{' '}
          <em>before</em> the last one are shuffled, because the word doing the
          asking always affects the answer. And the measurement is taken at the
          first attention layer, where the effect is exact &mdash; stack a
          second layer and a little order information leaks back in anyway,
          since each layer reads the ones before it and those change when the
          order changes.
        </p>
      </Panel>
    </Stage>
  )
}
