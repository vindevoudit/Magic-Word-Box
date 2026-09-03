import { useMemo, useState } from 'react'
import { Panel, Stage } from './Stage'
import { useStore } from '../state/store'
import { useInference } from '../state/useInference'
import { Heatmap } from '../viz/Heatmap'
import { AttentionArcs } from '../viz/AttentionArcs'
import { useInView } from '../viz/useInView'
import { useReveal } from '../viz/useTween'
import { encodePrompt } from '../model/tokenizer'
import { rampCss } from '../viz/ramp'

/** Pull the top-left n by n block out of a [T, T] head, ignoring padding. */
function submatrix(probs: Float32Array, T: number, n: number): Float32Array {
  const out = new Float32Array(n * n)
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) out[r * n + c] = probs[r * T + c]
  }
  return out
}

export function Attention() {
  const { model, vocab, modelVersion, layer, setLayer, head, setHead } = useStore()
  const [text, setText] = useState('the little lamp is on the hill')
  const [hovered, setHovered] = useState<{ row: number; col: number } | null>(null)
  const [ref, seen] = useInView<HTMLDivElement>(0.2)
  const reveal = useReveal(seen, 1400)

  const ids = useMemo(() => (vocab ? encodePrompt(text, vocab) : []), [text, vocab])
  const { act, labels, used } = useInference(ids)

  const T = model?.cfg.contextLength ?? 0
  const n = Math.min(used.length, T)

  const probs = useMemo(() => {
    if (!act || !act.layers[layer] || n < 1) return null
    const full = act.layers[layer].attn.probs
    return submatrix(full.subarray(head * T * T, (head + 1) * T * T), T, n)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [act, layer, head, T, n, modelVersion])

  const focusRow = hovered?.row ?? Math.max(0, n - 1)
  const rowWeights = useMemo(() => {
    if (!probs) return []
    return Array.from({ length: focusRow + 1 }, (_, c) => ({
      col: c,
      p: probs[focusRow * n + c],
    })).sort((a, b) => b.p - a.p)
  }, [probs, focusRow, n])

  return (
    <Stage
      id="attention"
      index={6}
      kicker="scaled dot-product attention"
      shape={`[1, ${model?.cfg.nHeads ?? 0}, ${n}, ${n}]`}
      title="Every word looks back"
      lede={
        <>
          This is the mechanism the rest of the machine is built around. Each
          word writes a <b>query</b> describing what it needs. Every earlier word
          offers a <b>key</b> describing what it has. Match them up, turn the
          matches into percentages, and mix the earlier words together in those
          proportions.
        </>
      }
    >
      <Panel wide>
        <label className="field">
          <span className="field__label eyebrow">Try your own phrase</span>
          <input
            className="field__input mono"
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
          />
        </label>

        <div className="selectors">
          <div className="selector">
            <span className="eyebrow">Layer</span>
            <div className="segmented">
              {Array.from({ length: model?.cfg.nLayers ?? 1 }, (_, i) => (
                <button
                  key={i}
                  type="button"
                  className={layer === i ? 'is-active' : ''}
                  onClick={() => setLayer(i)}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
          <div className="selector">
            <span className="eyebrow">Head</span>
            <div className="segmented">
              {Array.from({ length: model?.cfg.nHeads ?? 1 }, (_, i) => (
                <button
                  key={i}
                  type="button"
                  className={head === i ? 'is-active' : ''}
                  onClick={() => setHead(i)}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
          <p className="muted selectors__hint">
            Heads run in parallel and learn different jobs. Switch between them
            and watch the pattern change.
          </p>
        </div>
      </Panel>

      <div ref={ref}>
        <Panel
          title="The attention matrix"
          note="Read a row as: when the model was at this word, how much did it look at each earlier one? The dark triangle is not missing data - it is the rule that no word may see its own future."
          wide
        >
          {probs && n > 0 ? (
            <Heatmap
              probs={probs}
              n={n}
              labels={labels}
              reveal={reveal}
              hovered={hovered}
              onHoverCell={setHovered}
            />
          ) : (
            <p className="muted">Type a phrase the model knows to see its attention.</p>
          )}
        </Panel>
      </div>

      <Panel
        title="The same numbers, as connections"
        note="Click any word to see what it looked at. Thickness is how much weight it gave."
        wide
      >
        {probs && n > 0 ? (
          <AttentionArcs
            probs={probs}
            n={n}
            labels={labels}
            row={focusRow}
            onSelectRow={(r) => setHovered({ row: r, col: r })}
            highlightCol={hovered?.col ?? null}
          />
        ) : null}
      </Panel>

      <Panel
        title={`Where "${labels[focusRow] ?? ''}" put its attention`}
        note="These weights always add up to 100%. Attention is a budget: giving more to one word necessarily means giving less to another."
      >
        <ul className="weightlist">
          {rowWeights.map(({ col, p }) => (
            <li key={col}>
              <span className="weightlist__bar">
                <span
                  style={{ width: `${p * 100}%`, background: rampCss(Math.sqrt(p)) }}
                />
              </span>
              <span className="mono weightlist__word">{labels[col]}</span>
              <span className="mono weightlist__pct">{(p * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ul>
      </Panel>
    </Stage>
  )
}
