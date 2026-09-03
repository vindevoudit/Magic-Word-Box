import { useMemo, useState } from 'react'
import { Panel, Stage } from './Stage'
import { useStore } from '../state/store'
import { EmbeddingMap } from '../viz/EmbeddingMap'
import { pca2d } from '../model/pca'
import { rampCss } from '../viz/ramp'

const MAX_PLOTTED = 120

/**
 * Stage 4. Ids become vectors.
 *
 * The key idea is that these numbers are not given, they are *learned*. The map
 * at the bottom is the same one the training stage animates, so a reader who
 * sees it settled here recognises it moving later.
 */
export function Embeddings() {
  const { model, vocab, modelVersion, training } = useStore()
  const [pick, setPick] = useState<number | null>(null)

  const d = model?.cfg.dModel ?? 0

  const { coords, ids } = useMemo(() => {
    if (!model || !vocab) return { coords: null, ids: null }
    // Live training supplies its own projection, kept stable frame to frame.
    if (training.embedding) return training.embedding
    const count = Math.min(MAX_PLOTTED, vocab.tokens.length - 3)
    if (count < 2) return { coords: null, ids: null }
    const plotted = Int32Array.from({ length: count }, (_, i) => i + 3)
    const rows = new Float32Array(count * d)
    for (let i = 0; i < count; i++) {
      rows.set(model.tokenEmbed.data.subarray(plotted[i] * d, plotted[i] * d + d), i * d)
    }
    return { coords: pca2d(rows, count, d).coords, ids: plotted }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, vocab, modelVersion, training.embedding, d])

  const chosen = pick ?? (vocab ? Math.min(4, vocab.tokens.length - 1) : 0)
  const vector = model ? model.tokenEmbed.data.subarray(chosen * d, chosen * d + d) : null
  const vMax = vector ? Math.max(...Array.from(vector, Math.abs), 1e-6) : 1

  return (
    <Stage
      id="embeddings"
      index={4}
      kicker="ids to vectors"
      shape={`[${vocab?.tokens.length ?? 0}, ${d}]`}
      title="Numbers become meaning"
      lede={
        <>
          An id like <span className="mono">42</span> says nothing about a word.
          So every token gets a list of {d} numbers instead &mdash; and those
          numbers are not written by anyone. The model invents them, and adjusts
          them every training step.
        </>
      }
    >
      <Panel
        title="One word, as the model stores it"
        note="Nothing here is human-readable, and it is not supposed to be. What matters is how these vectors sit relative to each other."
      >
        <div className="wordpicker">
          {vocab?.tokens.slice(3, 18).map((t, i) => (
            <button
              key={t}
              type="button"
              className={`pill${chosen === i + 3 ? ' is-active' : ''}`}
              onClick={() => setPick(i + 3)}
            >
              {t === '<br>' ? '\u21b5' : t}
            </button>
          ))}
        </div>
        {vector ? (
          <div className="vector" aria-label="The embedding vector for the selected word">
            {Array.from(vector, (v, i) => (
              <span
                key={i}
                className="vector__cell"
                style={{ background: rampCss((v / vMax + 1) / 2) }}
                title={v.toFixed(4)}
              />
            ))}
          </div>
        ) : null}
        <p className="muted mono vector__caption">
          {d} numbers, one word. Hover a cell for its value.
        </p>
      </Panel>

      <Panel
        title="Every word, on one map"
        note={
          training.status === 'done' && training.isOwnModel
            ? 'This is your trained model. Words that get used the same way have been pulled together.'
            : 'Flattened from many dimensions down to two, so the axes mean nothing on their own. Only the grouping matters. Train a model in stage 8 and watch this rearrange itself.'
        }
      >
        <EmbeddingMap
          coords={coords}
          ids={ids}
          labels={vocab?.tokens ?? []}
          highlight={new Set([chosen])}
        />
      </Panel>
    </Stage>
  )
}
