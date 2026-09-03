import { Panel, Stage, Stat } from './Stage'
import { useStore } from '../state/store'
import { LossChart } from '../viz/LossChart'
import { EmbeddingMap } from '../viz/EmbeddingMap'
import { PRESETS } from '../model/presets'

/**
 * Stage 8. The payoff.
 *
 * Everything above is static until this runs. The loss curve, the sample text
 * and the embedding map all update from the same worker messages, so the
 * reader can watch one number falling cause the other two to change.
 */
export function Training() {
  const { preset, setPreset, training, startTraining, cancelTraining, vocab } = useStore()
  const busy = training.status === 'training'
  const pct = training.totalSteps
    ? Math.round((training.step / training.totalSteps) * 100)
    : 0

  return (
    <Stage
      id="training"
      index={8}
      kicker="gradient descent"
      shape={`${training.totalSteps || preset.steps} steps`}
      title="Now make it learn"
      lede={
        <>
          Training is a loop: guess the next word, measure how wrong the guess
          was, and nudge every number in the model slightly in the direction that
          would have been less wrong. Repeat a few hundred times.
        </>
      }
    >
      <Panel wide>
        <div className="presets">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              disabled={busy}
              className={`preset${preset.id === p.id ? ' is-active' : ''}`}
              onClick={() => setPreset(p)}
            >
              <b>{p.name}</b>
              <span>{p.blurb}</span>
              <span className="preset__spec mono">
                {p.nLayers} block{p.nLayers > 1 ? 's' : ''} &middot; {p.nHeads} heads
                &middot; width {p.dModel} &middot; {p.steps} steps
              </span>
            </button>
          ))}
        </div>

        <div className="trainbar">
          <button
            type="button"
            className="btn btn--primary"
            onClick={busy ? cancelTraining : startTraining}
          >
            {busy ? 'Stop' : training.status === 'done' ? 'Train again' : 'Train'}
          </button>
          <div className="progress" role="progressbar" aria-valuenow={pct}>
            <span style={{ width: `${pct}%` }} />
          </div>
          <span className="mono trainbar__pct">{pct}%</span>
        </div>

        {training.error ? (
          <p className="notice notice--warn">{training.error}</p>
        ) : null}

        <div className="stats">
          <Stat label="step" value={`${training.step}/${training.totalSteps || preset.steps}`} />
          <Stat label="loss" value={training.loss ? training.loss.toFixed(3) : '--'} sub="nats per word" />
          <Stat
            label="perplexity"
            value={training.loss ? Math.exp(training.loss).toFixed(1) : '--'}
            sub="effective number of guesses"
          />
          <Stat
            label="speed"
            value={training.stepsPerSecond ? `${training.stepsPerSecond.toFixed(0)}/s` : '--'}
            sub="steps per second"
          />
        </div>
      </Panel>

      <Panel
        title="Loss"
        note="Loss is the model's average surprise at the correct word. Lower is better, and the dashed line is what pure guessing would score."
        wide
      >
        <LossChart
          history={training.history}
          totalSteps={training.totalSteps || preset.steps}
          uniformLoss={training.uniformLoss}
        />
      </Panel>

      <div className="blockgrid">
        <Panel
          title="What it can write, so far"
          note="Regenerated every 40 steps from the same starting words. Early on this is noise; watch structure arrive before meaning does."
        >
          <pre className="sample mono">
            {training.sample || 'Press Train to watch this fill in.'}
          </pre>
        </Panel>

        <Panel
          title="Embeddings, moving"
          note="The same map from stage 4. Every dot is a word being pushed toward the words it behaves like."
        >
          <EmbeddingMap
            coords={training.embedding?.coords ?? null}
            ids={training.embedding?.ids ?? null}
            labels={vocab?.tokens ?? []}
          />
        </Panel>
      </div>

      {training.status === 'done' && training.isOwnModel ? (
        <p className="notice notice--ok">
          Done in {(training.elapsedMs / 1000).toFixed(1)}s. Final loss{' '}
          <b className="mono">{training.loss.toFixed(3)}</b> against a guessing
          baseline of <b className="mono">{training.uniformLoss.toFixed(3)}</b>. Every
          visual on this page now uses your model &mdash; scroll back up and the
          attention patterns will have changed.
        </p>
      ) : null}
    </Stage>
  )
}
