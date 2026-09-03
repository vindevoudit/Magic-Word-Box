import { Panel, Stage } from './Stage'
import { useStore } from '../state/store'

const OMITTED = [
  {
    name: 'Subword tokens',
    detail:
      'Real models split rare words into pieces, so they can spell something they have never seen. This one maps anything unfamiliar to a single unknown token and can never say it.',
  },
  {
    name: 'Rotary positions',
    detail:
      'Position here is a learned vector per slot, capped at the window length. Modern models encode position as a rotation instead, which extends to far longer inputs.',
  },
  {
    name: 'Scale',
    detail:
      'More layers, wider vectors, and text measured in trillions of words rather than hundreds. Almost everything people find surprising about large models comes from this row.',
  },
  {
    name: 'Instruction tuning',
    detail:
      'A raw model like this only continues text. Being helpful when asked a question is a second training stage that this site does not cover at all.',
  },
]

/**
 * Stage 11. The honest close.
 *
 * A model this small produces text that loops and repeats. Naming that plainly
 * turns the obvious limitation into the final lesson, rather than leaving the
 * reader to conclude something went wrong.
 */
export function WhatScaleBuys() {
  const { model, training, corpus } = useStore()
  const params = model?.paramCount ?? 0
  const words = corpus.split(/\s+/).filter(Boolean).length
  const ratio = params > 0 ? 175_000_000_000 / params : 0

  return (
    <Stage
      id="scale"
      index={11}
      kicker="the honest part"
      title="What you just built, and what you did not"
      lede={
        <>
          Your model has <b className="mono accent">{params.toLocaleString()}</b>{' '}
          parameters and read <b className="mono accent">{words.toLocaleString()}</b>{' '}
          words. It is a real transformer &mdash; the same architecture, every
          piece present. It is simply very small.
        </>
      }
    >
      <Panel wide>
        <div className="scalebar">
          <div className="scalebar__row">
            <span className="scalebar__label mono">yours</span>
            <span className="scalebar__track">
              <span className="scalebar__fill" style={{ width: '0.6%' }} />
            </span>
            <span className="mono scalebar__num">{params.toLocaleString()}</span>
          </div>
          <div className="scalebar__row">
            <span className="scalebar__label mono">GPT-3</span>
            <span className="scalebar__track">
              <span className="scalebar__fill scalebar__fill--big" style={{ width: '100%' }} />
            </span>
            <span className="mono scalebar__num">175,000,000,000</span>
          </div>
        </div>
        <p className="muted">
          About{' '}
          <b className="mono accent">
            {ratio > 0 ? `${Math.round(ratio).toLocaleString()}x` : '--'}
          </b>{' '}
          more parameters, trained on many millions of times more text.
        </p>
      </Panel>

      <Panel title="Why your model repeats itself">
        <p>
          With this little text, the cheapest way to lower the loss is to
          memorise. So it does &mdash; it reproduces lines it has seen, and
          loops when it runs out. That is not a bug in the code, it is what a
          model does when it has far more capacity than data.
          {training.isOwnModel && training.loss > 0 ? (
            <>
              {' '}
              Yours reached a loss of{' '}
              <b className="mono">{training.loss.toFixed(3)}</b>, against{' '}
              <b className="mono">{training.uniformLoss.toFixed(3)}</b> for random
              guessing.
            </>
          ) : null}{' '}
          Give it more text with the same patterns and it starts generalising
          instead.
        </p>
      </Panel>

      <Panel title="What a real model adds" wide>
        <ul className="omitted">
          {OMITTED.map((o) => (
            <li key={o.name}>
              <b>{o.name}</b>
              <span className="muted">{o.detail}</span>
            </li>
          ))}
        </ul>
      </Panel>

      <Panel>
        <p className="closing">
          Everything you just watched &mdash; the lookup, the queries and keys,
          the triangle, the loop that nudges numbers downhill &mdash; is what
          runs inside a model a million times this size. The parts do not change.
          Only the number of them does.
        </p>
      </Panel>
    </Stage>
  )
}
