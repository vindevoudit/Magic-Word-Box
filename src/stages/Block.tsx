import { useState } from 'react'
import { Panel, Stage } from './Stage'
import { useStore } from '../state/store'
import { BlockDiagram, type DiagramNode } from '../viz/BlockDiagram'
import { rampCss } from '../viz/ramp'

function nodesFor(d: number, heads: number): DiagramNode[] {
  return [
    {
      id: 'in',
      label: 'token + position vectors',
      kind: 'io',
      detail: `Each word arrives as ${d} numbers: what it is, plus where it sits.`,
    },
    {
      id: 'ln1',
      label: 'normalise',
      kind: 'norm',
      detail:
        'Rescales each vector to a consistent size. Without this, values drift bigger through every layer until training falls apart.',
    },
    {
      id: 'attn',
      label: `attention, ${heads} head${heads > 1 ? 's' : ''}`,
      kind: 'attn',
      detail:
        'The only step where words see each other. Everything else treats each position on its own.',
    },
    {
      id: 'add1',
      label: 'add back into the stream',
      kind: 'add',
      detail:
        'Attention does not replace the word, it adds a correction to it. The original survives, which is what lets learning reach the earliest layers.',
    },
    {
      id: 'ln2',
      label: 'normalise',
      kind: 'norm',
      detail: 'Same rescaling, before the second sublayer.',
    },
    {
      id: 'mlp',
      label: `feed-forward, ${d} to ${d * 4} to ${d}`,
      kind: 'mlp',
      detail:
        'Each position is expanded fourfold, passed through a curve, then squeezed back. This is where most of the parameters live, and where the model stores what it knows about individual words.',
    },
    {
      id: 'add2',
      label: 'add back into the stream',
      kind: 'add',
      detail: 'A second correction, added the same way.',
    },
  ]
}

/**
 * Stage 7. Assembles the parts into the repeating unit.
 *
 * The residual stream is drawn as a continuous line down the left rather than
 * as a chain of boxes, because that is the honest picture: each sublayer reads
 * the stream and adds to it, never replaces it.
 */
export function Block() {
  const { model, training } = useStore()
  const [selected, setSelected] = useState<string | null>(null)

  const d = model?.cfg.dModel ?? 32
  const heads = model?.cfg.nHeads ?? 2
  const layers = model?.cfg.nLayers ?? 1
  const nodes = nodesFor(d, heads)
  const detail = nodes.find((n) => n.id === selected)

  const breakdown = model?.paramBreakdown() ?? []
  const total = model?.paramCount ?? 0

  return (
    <Stage
      id="block"
      index={7}
      kicker="the repeating unit"
      shape={`x${layers} block${layers > 1 ? 's' : ''}`}
      title="One block, stacked"
      lede={
        <>
          Attention is one part of a larger unit. That unit gets stacked
          &mdash; your model uses {layers} of them &mdash; and each one refines
          what the last produced. Click any step to see what it does.
        </>
      }
    >
      <div className="blockgrid">
        <Panel title="Data flow">
          <BlockDiagram
            nodes={nodes}
            selected={selected}
            onSelect={setSelected}
            running={training.status === 'training'}
          />
        </Panel>

        <div className="blockgrid__side">
          <Panel title={detail ? detail.label : 'Pick a step'}>
            <p className="muted">
              {detail
                ? detail.detail
                : 'Each box is one operation in the forward pass. They run top to bottom, and the line down the left is the residual stream every one of them adds into.'}
            </p>
          </Panel>

          <Panel
            title="Where the parameters are"
            note={`${total.toLocaleString()} numbers in total. GPT-3 had 175 billion.`}
          >
            <ul className="paramlist">
              {breakdown.map((row, i) => (
                <li key={row.label}>
                  <span className="paramlist__label">{row.label}</span>
                  <span className="paramlist__bar">
                    <span
                      style={{
                        width: `${(row.count / Math.max(1, total)) * 100}%`,
                        background: rampCss(0.25 + (i / breakdown.length) * 0.6),
                      }}
                    />
                  </span>
                  <span className="mono paramlist__count">
                    {row.count.toLocaleString()}
                  </span>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      </div>
    </Stage>
  )
}
