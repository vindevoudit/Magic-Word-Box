import { useEffect, useState } from 'react'
import { prefersReducedMotion } from './useTween'

export interface DiagramNode {
  id: string
  label: string
  detail: string
  kind: 'io' | 'norm' | 'attn' | 'mlp' | 'add'
}

export interface BlockDiagramProps {
  nodes: DiagramNode[]
  selected: string | null
  onSelect: (id: string | null) => void
  /** Set true to send a pulse travelling down the stack. */
  running: boolean
}

const NODE_H = 44
const GAP = 15
const W = 260
const X = 42

/**
 * The forward pass as a diagram, with the residual stream drawn as what it is:
 * a single line running the whole height of the block that each sublayer adds
 * into, rather than a chain each sublayer replaces.
 */
export function BlockDiagram({ nodes, selected, onSelect, running }: BlockDiagramProps) {
  const [pulse, setPulse] = useState(-1)
  const height = nodes.length * (NODE_H + GAP) + GAP

  useEffect(() => {
    if (!running) {
      setPulse(-1)
      return
    }
    if (prefersReducedMotion()) {
      setPulse(nodes.length - 1)
      return
    }
    let i = 0
    setPulse(0)
    const timer = setInterval(() => {
      i += 1
      if (i >= nodes.length) {
        clearInterval(timer)
        setTimeout(() => setPulse(-1), 500)
      } else {
        setPulse(i)
      }
    }, 260)
    return () => clearInterval(timer)
  }, [running, nodes.length])

  const yOf = (i: number) => GAP + i * (NODE_H + GAP)

  return (
    <div className="diagram-scroll">
      <svg
        className="diagram"
        viewBox={`0 0 ${W + X * 2} ${height}`}
        width={W + X * 2}
        height={height}
        role="img"
        aria-label="The forward pass through one transformer block"
      >
        <line
          className="diagram__stream"
          x1={X - 20}
          y1={GAP}
          x2={X - 20}
          y2={height - GAP}
        />
        {nodes.map((n, i) => (
          <line
            key={`c${n.id}`}
            className="diagram__wire"
            x1={X - 20}
            y1={yOf(i) + NODE_H / 2}
            x2={X}
            y2={yOf(i) + NODE_H / 2}
          />
        ))}

        {nodes.map((n, i) => {
          const active = selected === n.id
          const lit = pulse === i
          return (
            <g
              key={n.id}
              className={`dnode dnode--${n.kind}${active ? ' is-selected' : ''}${lit ? ' is-lit' : ''}`}
              onClick={() => onSelect(active ? null : n.id)}
              tabIndex={0}
              role="button"
              aria-pressed={active}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onSelect(active ? null : n.id)
                }
              }}
            >
              <rect x={X} y={yOf(i)} width={W} height={NODE_H} rx={6} />
              <text x={X + 14} y={yOf(i) + 26}>
                {n.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
