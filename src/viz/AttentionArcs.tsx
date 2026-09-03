import { rampCss } from './ramp'

export interface AttentionArcsProps {
  probs: Float32Array
  n: number
  labels: string[]
  /** Which query position to draw arcs for. */
  row: number
  onSelectRow?: (row: number) => void
  highlightCol?: number | null
}

const H = 132
const TOKEN_Y = H - 26

/**
 * The same numbers as the heatmap, drawn as connections.
 *
 * The matrix is precise but abstract; arcs make one row concrete - this word,
 * reaching back to these earlier words, this strongly. Together they teach the
 * grid and the meaning at once.
 */
export function AttentionArcs({
  probs,
  n,
  labels,
  row,
  onSelectRow,
  highlightCol,
}: AttentionArcsProps) {
  const width = Math.max(320, n * 84)
  const step = width / Math.max(1, n)
  const x = (i: number) => step * i + step / 2

  const arcs: { col: number; p: number }[] = []
  for (let col = 0; col <= row && col < n; col++) {
    const p = probs[row * n + col]
    if (p > 0.004) arcs.push({ col, p })
  }
  arcs.sort((a, b) => a.p - b.p)

  return (
    <div className="arcs-scroll">
      <svg
        className="arcs"
        viewBox={`0 0 ${width} ${H}`}
        width={width}
        height={H}
        role="img"
        aria-label={`Attention from the word "${labels[row] ?? ''}" to the words before it`}
      >
        {arcs.map(({ col, p }) => {
          const from = x(row)
          const to = x(col)
          const span = Math.abs(from - to)
          const lift = Math.min(TOKEN_Y - 12, 22 + span * 0.42)
          const dim = highlightCol != null && highlightCol !== col
          return (
            <path
              key={col}
              d={`M ${from} ${TOKEN_Y - 14} C ${from} ${TOKEN_Y - lift}, ${to} ${TOKEN_Y - lift}, ${to} ${TOKEN_Y - 14}`}
              fill="none"
              stroke={rampCss(Math.sqrt(p))}
              strokeWidth={1 + p * 9}
              strokeLinecap="round"
              opacity={dim ? 0.16 : 0.42 + p * 0.58}
            />
          )
        })}

        {Array.from({ length: n }, (_, i) => {
          const isQuery = i === row
          const isKey = i <= row
          return (
            <g
              key={i}
              className={`arc-token${isQuery ? ' is-query' : ''}${isKey ? '' : ' is-future'}`}
              onClick={() => onSelectRow?.(i)}
              tabIndex={onSelectRow ? 0 : -1}
              role={onSelectRow ? 'button' : undefined}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onSelectRow?.(i)
                }
              }}
            >
              <rect
                x={x(i) - step / 2 + 4}
                y={TOKEN_Y - 13}
                width={step - 8}
                height={26}
                rx={4}
              />
              <text x={x(i)} y={TOKEN_Y + 1} textAnchor="middle">
                {display(labels[i] ?? '')}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function display(s: string): string {
  if (s === '<br>') return '\u21b5'
  if (s === '<start>') return 'start'
  return s.length > 10 ? `${s.slice(0, 9)}\u2026` : s
}
