import { rampCss } from './ramp'

export interface ProbBarsProps {
  probs: Float32Array
  labels: string[]
  k?: number
  onPick?: (id: number) => void
  /** Marks the token that was actually sampled. */
  chosen?: number | null
}

/**
 * The model's answer, as a distribution.
 *
 * This is the one picture the whole site builds toward: not a single next word,
 * but every candidate word with a number attached. Bars are laid out with CSS
 * transitions so that moving the temperature slider visibly *reshapes* the
 * distribution rather than replacing it.
 */
export function ProbBars({ probs, labels, k = 8, onPick, chosen }: ProbBarsProps) {
  const idx = Array.from({ length: probs.length }, (_, i) => i)
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, k)

  const top = probs[idx[0]] ?? 1

  return (
    <ol className="probbars" aria-label="Most likely next words">
      {idx.map((id) => {
        const p = probs[id]
        const label = labels[id] ?? '?'
        const Tag = onPick ? 'button' : 'div'
        return (
          <li key={id}>
            <Tag
              className={`probbar${chosen === id ? ' is-chosen' : ''}`}
              onClick={onPick ? () => onPick(id) : undefined}
              type={onPick ? 'button' : undefined}
              title={onPick ? `Continue with "${label}"` : undefined}
            >
              <span className="probbar__word mono">{display(label)}</span>
              <span className="probbar__track">
                <span
                  className="probbar__fill"
                  style={{
                    width: `${Math.max(1.5, (p / top) * 100)}%`,
                    background: rampCss(Math.sqrt(p)),
                  }}
                />
              </span>
              <span className="probbar__pct mono">{(p * 100).toFixed(1)}%</span>
            </Tag>
          </li>
        )
      })}
    </ol>
  )
}

function display(s: string): string {
  if (s === '<br>') return 'new line'
  if (s === '<start>') return 'start'
  if (s === '<unk>') return 'unknown'
  return s
}
