import { rampCss } from './ramp'

export interface VocabGridProps {
  tokens: string[]
  counts: number[]
  max?: number
  onHover?: (id: number | null) => void
  activeId?: number | null
}

/**
 * The vocabulary, sorted by how often each word appears.
 *
 * Every word the model can ever say is in this grid, and each one is really
 * just its index. Colour encodes frequency, which is also the reason the grid
 * is ordered this way: the words at the top carry most of the corpus.
 */
export function VocabGrid({
  tokens,
  counts,
  max = 120,
  onHover,
  activeId,
}: VocabGridProps) {
  const maxCount = Math.max(1, ...counts.slice(0, max))
  const shown = tokens.slice(0, max)

  return (
    <div className="vocabgrid" onMouseLeave={() => onHover?.(null)}>
      {shown.map((t, id) => (
        <span
          key={id}
          className={`vocabcell${activeId === id ? ' is-active' : ''}`}
          onMouseEnter={() => onHover?.(id)}
          style={{
            borderColor:
              activeId === id ? '#ff6b4a' : rampCss(Math.sqrt(counts[id] / maxCount), 0.55),
          }}
          title={`id ${id} - appears ${counts[id].toLocaleString()} times`}
        >
          <b className="mono">{id}</b>
          <span>{t === '<br>' ? '\u21b5' : t}</span>
        </span>
      ))}
      {tokens.length > max ? (
        <span className="vocabcell vocabcell--more">
          +{(tokens.length - max).toLocaleString()}
        </span>
      ) : null}
    </div>
  )
}
