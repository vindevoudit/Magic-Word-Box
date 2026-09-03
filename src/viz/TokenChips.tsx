export interface TokenChipsProps {
  tokens: string[]
  /** 0 to 1: how many chips have landed. Drives the shatter animation. */
  reveal?: number
  ids?: number[]
  max?: number
  onHover?: (index: number | null) => void
  activeIndex?: number | null
}

/** Raw text after splitting: one chip per token, punctuation included. */
export function TokenChips({
  tokens,
  reveal = 1,
  ids,
  max = 160,
  onHover,
  activeIndex,
}: TokenChipsProps) {
  const shown = tokens.slice(0, max)
  const cut = Math.ceil(shown.length * Math.max(0, Math.min(1, reveal)))

  return (
    <div className="chips" onMouseLeave={() => onHover?.(null)}>
      {shown.map((t, i) => (
        <span
          key={i}
          className={[
            'chip',
            i < cut ? 'chip--in' : '',
            t === '<br>' ? 'chip--break' : '',
            activeIndex === i ? 'chip--active' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          style={{ transitionDelay: `${Math.min(600, (i % 40) * 12)}ms` }}
          onMouseEnter={() => onHover?.(i)}
        >
          {t === '<br>' ? '\u21b5' : t}
          {ids ? <b className="chip__id">{ids[i]}</b> : null}
        </span>
      ))}
      {tokens.length > max ? (
        <span className="chip chip--more chip--in">
          +{(tokens.length - max).toLocaleString()} more
        </span>
      ) : null}
    </div>
  )
}
