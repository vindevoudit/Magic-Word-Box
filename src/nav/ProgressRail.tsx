import { useEffect, useState } from 'react'

export interface RailItem {
  id: string
  label: string
}

/**
 * Sticky navigation.
 *
 * The markers are drawn as a filling triangle, echoing the causal mask: by the
 * last stage the reader has assembled the same shape the model uses. It doubles
 * as a progress meter, so the device carries information rather than ornament.
 */
export function ProgressRail({ items }: { items: RailItem[] }) {
  const [active, setActive] = useState(items[0]?.id ?? '')

  useEffect(() => {
    const sections = items
      .map((i) => document.getElementById(i.id))
      .filter((el): el is HTMLElement => el != null)
    if (sections.length === 0) return

    const obs = new IntersectionObserver(
      (entries) => {
        // The heading closest to the top of the viewport wins, which keeps the
        // rail stable when two tall stages overlap.
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible[0]) setActive(visible[0].target.id)
      },
      { rootMargin: '-12% 0px -70% 0px', threshold: 0 },
    )
    sections.forEach((s) => obs.observe(s))
    return () => obs.disconnect()
  }, [items])

  const activeIndex = Math.max(
    0,
    items.findIndex((i) => i.id === active),
  )

  return (
    <nav className="rail" aria-label="Stages">
      <ol className="rail__list">
        {items.map((item, i) => {
          const done = i < activeIndex
          const isActive = i === activeIndex
          return (
            <li key={item.id}>
              <a
                href={`#${item.id}`}
                className={`rail__item${isActive ? ' is-active' : ''}${done ? ' is-done' : ''}`}
                aria-current={isActive ? 'step' : undefined}
              >
                <span className="rail__mark" aria-hidden="true">
                  {Array.from({ length: i + 1 }, (_, c) => (
                    <i key={c} />
                  ))}
                </span>
                <span className="rail__label">{item.label}</span>
              </a>
            </li>
          )
        })}
      </ol>
      <div className="rail__meter" aria-hidden="true">
        <span style={{ width: `${((activeIndex + 1) / items.length) * 100}%` }} />
      </div>
    </nav>
  )
}
