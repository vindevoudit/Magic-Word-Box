import type { ReactNode } from 'react'
import { useInView } from '../viz/useInView'

export interface StageProps {
  id: string
  index: number
  title: string
  /** The real tensor shape at this point in the pipeline, e.g. "[16, 8]". */
  shape?: string
  kicker?: string
  lede?: ReactNode
  children?: ReactNode
}

/**
 * One numbered stage.
 *
 * The numbering is not decoration: these stages are the forward pass in order,
 * so the number tells the reader where in the machine they are standing. The
 * eyebrow carries the tensor shape at that point for the same reason.
 */
export function Stage({ id, index, title, shape, kicker, lede, children }: StageProps) {
  const [ref, seen] = useInView<HTMLElement>(0.15)

  return (
    <section
      id={id}
      ref={ref}
      className={`stage${seen ? ' is-seen' : ''}`}
      data-stage={index}
      aria-labelledby={`${id}-title`}
    >
      <div className="stage__head">
        <span className="eyebrow">
          Stage {String(index).padStart(2, '0')}
          {kicker ? <> &middot; {kicker}</> : null}
          {shape ? (
            <>
              {' '}
              &middot; <em>{shape}</em>
            </>
          ) : null}
        </span>
        <h2 id={`${id}-title`} className="stage__title">
          {title}
        </h2>
        {lede ? <div className="lede stage__lede">{lede}</div> : null}
      </div>
      <div className="stage__body">{children}</div>
    </section>
  )
}

/** A titled panel around a visual. */
export function Panel({
  title,
  note,
  children,
  wide,
}: {
  title?: string
  note?: ReactNode
  children: ReactNode
  wide?: boolean
}) {
  return (
    <div className={`panel${wide ? ' panel--wide' : ''}`}>
      {title ? <h3 className="panel__title">{title}</h3> : null}
      {children}
      {note ? <p className="panel__note">{note}</p> : null}
    </div>
  )
}

/** A labelled number, used for corpus and model statistics. */
export function Stat({
  label,
  value,
  sub,
}: {
  label: string
  value: string | number
  sub?: string
}) {
  return (
    <div className="stat">
      <span className="stat__value mono">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </span>
      <span className="stat__label">{label}</span>
      {sub ? <span className="stat__sub">{sub}</span> : null}
    </div>
  )
}
