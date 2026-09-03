import { useEffect, useRef, useState } from 'react'

/**
 * Fires once when an element scrolls into view.
 *
 * Stage animations are one-shot on purpose: replaying them every time the
 * reader scrolls back past a section turns a considered reveal into a twitch.
 */
export function useInView<T extends HTMLElement>(
  threshold = 0.25,
): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null)
  const [seen, setSeen] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el || seen) return
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(true)
      return
    }
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setSeen(true)
            obs.disconnect()
          }
        }
      },
      { threshold },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [threshold, seen])

  return [ref, seen]
}
