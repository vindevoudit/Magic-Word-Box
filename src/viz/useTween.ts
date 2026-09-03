import { useEffect, useRef, useState } from 'react'

/** Whether the visitor asked for less motion. Checked at call time, not cached. */
export function prefersReducedMotion(): boolean {
  return (
    typeof matchMedia !== 'undefined' &&
    matchMedia('(prefers-reduced-motion: reduce)').matches
  )
}

export const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3)
export const easeInOutCubic = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2

/**
 * Animate a number from its previous value to `target`.
 *
 * Hand-rolled rather than pulled from a library: the page needs exactly this
 * one behaviour, and a motion library would cost more bytes than the entire
 * transformer implementation.
 */
export function useTweenedNumber(target: number, duration = 600): number {
  const [value, setValue] = useState(target)
  const fromRef = useRef(target)
  const startRef = useRef(0)
  const rafRef = useRef(0)

  useEffect(() => {
    if (prefersReducedMotion() || duration <= 1) {
      setValue(target)
      return
    }
    fromRef.current = value
    startRef.current = performance.now()

    const tick = (now: number) => {
      const t = Math.min(1, (now - startRef.current) / duration)
      setValue(fromRef.current + (target - fromRef.current) * easeOutCubic(t))
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
    // `value` is intentionally not a dependency: including it would restart the
    // tween on every frame it sets.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, duration])

  return value
}

/** Drives a 0 to 1 progress value once `active` becomes true. */
export function useReveal(active: boolean, duration = 900): number {
  const [t, setT] = useState(0)
  const rafRef = useRef(0)

  useEffect(() => {
    if (!active) return
    if (prefersReducedMotion() || duration <= 1) {
      setT(1)
      return
    }
    const start = performance.now()
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / duration)
      setT(p)
      if (p < 1) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [active, duration])

  return t
}

/**
 * A canvas sized to its container and to the device pixel ratio.
 *
 * Every canvas visual on the page needs the same three things: a ref, a resize
 * observer, and a context already scaled so one unit is one CSS pixel.
 */
export function useCanvas(
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  deps: unknown[],
): React.RefObject<HTMLCanvasElement> {
  const ref = useRef<HTMLCanvasElement>(null)
  const drawRef = useRef(draw)
  drawRef.current = draw

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const parent = canvas.parentElement
    if (!parent) return

    const render = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const w = parent.clientWidth
      const h = parent.clientHeight
      if (w === 0 || h === 0) return
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      drawRef.current(ctx, w, h)
    }

    render()
    const ro = new ResizeObserver(render)
    ro.observe(parent)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return ref
}
