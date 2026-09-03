import { useEffect, useRef } from 'react'
import { prefersReducedMotion } from './useTween'

/**
 * The live causal triangle behind the page.
 *
 * Real attention weights from the loaded model, redrawn slowly at low opacity.
 * It is the one ambient effect on the site, and it earns its place by being
 * actual data rather than decoration: the shape it traces is the rule that
 * every token may only look backward.
 */
export function AmbientGrid({ probs, n }: { probs: Float32Array | null; n: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef(0)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !probs || n < 2) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const still = prefersReducedMotion()
    let phase = 0

    const render = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const w = window.innerWidth
      const h = window.innerHeight
      if (canvas.width !== Math.round(w * dpr)) {
        canvas.width = Math.round(w * dpr)
        canvas.height = Math.round(h * dpr)
        canvas.style.width = `${w}px`
        canvas.style.height = `${h}px`
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const cell = Math.max(26, Math.min(w, h) / (n + 2))
      const size = cell * n
      const ox = w - size - cell * 0.8
      const oy = h * 0.5 - size / 2

      for (let r = 0; r < n; r++) {
        for (let c = 0; c <= r; c++) {
          const p = probs[r * n + c]
          // A slow travelling wave along the diagonal keeps it alive without
          // ever becoming something the eye tracks.
          const wave = still ? 0.5 : 0.5 + 0.5 * Math.sin(phase - (r + c) * 0.35)
          const a = 0.018 + p * 0.14 * (0.45 + wave * 0.55)
          ctx.fillStyle = `rgba(47,168,160,${a.toFixed(4)})`
          ctx.fillRect(ox + c * cell, oy + r * cell, cell - 2, cell - 2)
        }
      }

      if (!still) {
        phase += 0.011
        rafRef.current = requestAnimationFrame(render)
      }
    }

    render()
    const onResize = () => render()
    window.addEventListener('resize', onResize)
    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', onResize)
    }
  }, [probs, n])

  return <canvas ref={ref} className="ambient" aria-hidden="true" />
}
