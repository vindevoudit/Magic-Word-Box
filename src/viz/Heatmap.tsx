import { useCallback, useRef, useState } from 'react'
import { emphasise, rampCss } from './ramp'
import { useCanvas } from './useTween'

export interface HeatmapProps {
  /** Row-major [n, n] attention weights for one batch item and one head. */
  probs: Float32Array
  n: number
  labels: string[]
  /** 0 to 1: how much of the causal mask has swept in. 1 draws the full matrix. */
  reveal?: number
  onHoverCell?: (cell: { row: number; col: number } | null) => void
  hovered?: { row: number; col: number } | null
}

const PAD_LEFT = 92
const PAD_TOP = 76

/**
 * The attention matrix.
 *
 * Read a row as: "when the model was at this word, how much did it look at
 * each earlier word?" The dark triangle above the diagonal is not missing
 * data - it is the causal mask, the rule that a token may never see its own
 * future.
 */
export function Heatmap({
  probs,
  n,
  labels,
  reveal = 1,
  onHoverCell,
  hovered,
}: HeatmapProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [cell, setCell] = useState(0)

  const canvasRef = useCanvas(
    (ctx, w, h) => {
      const size = Math.max(0, Math.min(w - PAD_LEFT, h - PAD_TOP))
      const c = size / Math.max(1, n)
      setCell(c)

      ctx.font = '11px "JetBrains Mono", monospace'
      ctx.textBaseline = 'middle'

      for (let r = 0; r < n; r++) {
        for (let col = 0; col < n; col++) {
          const x = PAD_LEFT + col * c
          const y = PAD_TOP + r * c

          if (col > r) {
            // Masked. Drawn, not omitted, so the triangle reads as a rule
            // rather than as an absence.
            ctx.fillStyle = '#0d1220'
            ctx.fillRect(x, y, c - 1, c - 1)
            continue
          }

          // Cells appear diagonal by diagonal, which traces the mask forming.
          const appear = Math.max(0, Math.min(1, reveal * n - r))
          const p = probs[r * n + col] * appear
          ctx.fillStyle = rampCss(emphasise(p))
          ctx.fillRect(x, y, c - 1, c - 1)
        }
      }

      if (hovered && hovered.row < n && hovered.col <= hovered.row) {
        ctx.strokeStyle = 'rgba(255,107,74,0.28)'
        ctx.lineWidth = 1
        ctx.strokeRect(PAD_LEFT, PAD_TOP + hovered.row * c, n * c, c)
        ctx.strokeRect(PAD_LEFT + hovered.col * c, PAD_TOP, c, n * c)
        ctx.strokeStyle = '#ff6b4a'
        ctx.lineWidth = 2
        ctx.strokeRect(
          PAD_LEFT + hovered.col * c + 1,
          PAD_TOP + hovered.row * c + 1,
          c - 3,
          c - 3,
        )
      }

      // Row labels: the token doing the looking.
      ctx.textAlign = 'right'
      for (let r = 0; r < n; r++) {
        const active = hovered?.row === r
        ctx.fillStyle = active ? '#ff6b4a' : '#8a9ab8'
        ctx.fillText(truncate(labels[r] ?? ''), PAD_LEFT - 10, PAD_TOP + r * c + c / 2)
      }

      // Column labels: the token being looked at. Rotated so long words fit.
      ctx.textAlign = 'left'
      for (let col = 0; col < n; col++) {
        const active = hovered?.col === col
        ctx.save()
        ctx.translate(PAD_LEFT + col * c + c / 2, PAD_TOP - 10)
        ctx.rotate(-Math.PI / 4)
        ctx.fillStyle = active ? '#ff6b4a' : '#5f6f8c'
        ctx.fillText(truncate(labels[col] ?? ''), 0, 0)
        ctx.restore()
      }
    },
    [probs, n, labels, reveal, hovered],
  )

  const handleMove = useCallback(
    (e: React.MouseEvent) => {
      if (!onHoverCell || cell <= 0) return
      const rect = wrapRef.current?.getBoundingClientRect()
      if (!rect) return
      const col = Math.floor((e.clientX - rect.left - PAD_LEFT) / cell)
      const row = Math.floor((e.clientY - rect.top - PAD_TOP) / cell)
      if (row < 0 || col < 0 || row >= n || col >= n || col > row) {
        onHoverCell(null)
      } else {
        onHoverCell({ row, col })
      }
    },
    [cell, n, onHoverCell],
  )

  return (
    <div
      ref={wrapRef}
      className="heatmap"
      onMouseMove={handleMove}
      onMouseLeave={() => onHoverCell?.(null)}
    >
      <canvas ref={canvasRef} />
    </div>
  )
}

function truncate(s: string): string {
  if (s === '<br>') return '\u21b5'
  if (s === '<start>') return '\u25b8'
  return s.length > 9 ? `${s.slice(0, 8)}\u2026` : s
}
