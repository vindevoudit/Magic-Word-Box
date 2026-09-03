import { useCanvas } from './useTween'

export interface EmbeddingMapProps {
  coords: Float32Array | null
  ids: Int32Array | null
  labels: string[]
  /** Ids to draw in the live accent colour. */
  highlight?: Set<number>
}

/**
 * Token embeddings, projected to two dimensions.
 *
 * Each dot is a word, positioned by the vector the model invented for it. The
 * axes have no inherent meaning - they are just the two directions along which
 * the words differ most. What matters is which words end up near each other,
 * and watching that arrangement rearrange itself during training.
 */
export function EmbeddingMap({ coords, ids, labels, highlight }: EmbeddingMapProps) {
  const canvasRef = useCanvas(
    (ctx, w, h) => {
      if (!coords || !ids || coords.length < 4) {
        ctx.fillStyle = '#5f6f8c'
        ctx.font = '12px "Public Sans", sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('Train the model to watch this map form.', w / 2, h / 2)
        return
      }

      const n = ids.length
      let minX = Infinity
      let maxX = -Infinity
      let minY = Infinity
      let maxY = -Infinity
      for (let i = 0; i < n; i++) {
        minX = Math.min(minX, coords[i * 2])
        maxX = Math.max(maxX, coords[i * 2])
        minY = Math.min(minY, coords[i * 2 + 1])
        maxY = Math.max(maxY, coords[i * 2 + 1])
      }
      const pad = 34
      const sx = (maxX - minX) || 1
      const sy = (maxY - minY) || 1
      const px = (v: number) => pad + ((v - minX) / sx) * (w - pad * 2)
      const py = (v: number) => h - pad - ((v - minY) / sy) * (h - pad * 2)

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'

      // Labels are drawn only where they will not collide, so the map stays
      // readable at any vocabulary size. Dots are always drawn.
      const placed: { x: number; y: number }[] = []
      for (let i = 0; i < n; i++) {
        const x = px(coords[i * 2])
        const y = py(coords[i * 2 + 1])
        const hot = highlight?.has(ids[i]) ?? false

        ctx.beginPath()
        ctx.arc(x, y, hot ? 3.5 : 2.2, 0, Math.PI * 2)
        ctx.fillStyle = hot ? '#ff6b4a' : '#2fa8a0'
        ctx.globalAlpha = hot ? 1 : 0.72
        ctx.fill()
        ctx.globalAlpha = 1

        const label = labels[ids[i]] ?? ''
        if (!label || label.startsWith('<')) continue
        const clash = placed.some((p) => Math.abs(p.x - x) < 42 && Math.abs(p.y - y) < 13)
        if (clash && !hot) continue
        placed.push({ x, y })
        ctx.fillStyle = hot ? '#ff6b4a' : '#8a9ab8'
        ctx.fillText(label, x, y - 9)
      }
    },
    [coords, ids, labels, highlight],
  )

  return (
    <div className="chart chart--tall">
      <canvas ref={canvasRef} />
    </div>
  )
}
