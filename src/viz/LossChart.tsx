import { useCanvas } from './useTween'

export interface LossChartProps {
  history: number[]
  totalSteps: number
  /** Loss of a model that guesses uniformly; the line worth beating. */
  uniformLoss: number
}

/**
 * The loss curve.
 *
 * Loss is in nats: the average surprise, per word, of the correct answer. The
 * dashed line is what pure guessing scores, so the gap between the curve and
 * that line is exactly how much the model has actually learned.
 */
export function LossChart({ history, totalSteps, uniformLoss }: LossChartProps) {
  const canvasRef = useCanvas(
    (ctx, w, h) => {
      const padL = 44
      const padB = 26
      const padT = 14
      const padR = 12
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      if (plotW <= 0 || plotH <= 0) return

      const maxY = Math.max(uniformLoss * 1.08, ...history.filter(Number.isFinite), 0.5)
      const xAt = (i: number) => padL + (i / Math.max(1, totalSteps - 1)) * plotW
      const yAt = (v: number) => padT + plotH - (v / maxY) * plotH

      ctx.strokeStyle = '#1e2739'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#5f6f8c'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      for (let i = 0; i <= 4; i++) {
        const v = (maxY / 4) * i
        const y = yAt(v)
        ctx.beginPath()
        ctx.moveTo(padL, y)
        ctx.lineTo(w - padR, y)
        ctx.stroke()
        ctx.fillText(v.toFixed(1), padL - 8, y)
      }

      if (uniformLoss > 0) {
        ctx.save()
        ctx.setLineDash([4, 4])
        ctx.strokeStyle = '#5f6f8c'
        ctx.beginPath()
        ctx.moveTo(padL, yAt(uniformLoss))
        ctx.lineTo(w - padR, yAt(uniformLoss))
        ctx.stroke()
        ctx.restore()
        ctx.textAlign = 'right'
        ctx.fillStyle = '#5f6f8c'
        ctx.fillText('random guessing', w - padR - 4, yAt(uniformLoss) - 9)
      }

      const pts = history.filter((v) => Number.isFinite(v))
      if (pts.length > 1) {
        // Fill under the curve first so the stroke sits on top of it cleanly.
        const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH)
        grad.addColorStop(0, 'rgba(47,168,160,0.22)')
        grad.addColorStop(1, 'rgba(47,168,160,0)')
        ctx.beginPath()
        ctx.moveTo(xAt(0), padT + plotH)
        history.forEach((v, i) => {
          if (Number.isFinite(v)) ctx.lineTo(xAt(i), yAt(v))
        })
        ctx.lineTo(xAt(history.length - 1), padT + plotH)
        ctx.closePath()
        ctx.fillStyle = grad
        ctx.fill()

        ctx.beginPath()
        let started = false
        history.forEach((v, i) => {
          if (!Number.isFinite(v)) return
          if (!started) {
            ctx.moveTo(xAt(i), yAt(v))
            started = true
          } else {
            ctx.lineTo(xAt(i), yAt(v))
          }
        })
        ctx.strokeStyle = '#2fa8a0'
        ctx.lineWidth = 2
        ctx.lineJoin = 'round'
        ctx.stroke()

        const lastIdx = history.length - 1
        const lastVal = history[lastIdx]
        if (Number.isFinite(lastVal)) {
          ctx.beginPath()
          ctx.arc(xAt(lastIdx), yAt(lastVal), 3.5, 0, Math.PI * 2)
          ctx.fillStyle = '#ff6b4a'
          ctx.fill()
        }
      }

      ctx.fillStyle = '#5f6f8c'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText('0', padL, h - padB + 8)
      ctx.fillText(String(totalSteps), w - padR, h - padB + 8)
      ctx.fillText('training step', padL + plotW / 2, h - padB + 8)
    },
    [history, totalSteps, uniformLoss],
  )

  return (
    <div className="chart">
      <canvas ref={canvasRef} />
    </div>
  )
}
