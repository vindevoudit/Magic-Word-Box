/**
 * The probability ramp, shared by every visual on the page.
 *
 * One scale for one meaning: how much of something there is, from 0 to 1. An
 * attention weight, a token probability and a gradient magnitude all colour the
 * same way, so a reader who learns the ramp once can read every picture.
 */

const STOPS: [number, number, number][] = [
  [0x1b, 0x2a, 0x4a],
  [0x2c, 0x6e, 0x9b],
  [0x2f, 0xa8, 0xa0],
  [0x8f, 0xd1, 0x6a],
  [0xf2, 0xc7, 0x44],
]

/** Map t in [0,1] to the ramp. Values outside the range clamp. */
export function ramp(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1)
  const i = Math.min(STOPS.length - 2, Math.floor(x))
  const f = x - i
  const a = STOPS[i]
  const b = STOPS[i + 1]
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ]
}

export function rampCss(t: number, alpha = 1): string {
  const [r, g, b] = ramp(t)
  return alpha >= 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`
}

/**
 * Perceptual boost for small values.
 *
 * Attention rows are often dominated by one position, leaving everything else
 * near zero and visually black. A square-root emphasis keeps the weak-but-real
 * connections legible without misrepresenting the ordering.
 */
export function emphasise(t: number): number {
  return Math.sqrt(Math.max(0, Math.min(1, t)))
}
