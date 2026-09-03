import { useMemo, useState } from 'react'
import { Panel, Stage } from './Stage'
// Imported from the actual module, so what is displayed is what runs. If the
// implementation changes, this stage changes with it and cannot go stale.
import attentionSource from '../model/attention.ts?raw'

/** Pull one function out of the real source file by its signature. */
function extract(src: string, startMarker: string, endMarker: string): string {
  const a = src.indexOf(startMarker)
  if (a < 0) return src.slice(0, 2000)
  const b = src.indexOf(endMarker, a)
  return src.slice(a, b < 0 ? undefined : b).trimEnd()
}

const HIGHLIGHT =
  /(\/\/[^\n]*)|(\/\*[\s\S]*?\*\/)|\b(const|let|for|if|continue|function|return|export|interface|void|number)\b|\b(\d+(?:\.\d+)?)\b/g

function highlight(code: string) {
  const out: React.ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  HIGHLIGHT.lastIndex = 0
  while ((m = HIGHLIGHT.exec(code))) {
    if (m.index > last) out.push(code.slice(last, m.index))
    const cls = m[1] || m[2] ? 'c-comment' : m[3] ? 'c-key' : 'c-num'
    out.push(
      <span key={m.index} className={cls}>
        {m[0]}
      </span>,
    )
    last = m.index + m[0].length
  }
  out.push(code.slice(last))
  return out
}

const MATH = [
  {
    label: 'Scores',
    tex: 'S = Q K\u1d40 / \u221ad\u2096',
    plain:
      'Compare every query against every key. Divide by the square root of the head width so the numbers do not grow with size.',
  },
  {
    label: 'Mask',
    tex: 'S\u1d62\u2c7c = \u2212\u221e  for j > i',
    plain: 'Erase everything a word is not allowed to see: its own future.',
  },
  {
    label: 'Weights',
    tex: 'A = softmax(S)',
    plain: 'Turn the scores into percentages that add up to one, row by row.',
  },
  {
    label: 'Output',
    tex: 'O = A V',
    plain: 'Mix the earlier words together in exactly those proportions.',
  },
]

export function UnderTheHood() {
  const [tab, setTab] = useState<'forward' | 'backward'>('forward')

  const code = useMemo(() => {
    if (tab === 'forward') {
      const fn = extract(
        attentionSource,
        'export function attentionForward(',
        'export interface AttentionGrads',
      )
      // Trim to the loop that is actually worth reading.
      const from = fn.indexOf('  for (let b = 0')
      const to = fn.indexOf('  matmul(out, cache.ctx')
      return from >= 0 && to > from ? fn.slice(from, to).trimEnd() : fn
    }
    const fn = extract(attentionSource, 'export function attentionBackward(', '\n}\n')
    return fn.slice(0, 2600)
  }, [tab])

  return (
    <Stage
      id="code"
      index={10}
      kicker="the implementation"
      title="The actual code"
      lede={
        <>
          Everything on this page runs on the code below &mdash; no machine
          learning library, no server. This is read straight out of the source
          file at build time, so it cannot drift from what is really executing.
        </>
      }
    >
      <Panel wide title="Multi-head causal self-attention">
        <div className="segmented segmented--wide">
          <button
            type="button"
            className={tab === 'forward' ? 'is-active' : ''}
            onClick={() => setTab('forward')}
          >
            Forward
          </button>
          <button
            type="button"
            className={tab === 'backward' ? 'is-active' : ''}
            onClick={() => setTab('backward')}
          >
            Backward
          </button>
        </div>
        <pre className="code">
          <code>{highlight(code)}</code>
        </pre>
      </Panel>

      <Panel title="The same thing, in four lines of maths">
        <ol className="mathlist">
          {MATH.map((m) => (
            <li key={m.label}>
              <span className="eyebrow">{m.label}</span>
              <b className="mono mathlist__tex">{m.tex}</b>
              <span className="muted">{m.plain}</span>
            </li>
          ))}
        </ol>
      </Panel>
    </Stage>
  )
}
