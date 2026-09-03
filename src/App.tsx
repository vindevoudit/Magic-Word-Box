import { useMemo } from 'react'
import { StoreProvider, useStore } from './state/store'
import { ProgressRail } from './nav/ProgressRail'
import { AmbientGrid } from './viz/AmbientGrid'
import { Hero } from './stages/Hero'
import { GuessingGame } from './stages/GuessingGame'
import { Corpus } from './stages/Corpus'
import { Tokenization } from './stages/Tokenization'
import { Embeddings } from './stages/Embeddings'
import { Position } from './stages/Position'
import { Attention } from './stages/Attention'
import { Block } from './stages/Block'
import { Training } from './stages/Training'
import { Playground } from './stages/Playground'
import { UnderTheHood } from './stages/UnderTheHood'
import { WhatScaleBuys } from './stages/WhatScaleBuys'

const RAIL = [
  { id: 'guess', label: 'The guess' },
  { id: 'corpus', label: 'Your text' },
  { id: 'tokens', label: 'Tokens' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'position', label: 'Position' },
  { id: 'attention', label: 'Attention' },
  { id: 'block', label: 'The block' },
  { id: 'training', label: 'Training' },
  { id: 'playground', label: 'Playground' },
  { id: 'code', label: 'The code' },
  { id: 'scale', label: 'Scale' },
]

/** Real attention weights from the loaded model, for the page background. */
function useAmbientProbs() {
  const { model, modelVersion } = useStore()
  return useMemo(() => {
    if (!model) return { probs: null, n: 0 }
    const T = model.cfg.contextLength
    const ids = Array.from({ length: Math.min(T, 10) }, (_, i) => (i % 12) + 3)
    const { act } = model.predictNext(ids)
    const n = Math.min(ids.length, T)
    const head = act.layers[0].attn.probs.subarray(0, T * T)
    const out = new Float32Array(n * n)
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) out[r * n + c] = head[r * T + c]
    }
    return { probs: out, n }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, modelVersion])
}

function Page() {
  const { probs, n } = useAmbientProbs()

  return (
    <>
      <AmbientGrid probs={probs} n={n} />
      <a className="skip" href="#guess">
        Skip to the lesson
      </a>
      <div className="layout">
        <ProgressRail items={RAIL} />
        <main className="content">
          <Hero />
          <GuessingGame />
          <Corpus />
          <Tokenization />
          <Embeddings />
          <Position />
          <Attention />
          <Block />
          <Training />
          <Playground />
          <UnderTheHood />
          <WhatScaleBuys />
          <footer className="footer">
            <p className="muted">
              Everything here runs in your browser. No text you paste is ever
              sent anywhere.
            </p>
          </footer>
        </main>
      </div>
    </>
  )
}

export default function App() {
  return (
    <StoreProvider>
      <Page />
    </StoreProvider>
  )
}
