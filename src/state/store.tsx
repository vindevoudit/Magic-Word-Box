/**
 * Shared state for the whole page.
 *
 * The model is a mutable class instance rather than immutable data, so it is
 * held outside React state and paired with a `modelVersion` counter. Components
 * read the live object and re-render when the counter changes. Copying ~100k
 * float32 weights on every render to satisfy immutability would cost more than
 * the training step that produced them.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { TinyTransformer } from '../model/model'
import { loadInto, type SerializedModel } from '../model/quantize'
import { DEFAULT_PRESET, type Preset } from '../model/presets'
import { DEFAULT_CORPUS } from '../data/corpora'
import { encodePrompt, vocabFromTokens, type Vocab } from '../model/tokenizer'
import type { ProgressMessage, WorkerResponse } from '../worker/protocol'

export type TrainStatus = 'idle' | 'training' | 'done' | 'error'

export interface TrainingState {
  status: TrainStatus
  step: number
  totalSteps: number
  loss: number
  lr: number
  gradNorm: number
  stepsPerSecond: number
  history: number[]
  sample: string
  embedding: { coords: Float32Array; ids: Int32Array } | null
  uniformLoss: number
  elapsedMs: number
  error: string | null
  /** True once the visitor has trained their own model, replacing the shipped one. */
  isOwnModel: boolean
}

const emptyTraining: TrainingState = {
  status: 'idle',
  step: 0,
  totalSteps: 0,
  loss: 0,
  lr: 0,
  gradNorm: 0,
  stepsPerSecond: 0,
  history: [],
  sample: '',
  embedding: null,
  uniformLoss: 0,
  elapsedMs: 0,
  error: null,
  isOwnModel: false,
}

interface Store {
  model: TinyTransformer | null
  vocab: Vocab | null
  modelVersion: number
  corpus: string
  setCorpus: (text: string) => void
  preset: Preset
  setPreset: (p: Preset) => void
  training: TrainingState
  startTraining: () => void
  cancelTraining: () => void
  promptText: string
  setPromptText: (t: string) => void
  promptIds: number[]
  temperature: number
  setTemperature: (t: number) => void
  layer: number
  setLayer: (l: number) => void
  head: number
  setHead: (h: number) => void
}

const StoreContext = createContext<Store | null>(null)

export function useStore(): Store {
  const ctx = useContext(StoreContext)
  if (!ctx) throw new Error('useStore must be used inside StoreProvider')
  return ctx
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const modelRef = useRef<TinyTransformer | null>(null)
  const vocabRef = useRef<Vocab | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const [modelVersion, setModelVersion] = useState(0)

  const [corpus, setCorpus] = useState(DEFAULT_CORPUS.text)
  const [preset, setPreset] = useState<Preset>(DEFAULT_PRESET)
  const [training, setTraining] = useState<TrainingState>(emptyTraining)
  const [promptText, setPromptText] = useState('the little lamp is')
  const [temperature, setTemperature] = useState(0.8)
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)

  // Load the shipped model so every stage has something real to draw before
  // the visitor trains anything of their own.
  useEffect(() => {
    let cancelled = false
    fetch(`${import.meta.env.BASE_URL}pretrained.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`pretrained.json: ${r.status}`)
        return r.json() as Promise<SerializedModel>
      })
      .then((saved) => {
        if (cancelled) return
        const model = new TinyTransformer(saved.config)
        loadInto(model.params, saved)
        modelRef.current = model
        vocabRef.current = vocabFromTokens(saved.vocab)
        setModelVersion((v) => v + 1)
      })
      .catch((err) => {
        if (cancelled) return
        setTraining((t) => ({
          ...t,
          status: 'error',
          error: `Could not load the starter model (${err.message}). Training your own still works.`,
        }))
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => () => workerRef.current?.terminate(), [])

  const startTraining = useCallback(() => {
    workerRef.current?.terminate()
    const worker = new Worker(new URL('../worker/trainer.worker.ts', import.meta.url), {
      type: 'module',
    })
    workerRef.current = worker

    setTraining({
      ...emptyTraining,
      status: 'training',
      totalSteps: preset.steps,
      isOwnModel: true,
    })

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data
      if (msg.type === 'ready') {
        const model = new TinyTransformer(msg.config)
        modelRef.current = model
        vocabRef.current = vocabFromTokens(msg.vocab)
        setModelVersion((v) => v + 1)
        setTraining((t) => ({ ...t, uniformLoss: msg.uniformLoss }))
      } else if (msg.type === 'progress') {
        setTraining((t) => applyProgress(t, msg))
      } else if (msg.type === 'done') {
        const model = modelRef.current
        if (model) {
          loadInto(model.params, {
            version: 1,
            config: model.cfg,
            vocab: vocabRef.current?.tokens ?? [],
            tensors: msg.weights,
          })
          setModelVersion((v) => v + 1)
        }
        setTraining((t) => ({
          ...t,
          status: 'done',
          loss: msg.finalLoss,
          elapsedMs: msg.elapsedMs,
          history: Array.from(msg.lossHistory),
        }))
        worker.terminate()
        workerRef.current = null
      } else if (msg.type === 'error') {
        setTraining((t) => ({ ...t, status: 'error', error: msg.message }))
        worker.terminate()
        workerRef.current = null
      }
    }

    worker.postMessage({
      type: 'train',
      text: corpus,
      dModel: preset.dModel,
      nLayers: preset.nLayers,
      nHeads: preset.nHeads,
      contextLength: preset.contextLength,
      batchSize: preset.batchSize,
      steps: preset.steps,
      lr: preset.lr,
      maxVocab: preset.maxVocab,
      seed: 1234,
    })
  }, [corpus, preset])

  const cancelTraining = useCallback(() => {
    workerRef.current?.postMessage({ type: 'cancel' })
    setTraining((t) => ({ ...t, status: t.status === 'training' ? 'done' : t.status }))
  }, [])

  // Keep the selected layer and head inside range when the model changes shape.
  useEffect(() => {
    const m = modelRef.current
    if (!m) return
    setLayer((l) => Math.min(l, m.cfg.nLayers - 1))
    setHead((h) => Math.min(h, m.cfg.nHeads - 1))
  }, [modelVersion])

  const promptIds = useMemo(() => {
    const v = vocabRef.current
    if (!v) return []
    return encodePrompt(promptText, v)
  }, [promptText, modelVersion])

  const value: Store = {
    model: modelRef.current,
    vocab: vocabRef.current,
    modelVersion,
    corpus,
    setCorpus,
    preset,
    setPreset,
    training,
    startTraining,
    cancelTraining,
    promptText,
    setPromptText,
    promptIds,
    temperature,
    setTemperature,
    layer,
    setLayer,
    head,
    setHead,
  }

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>
}

function applyProgress(t: TrainingState, msg: ProgressMessage): TrainingState {
  const history = t.history.slice()
  history[msg.step] = msg.loss
  return {
    ...t,
    step: msg.step,
    totalSteps: msg.totalSteps,
    loss: msg.loss,
    lr: msg.lr,
    gradNorm: msg.gradNorm,
    stepsPerSecond: msg.stepsPerSecond,
    history,
    sample: msg.sample ?? t.sample,
    embedding: msg.embedding ?? t.embedding,
  }
}
