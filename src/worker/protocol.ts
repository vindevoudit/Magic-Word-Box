/**
 * Messages between the page and the training worker.
 *
 * Progress messages deliberately carry a *summary* rather than the weights:
 * loss, a sample generation, and 2D embedding coordinates. Posting ~250 KB of
 * float32 ten times a second would cost more than the training step itself.
 * Full weights cross once, when training finishes.
 */

import type { ModelConfig } from '../model/model'

export interface TrainRequest {
  type: 'train'
  text: string
  dModel: number
  nLayers: number
  nHeads: number
  contextLength: number
  batchSize: number
  steps: number
  lr: number
  maxVocab: number
  seed: number
}

export interface CancelRequest {
  type: 'cancel'
}

export type WorkerRequest = TrainRequest | CancelRequest

export interface ReadyMessage {
  type: 'ready'
  vocab: string[]
  tokenCount: number
  windowCount: number
  distinctWords: number
  unkOccurrences: number
  paramCount: number
  uniformLoss: number
  config: ModelConfig
}

export interface ProgressMessage {
  type: 'progress'
  step: number
  totalSteps: number
  loss: number
  lr: number
  gradNorm: number
  stepsPerSecond: number
  /** Only present on sample steps, to keep messages small. */
  sample?: string
  /** Flat [n, 2] PCA coordinates of the token embeddings, plus their ids. */
  embedding?: { coords: Float32Array; ids: Int32Array }
}

export interface DoneMessage {
  type: 'done'
  /** Serialized weights, ready to rebuild the model on the page side. */
  weights: { name: string; scale: number; b64: string }[]
  finalLoss: number
  elapsedMs: number
  lossHistory: Float32Array
}

export interface ErrorMessage {
  type: 'error'
  message: string
}

export type WorkerResponse = ReadyMessage | ProgressMessage | DoneMessage | ErrorMessage
