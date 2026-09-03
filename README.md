# Magic Word Box

An interactive site that teaches how a language model works by having you build
one. Paste in your own text, train a real transformer on it in your browser, and
watch it guess the next word with the attention weights, embeddings and loss
curve all visible while it learns.

**Everything runs client-side.** There is no backend, no API key, and no text
ever leaves the browser.

## Why the model runs in the browser

The site is designed to be hosted on Render, which set the architecture:

|                | Free web service       | Free static site |
| -------------- | ---------------------- | ---------------- |
| RAM / CPU      | 512 MB / 0.1 CPU       | n/a, CDN-served  |
| Idle behaviour | spins down after 15 min| never            |
| Cold start     | 30-60s                 | none             |

A server-side trainer was never viable: PyTorch alone exceeds the memory and
disk budget, 0.1 CPU cannot train even a toy model at interactive speed, and
concurrent visitors would queue behind one another. Training in the browser is
better on every axis that matters here, and it means every intermediate tensor
is available to draw, which is the entire point of the site.

## The model

A decoder-only transformer written from scratch in TypeScript, forward *and*
backward pass, with no machine-learning library:

- learned token and position embeddings
- pre-norm blocks: multi-head causal self-attention, then a 4x GELU feed-forward
- residual connections, final layer norm, linear head over the vocabulary
- Adam with bias correction, global-norm gradient clipping, warmup plus cosine decay
- word-level tokenizer, vocabulary capped by frequency with an unknown token

Training runs in a Web Worker, so the page never drops a frame. On a laptop the
default preset finishes **about 400 steps in under four seconds**.

## Correctness

The backward pass is hand-derived, so it is tested rather than trusted. A wrong
gradient still trains *something* — just to a worse model — and the site would
then be teaching a lie with a straight face.

```bash
npm test
```

- `tests/gradcheck.test.ts` — every parameter tensor is checked against a central
  finite difference. Verified to fail loudly when a gradient is deliberately broken.
- `tests/overfit.test.ts` — trains until it memorises a sentence and reproduces it.
- `tests/attention.test.ts` — causality, normalisation, and the exact claim stage 5
  makes on screen, including the case where that claim stops being true.
- `tests/tokenizer.test.ts` — round-trips, frequency capping, batch construction.

## Development

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # the suite above
npm run typecheck
npm run build      # -> dist/
```

### Regenerating the shipped model

`public/pretrained.json` is a small int8-quantised model, trained offline and
committed. It exists so every visualisation is live and real from the first
scroll, before the visitor has trained anything of their own. Rerun it whenever
the model code or the default corpus changes:

```bash
npm run pretrain
```

It is deliberately **not** part of `npm run build`, so the Render build stays a
plain `vite build` with no training step and no devDependency surprises.

## Deploying to Render

The repository contains a `render.yaml` blueprint. Point Render at the repo and
it creates a static site with:

- build command `npm ci && npm run build`
- publish directory `dist`
- an SPA rewrite and long-lived caching for hashed assets

Free, CDN-served, no spin-down, no cold start.

## Layout

```
src/model/     the transformer: tensor ops, attention, blocks, Adam, tokenizer
src/worker/    training loop, off the main thread
src/viz/       canvas and SVG visuals, plus the shared probability ramp
src/stages/    the eleven stages, in forward-pass order
src/state/     shared store and inference hooks
scripts/       offline pretraining
tests/         gradient checks and end-to-end learning tests
```

`src/model/attention.ts` is displayed on the site itself, read from source at
build time, so the code shown can never drift from the code that runs.
