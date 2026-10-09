import { useEffect, useMemo, useState } from 'react'
import { CountdownOverlay, LandmarkView } from '../components/LandmarkView'
import type { Prediction } from '../ml/classifier'
import { DOM_OFF, F, frameFeatures, NONDOM_OFF, resample } from '../ml/features'
import { downloadJson, measuredFps, toRawSequence, type Recording } from '../ml/sequence'
import type { Delegate } from '../mp/landmarker'
import { countdown, useLandmarkStream, type Source } from '../mp/useLandmarkStream'
import { useClassifier, useClipManifest } from './shared'

const RECORD_MS = 3000
// Measured on the FSL-105 training clips of the 24 classes (ml/features.py trimming).
const DATASET_TRIM_S = { p5: 1.23, p50: 1.6, p95: 2.31 }
const DATASET_ENTRY_WRIST_Y = 0.79 // median dominant-wrist y (shoulder widths) at first/last hand frame

interface DebugResult {
  pred: Prediction
  mpMsPerFrame: number
  handMs: number
  poseMs: number
  featuresMs: number
  rawFrames: number
  fps: number
  trimmedFrames: number
  trim: [number, number]
  dominant: string
  handsAtStart: boolean
  handsAtEnd: boolean
  handFrac: number
  wristYStart: number | null
  wristYEnd: number | null
  width: number
  height: number
  recording: Recording
}

declare global {
  interface Window {
    __kumpas?: { ready: boolean; busy: boolean; result?: Omit<DebugResult, 'recording' | 'pred'> & { top3: { gloss: string; prob: number }[]; modelMs: number } }
  }
}

function analyze(rec: Recording) {
  const raw = toRawSequence(rec)
  const t0 = performance.now()
  const { frames, meta } = frameFeatures(raw)
  const features = Float32Array.from(resample(frames).flat())
  const featuresMs = performance.now() - t0
  const anyHand = (f: number[]) => f[DOM_OFF + 63] > 0 || f[NONDOM_OFF + 63] > 0
  const wristY = (f: number[] | undefined) => (f && f[DOM_OFF + 63] > 0 ? f[DOM_OFF + 1] : null)
  const n = rec.frames.length
  return {
    features,
    featuresMs,
    partial: {
      mpMsPerFrame: rec.frames.reduce((s, f) => s + f.handMs + f.poseMs, 0) / Math.max(n, 1),
      handMs: rec.frames.reduce((s, f) => s + f.handMs, 0) / Math.max(n, 1),
      poseMs: rec.frames.reduce((s, f) => s + f.poseMs, 0) / Math.max(n, 1),
      featuresMs,
      rawFrames: n,
      fps: measuredFps(rec.frames),
      trimmedFrames: frames.length,
      trim: meta.trim,
      dominant: meta.dominant,
      handsAtStart: n > 0 && rec.frames[0].hand_present.some(Boolean),
      handsAtEnd: n > 0 && rec.frames[n - 1].hand_present.some(Boolean),
      handFrac: frames.filter(anyHand).length / Math.max(frames.length, 1),
      wristYStart: wristY(frames.find((f) => f[DOM_OFF + 63] > 0)),
      wristYEnd: wristY([...frames].reverse().find((f) => f[DOM_OFF + 63] > 0)),
      width: rec.width,
      height: rec.height,
      recording: rec,
    },
  }
}

export function Debug() {
  const params = new URLSearchParams(location.search)
  const delegate = (params.get('delegate') === 'GPU' ? 'GPU' : 'CPU') as Delegate
  const clips = useClipManifest()
  const [srcUrl, setSrcUrl] = useState(params.get('src') ?? '')
  const source: Source = useMemo(() => (srcUrl ? { kind: 'clip', url: srcUrl } : { kind: 'camera' }), [srcUrl])
  const stream = useLandmarkStream(source, delegate)
  const classifier = useClassifier()
  const [count, setCount] = useState<number | null>(null)
  const [phase, setPhase] = useState<'idle' | 'countdown' | 'recording' | 'processing'>('idle')
  const [result, setResult] = useState<DebugResult>()
  const ready = stream.status === 'ready' && !!classifier

  const run = async () => {
    if (!classifier) return
    window.__kumpas = { ready, busy: true }
    setResult(undefined)
    setPhase('countdown')
    await countdown(setCount, params.get('auto') ? 1 : 3)
    setPhase('recording')
    const rec = await stream.record(RECORD_MS)
    setPhase('processing')
    const { features, partial } = analyze(rec)
    const pred = await classifier.predict(features)
    const r = { ...partial, pred }
    setResult(r)
    setPhase('idle')
    const { recording: _rec, pred: _p, ...summary } = r
    window.__kumpas = {
      ready: true,
      busy: false,
      result: { ...summary, modelMs: pred.ms, top3: pred.ranked.slice(0, 3).map((x) => ({ gloss: x.cls.gloss, prob: x.prob })) },
    }
  }

  useEffect(() => {
    window.__kumpas = { ...window.__kumpas, ready, busy: phase !== 'idle' }
    if (ready && params.get('auto') && !window.__kumpas.result && phase === 'idle') void run()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready])

  const trimmedS = result ? result.trimmedFrames / result.fps : 0
  const idleSuspect = result && (trimmedS > DATASET_TRIM_S.p95 || (result.handsAtStart && result.handsAtEnd))

  return (
    <div className="page">
      <h1>Debug</h1>
      <div className="toolbar">
        <label>
          Source{' '}
          <select value={srcUrl} onChange={(e) => setSrcUrl(e.target.value)} disabled={phase !== 'idle'}>
            <option value="">Camera</option>
            {clips.map((c) => (
              <option key={c.file} value={`/clips/${c.file}`}>
                Reference clip: {c.gloss}
              </option>
            ))}
            {srcUrl && !clips.some((c) => `/clips/${c.file}` === srcUrl) && <option value={srcUrl}>{srcUrl}</option>}
          </select>
        </label>
        <button className="primary" onClick={run} disabled={!ready || phase !== 'idle'}>
          {phase === 'idle' ? 'Record' : phase === 'processing' ? 'Processing…' : phase === 'countdown' ? 'Get ready…' : 'Recording…'}
        </button>
        <span className="muted">
          {source.kind === 'camera' ? `${RECORD_MS / 1000} s from camera` : 'whole clip'} · MediaPipe {delegate} ·{' '}
          {stream.perf.width}×{stream.perf.height} · live {stream.perf.fps.toFixed(0)} fps · hand{' '}
          {stream.perf.handMs.toFixed(1)} ms + pose {stream.perf.poseMs.toFixed(1)} ms
        </span>
      </div>

      <div className="debug-grid">
        <LandmarkView stream={stream}>
          <CountdownOverlay count={count} recording={phase === 'recording'} />
        </LandmarkView>

        <div className="panel">
          {!result && <p className="muted">{classifier ? 'Press Record, then sign after the countdown.' : 'Loading model…'}</p>}
          {result && (
            <>
              <h2>Top 3</h2>
              {result.pred.ranked.slice(0, 3).map(({ cls, prob }) => (
                <div key={cls.id} className="bar-row">
                  <span className="bar-label">{cls.gloss}</span>
                  <div className="bar">
                    <div style={{ width: `${prob * 100}%` }} />
                  </div>
                  <span className="bar-val">{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}

              <h2>Timings</h2>
              <table>
                <tbody>
                  <tr><td>MediaPipe (hand + pose)</td><td>{result.mpMsPerFrame.toFixed(1)} ms/frame</td></tr>
                  <tr><td className="indent">hand / pose</td><td>{result.handMs.toFixed(1)} / {result.poseMs.toFixed(1)} ms</td></tr>
                  <tr><td>Features ({result.rawFrames} → 32 frames)</td><td>{result.featuresMs.toFixed(2)} ms</td></tr>
                  <tr><td>Model inference (ONNX, WASM)</td><td>{result.pred.ms.toFixed(2)} ms</td></tr>
                </tbody>
              </table>

              <h2>Trim check</h2>
              <table>
                <tbody>
                  <tr><td>Recorded</td><td>{result.rawFrames} frames @ {result.fps.toFixed(1)} fps ({(result.rawFrames / result.fps).toFixed(2)} s), {result.width}×{result.height}</td></tr>
                  <tr><td>Trimmed (hands ± {5} frames)</td><td>frames {result.trim[0]}–{result.trim[1]} = <b>{trimmedS.toFixed(2)} s</b></td></tr>
                  <tr><td>FSL-105 trimmed length</td><td>median {DATASET_TRIM_S.p50} s (5–95%: {DATASET_TRIM_S.p5}–{DATASET_TRIM_S.p95} s)</td></tr>
                  <tr><td>Hands visible at first / last frame</td><td>{result.handsAtStart ? 'yes' : 'no'} / {result.handsAtEnd ? 'yes' : 'no'}</td></tr>
                  <tr><td>Frames with a hand (after trim)</td><td>{(result.handFrac * 100).toFixed(0)}% (dataset ≈ 89%)</td></tr>
                  <tr><td>Dominant wrist y at start / end</td><td>{fmt(result.wristYStart)} / {fmt(result.wristYEnd)} shoulder widths (dataset ≈ {DATASET_ENTRY_WRIST_Y})</td></tr>
                  <tr><td>Dominant hand</td><td>{result.dominant}</td></tr>
                </tbody>
              </table>
              {idleSuspect && (
                <p className="warn">
                  Likely includes idle frames: hands were visible at {result.handsAtStart && result.handsAtEnd ? 'both ends' : 'one end'} of
                  the recording, so trimming could not cut them off. FSL-105 clips start and end with hands out of frame.
                </p>
              )}
              <button onClick={() => downloadJson(`debug_${Date.now()}.json`, { ...toRawSequence(result.recording), features_dim: F })}>
                Download raw landmarks
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

const fmt = (v: number | null) => (v === null ? '–' : v.toFixed(2))
