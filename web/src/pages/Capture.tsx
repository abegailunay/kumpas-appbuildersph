import { useMemo, useState } from 'react'
import { CountdownOverlay, LandmarkView } from '../components/LandmarkView'
import { extractFeatures, SPEC_VERSION } from '../ml/features'
import { downloadJson, measuredFps, toExtractRecord, toRawSequence, type Recording } from '../ml/sequence'
import { countdown, sleep, useLandmarkStream } from '../mp/useLandmarkStream'
import { useClassifier } from './shared'

const RECORD_MS = 3000

interface Attempt {
  rec: Recording
  clipId: string
  top1: string
  prob: number
}

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15)
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')

/** Dev tool: record N attempts of one sign and download raw landmarks for fine-tuning (ml/own_samples/). */
export function Capture() {
  const stream = useLandmarkStream(useMemo(() => ({ kind: 'camera' as const }), []))
  const classifier = useClassifier()
  const [classId, setClassId] = useState(0)
  const [n, setN] = useState(5)
  const [attempts, setAttempts] = useState<Attempt[]>([])
  const [count, setCount] = useState<number | null>(null)
  const [busy, setBusy] = useState<'idle' | 'countdown' | 'recording'>('idle')
  const [progress, setProgress] = useState('')
  const cls = classifier?.classes[classId]

  const session = async () => {
    if (!classifier || !cls) return
    for (let i = 0; i < n; i++) {
      setProgress(`Attempt ${i + 1} of ${n}: lower your hands, then sign ${cls.gloss} after the countdown`)
      setBusy('countdown')
      await countdown(setCount)
      setBusy('recording')
      const rec = await stream.record(RECORD_MS)
      const pred = await classifier.predict(extractFeatures(toRawSequence(rec)).features)
      const top = pred.ranked[0]
      const clipId = `own_${cls.dataset_label_id}_${stamp()}_${i}`
      setAttempts((a) => [...a, { rec, clipId, top1: top.cls.gloss, prob: top.prob }])
      setBusy('idle')
      if (i < n - 1) await sleep(1200)
    }
    setProgress('')
  }

  const download = () => {
    if (!cls) return
    downloadJson(`own_${slug(cls.gloss)}_${stamp()}.json`, {
      format: 'kumpas-capture-v1',
      note: 'Each attempt uses the same keys as ml/extract.py .npz files. Put this file in ml/own_samples/.',
      spec_version: SPEC_VERSION,
      created: new Date().toISOString(),
      user_agent: navigator.userAgent,
      sign: { class_id: cls.id, dataset_label_id: cls.dataset_label_id, gloss: cls.gloss },
      attempts: attempts.map((a) => toExtractRecord(a.rec, a.clipId, cls.dataset_label_id)),
    })
  }

  return (
    <div className="page">
      <h1>Capture</h1>
      <p className="muted">
        Records raw landmarks for fine-tuning. Start and end each attempt with your hands lowered out of frame, like the
        FSL-105 clips.
      </p>
      <div className="toolbar">
        <label>
          Sign{' '}
          <select
            value={classId}
            disabled={busy !== 'idle' || !!progress}
            onChange={(e) => {
              setClassId(+e.target.value)
              setAttempts([])
            }}
          >
            {classifier?.classes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.gloss} ({c.filipino})
              </option>
            ))}
          </select>
        </label>
        <label>
          Attempts <input type="number" min={1} max={30} value={n} onChange={(e) => setN(Math.max(1, +e.target.value))} />
        </label>
        <button className="primary" onClick={session} disabled={stream.status !== 'ready' || !classifier || !!progress}>
          Record {n}
        </button>
        <button onClick={download} disabled={!attempts.length || !!progress}>
          Download {attempts.length} as JSON
        </button>
      </div>
      {progress && <p className="progress">{progress}</p>}

      <div className="debug-grid">
        <LandmarkView stream={stream}>
          <CountdownOverlay count={count} recording={busy === 'recording'} />
        </LandmarkView>
        <div className="panel">
          <h2>Attempts for {cls?.gloss ?? '…'}</h2>
          {!attempts.length && <p className="muted">None yet.</p>}
          <ol className="attempts">
            {attempts.map((a, i) => {
              const handFrames = a.rec.frames.filter((f) => f.hand_present.some(Boolean)).length
              return (
                <li key={a.clipId}>
                  {a.rec.frames.length} frames @ {measuredFps(a.rec.frames).toFixed(0)} fps · hands in{' '}
                  {Math.round((100 * handFrames) / Math.max(a.rec.frames.length, 1))}% · model: {a.top1}{' '}
                  {(a.prob * 100).toFixed(0)}%{' '}
                  <button className="link" onClick={() => setAttempts((x) => x.filter((_, j) => j !== i))}>
                    delete
                  </button>
                </li>
              )
            })}
          </ol>
        </div>
      </div>
    </div>
  )
}
